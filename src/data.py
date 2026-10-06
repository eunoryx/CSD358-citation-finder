"""
Data loading for the citation finder.

Dataset: AMiner DBLP-Citation-network V19 (JSON Lines: one paper per line).
Each line looks like:
  {"id": "...", "title": "...", "abstract": "...", "year": 2015, "venue": "...",
   "references": ["id1", "id2", ...], "n_citation": 46, ...}

The file is ~18 GB, so we never load it all at once. We STREAM it line by line
(a generator), which keeps memory use flat no matter how big the file is.

Building a working subset
-------------------------
A random sample does not work for citation search: in a random 3k sample only
52 of 23,135 references pointed to another paper in the sample, so we would have
almost no ground truth. Instead we pick one *connected region* of the citation graph:

  Pass 1: stream the corpus, keep "seed" papers = papers from chosen venues
          (IR / NLP / web conferences) in a year range. Remember what they cite.
  Pass 2: stream again, keep seed papers + papers cited by at least
          `min_cited_by` seeds. Because the extra papers are exactly the ones the
          seeds cite, most seed references resolve inside the subset.

Usage:
  python -m src.data subset            # build data/subset.jsonl with default settings
  python -m src.data subset --max-papers 100000 --min-cited-by 1
  python -m src.data stats data/subset.jsonl
"""
import argparse
import collections
import io
import json
import os
import re
import time
import zipfile

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_JSONL = os.path.join(DATA_DIR, "DBLP-Citation-network-V19.jsonl")
RAW_ZIP = os.path.join(DATA_DIR, "DBLP-Citation-network-V19.zip")
SUBSET = os.path.join(DATA_DIR, "subset.jsonl")

# Fields we keep. Everything else (authors, doi, pages...) is dropped to save space.
KEEP = ("id", "title", "abstract", "year", "venue", "references", "n_citation")

# Seed venues: the main IR / web search / data-mining conferences. DBLP venue strings
# are messy ("SIGIR", "SIGIR'17 PROCEEDINGS OF THE 40TH INTERNATIONAL ACM SIGIR ..."),
# so we match with regular expressions on word boundaries. We checked the matches on
# the full V19 corpus (2012-2024): ~19k seed papers. (Adding ACL/EMNLP/NAACL gave
# 53k seeds, too many for a fast working subset.)
SEED_VENUE_PATTERNS = [
    r"\bSIGIR\b", r"\bCIKM\b", r"Information (&|and) Knowledge Management",
    r"\bWSDM\b", r"Web Search and Data Mining", r"\bECIR\b", r"\bICTIR\b",
    r"\bWWW\b", r"World Wide Web Conference", r"\bWeb Conference\b",
    r"\bSIGKDD\b", r"\bKDD\b",
]
# Venue strings that match the patterns above but are NOT the main conference:
# newsletters, companion/poster volumes, workshops, and look-alikes (PAKDD is
# "Advances in Knowledge Discovery and Data Mining").
EXCLUDE_VENUE_PATTERNS = [
    r"Forum", r"Companion", r"Workshop", r"@", r"Advances in Knowledge Discovery",
    r"Trends and Applications", r"PAKDD", r"MDM/KDD", r"IADIS",
    r"Journal of Information",  # J. of Information & Knowledge Management is not CIKM
]


def open_raw(path=None):
    """Return a text stream over the raw JSONL, reading from the .jsonl if it
    exists, otherwise straight out of the .zip (no unzip needed)."""
    path = path or (RAW_JSONL if os.path.exists(RAW_JSONL) else RAW_ZIP)
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        name = zf.namelist()[0]
        return io.TextIOWrapper(zf.open(name), encoding="utf-8")
    return open(path, encoding="utf-8")


def iter_papers(path=None, require_abstract=True):
    """Stream papers one at a time as small dicts with only the fields we use.

    Drops papers with no abstract (nothing to index) or no year (the year filter
    needs it). Bad JSON lines are counted and skipped, not fatal.
    """
    bad = 0
    with open_raw(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            abstract = (r.get("abstract") or "").strip()
            if require_abstract and not abstract:
                continue
            if not r.get("year"):
                continue
            yield {
                "id": r["id"],
                "title": (r.get("title") or "").strip(),
                "abstract": abstract,
                "year": int(r["year"]),
                "venue": (r.get("venue") or "").strip(),
                "references": r.get("references") or [],
                "n_citation": int(r.get("n_citation") or 0),
            }
    if bad:
        print(f"[iter_papers] skipped {bad} malformed lines")


def build_subset(year_min, year_max, min_cited_by, max_papers, out_path=SUBSET, path=None, full=False):
    """Two streaming passes -> one connected region of the citation graph.
    full=True keeps EVERY paper with an abstract + year (seeds are still marked, for evaluation)."""
    venue_re = re.compile("|".join(SEED_VENUE_PATTERNS), re.IGNORECASE)
    exclude_re = re.compile("|".join(EXCLUDE_VENUE_PATTERNS), re.IGNORECASE)

    # ---- Pass 1: find seed papers and count how often each id is cited by seeds
    t0 = time.time()
    seeds = set()
    cited_by_seeds = collections.Counter()
    matched_venues = collections.Counter()
    n_seen = 0
    for p in iter_papers(path):
        n_seen += 1
        if n_seen % 500_000 == 0:
            print(f"  pass 1: {n_seen:,} papers read, {len(seeds):,} seeds ({time.time()-t0:.0f}s)")
        if (year_min <= p["year"] <= year_max and venue_re.search(p["venue"])
                and not exclude_re.search(p["venue"])):
            seeds.add(p["id"])
            matched_venues[p["venue"]] += 1
            cited_by_seeds.update(set(p["references"]))
    print(f"pass 1 done: {n_seen:,} papers with abstract+year, {len(seeds):,} seeds ({time.time()-t0:.0f}s)")
    print("top matched venue strings (check for false matches):")
    for v, c in matched_venues.most_common(25):
        print(f"  {c:6d}  {v[:90]}")

    # Non-seed papers we will pull in: the ones seeds cite most, up to the cap.
    budget = max(0, max_papers - len(seeds))
    extra = [pid for pid, c in cited_by_seeds.most_common() if c >= min_cited_by and pid not in seeds]
    extra = set(extra[:budget])
    keep = seeds | extra
    if full:
        print("--full: keeping every paper with abstract + year")
    print(f"keeping {len(seeds):,} seeds + {len(extra):,} cited papers (min_cited_by={min_cited_by}, cap={max_papers:,})")

    # ---- Pass 2: write the kept papers (only those that actually have an abstract)
    t0 = time.time()
    n_out = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in iter_papers(path):
            if full or p["id"] in keep:
                p["is_seed"] = p["id"] in seeds
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"pass 2 done: wrote {n_out:,} papers to {out_path} ({time.time()-t0:.0f}s)")
    return out_path


def load_jsonl(path=SUBSET):
    """Load a (small) subset file fully into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def report(papers):
    """Corpus statistics: how much of the citation ground truth is usable."""
    ids = {p["id"] for p in papers}
    n = len(papers)
    with_refs = sum(1 for p in papers if p["references"])
    total_refs = sum(len(p["references"]) for p in papers)
    resolved_per_paper = [sum(r in ids for r in p["references"]) for p in papers]
    resolved = sum(resolved_per_paper)
    years = sorted(p["year"] for p in papers)
    print(f"papers:                         {n:,}")
    print(f"year range:                     {years[0]}-{years[-1]} (median {years[n//2]})")
    print(f"papers with >=1 reference:      {with_refs:,} ({with_refs/n:.1%})")
    print(f"references total:               {total_refs:,}")
    print(f"references resolving in corpus: {resolved:,} ({resolved/max(total_refs,1):.1%})")
    for k in (1, 5, 10, 20):
        m = sum(1 for x in resolved_per_paper if x >= k)
        print(f"papers with >={k:2d} resolved refs:  {m:,}")
    if "is_seed" in papers[0]:
        seeds = [p for p, x in zip(papers, resolved_per_paper) if p["is_seed"]]
        s_tot = sum(len(p["references"]) for p in seeds)
        s_res = sum(sum(r in ids for r in p["references"]) for p in seeds)
        print(f"seed papers:                    {len(seeds):,}; their refs resolving: {s_res:,}/{s_tot:,} ({s_res/max(s_tot,1):.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset", help="build a connected working subset")
    s.add_argument("--year-min", type=int, default=2012)
    s.add_argument("--year-max", type=int, default=2024)
    s.add_argument("--min-cited-by", type=int, default=2)
    s.add_argument("--max-papers", type=int, default=50_000)
    s.add_argument("--out", default=SUBSET)
    s.add_argument("--raw", default=None, help="path to raw .jsonl or .zip")
    s.add_argument("--full", action="store_true", help="whole corpus (~7M papers) instead of a subset")
    st = sub.add_parser("stats", help="print corpus statistics for a jsonl file")
    st.add_argument("path", nargs="?", default=SUBSET)
    pk = sub.add_parser("peek", help="print the first N raw records")
    pk.add_argument("-n", type=int, default=3)
    a = ap.parse_args()

    if a.cmd == "subset":
        build_subset(a.year_min, a.year_max, a.min_cited_by, a.max_papers, a.out, a.raw, a.full)
        report(load_jsonl(a.out))
    elif a.cmd == "stats":
        report(load_jsonl(a.path))
    elif a.cmd == "peek":
        with open_raw() as f:
            for _, line in zip(range(a.n), f):
                print(json.dumps(json.loads(line), indent=1)[:1500], "\n")
