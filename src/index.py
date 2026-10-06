"""
Stage 2: inverted index with zones.

An inverted index maps each TERM to the list of DOCUMENTS containing it:

    dictionary                postings list (sorted by docID)
    "retriev"  (df=9,812)  -> [(3, 2), (17, 1), (42, 4), ...]     each entry = (docID, tf)
    "bm25"     (df=611)    -> [(88, 1), (1201, 3), ...]

- dictionary: term -> postings. We also keep df (document frequency) = length of the postings list.
- postings are sorted by docID, which lets us intersect/merge lists in one linear pass.
- tf (term frequency) is stored in each posting because tf-idf ranking needs it.

ZONES: a paper has a title and an abstract. We build one separate index per zone,
so ranking (stage 3) can weight a title match more than an abstract match.

docIDs are small integers 0..N-1 (position in self.docs). The original AMiner id is
kept in self.docs[docID]["id"].

Run:
  python -m src.index build            # build + save data/index_stem.pkl
  python -m src.index build --no-stem  # build + save data/index_nostem.pkl
  python -m src.index show retrieval   # print the postings of a term in both zones
"""
import argparse
import collections
import os
import pickle
import random
import time

import numpy as np

from src.data import DATA_DIR, SUBSET, load_jsonl
from src.preprocess import preprocess

ZONES = ("title", "abstract")


class InvertedIndex:
    def __init__(self, stem_on=True):
        self.stem_on = stem_on
        self.docs = []  # docID -> {"id", "title", "year", "n_citation", "references"}
        # zone -> {term: (docIDs array, tfs array)}
        self.postings = {z: {} for z in ZONES}

    @property
    def N(self):
        return len(self.docs)

    def build(self, papers):
        """Single pass over the collection.

        For each document and zone: preprocess the text, count tf per term, and append
        (docID, tf) to that term's postings. Because we visit docs in docID order, every
        postings list comes out already sorted by docID (no sort step needed).
        """
        lists = {z: collections.defaultdict(lambda: ([], [])) for z in ZONES}
        for doc_id, p in enumerate(papers):
            self.docs.append({k: p[k] for k in ("id", "title", "year", "venue", "n_citation", "references")})
            for zone in ZONES:
                tf = collections.Counter(preprocess(p[zone], stem_on=self.stem_on))
                for term, count in tf.items():
                    ids, tfs = lists[zone][term]
                    ids.append(doc_id)
                    tfs.append(count)
        # Freeze each postings list into compact numpy arrays (int32 docIDs, int16 tfs).
        for zone in ZONES:
            self.postings[zone] = {
                t: (np.array(ids, dtype=np.int32), np.array(tfs, dtype=np.int16))
                for t, (ids, tfs) in lists[zone].items()
            }
        return self

    # ---- lookups -------------------------------------------------------
    def get_postings(self, term, zone):
        """Postings (docIDs, tfs) of an already-preprocessed term; empty if the term is unseen."""
        return self.postings[zone].get(term, (np.empty(0, np.int32), np.empty(0, np.int16)))

    def df(self, term, zone):
        return len(self.get_postings(term, zone)[0])

    # ---- save / load ---------------------------------------------------
    def path(self):
        return os.path.join(DATA_DIR, f"index_{'stem' if self.stem_on else 'nostem'}.pkl")

    def save(self, path=None):
        # store plain data (not the class), so the file loads from any script
        with open(path or self.path(), "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(stem_on=True, path=None):
        path = path or os.path.join(DATA_DIR, f"index_{'stem' if stem_on else 'nostem'}.pkl")
        idx = InvertedIndex()
        with open(path, "rb") as f:
            idx.__dict__.update(pickle.load(f))
        return idx


def intersect(p1, p2):
    """Classic postings-list merge (AND query): walk both sorted lists with two pointers.
    Runs in O(len(p1) + len(p2)) because both lists are sorted by docID.
    Shown here for the lecture concept; ranking (stage 3) uses scoring, not boolean AND."""
    i = j = 0
    out = []
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            out.append(int(p1[i])); i += 1; j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return out


def check_against_brute_force(index, papers, n_terms=200, seed=0):
    """Sanity test: for random terms, recount tf by scanning every document directly
    and compare with what the index stored. Returns number of mismatching terms."""
    rng = random.Random(seed)
    bad = 0
    for zone in ZONES:
        terms = rng.sample(sorted(index.postings[zone]), n_terms)
        wanted = set(terms)
        truth = {t: {} for t in terms}
        for d, p in enumerate(papers):
            for t, c in collections.Counter(preprocess(p[zone], stem_on=index.stem_on)).items():
                if t in wanted:
                    truth[t][d] = c
        for t in terms:
            ids, tfs = index.get_postings(t, zone)
            if dict(zip(ids.tolist(), tfs.tolist())) != truth[t] or list(ids) != sorted(ids):
                bad += 1
    return bad


def print_stats(index, seconds=None):
    for zone in ZONES:
        lens = np.array([len(ids) for ids, _ in index.postings[zone].values()])
        print(f"  zone {zone:8s}: {len(lens):,} terms, {lens.sum():,} postings, "
              f"mean list length {lens.mean():.1f}, longest {lens.max():,}, "
              f"terms with df=1: {(lens == 1).sum():,}")
    if seconds is not None:
        print(f"  build time: {seconds:.1f}s")


def show(index, word, limit=8):
    """Demo helper: print a term's dictionary entry and the start of its postings lists."""
    terms = preprocess(word, stem_on=index.stem_on)
    if not terms:
        print(f"'{word}' is removed by preprocessing (stop word / too short)")
        return
    for term in terms:
        for zone in ZONES:
            ids, tfs = index.get_postings(term, zone)
            print(f"\n'{word}' -> term '{term}'  zone={zone}  df={len(ids):,}")
            for d, tf in list(zip(ids, tfs))[:limit]:
                print(f"   docID {d:6d}  tf={tf}  ({index.docs[d]['year']}) {index.docs[d]['title'][:70]}")
            if len(ids) > limit:
                print(f"   ... {len(ids) - limit:,} more")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--no-stem", action="store_true")
    b.add_argument("--subset", default=SUBSET)
    b.add_argument("--out", default=None, help="index file (default data/index_stem.pkl)")
    s = sub.add_parser("show")
    s.add_argument("words", nargs="+")
    s.add_argument("--no-stem", action="store_true")
    a = ap.parse_args()

    if a.cmd == "build":
        papers = load_jsonl(a.subset)
        t0 = time.time()
        idx = InvertedIndex(stem_on=not a.no_stem).build(papers)
        secs = time.time() - t0
        print(f"built index over {idx.N:,} docs (stemming {'on' if idx.stem_on else 'off'})")
        print_stats(idx, secs)
        out = a.out or idx.path()
        idx.save(out)
        print(f"  saved {out} ({os.path.getsize(out)/1e6:.0f} MB)")
        t0 = time.time()
        bad = check_against_brute_force(idx, papers)
        print(f"  brute-force check on 200 random terms per zone: {bad} mismatches ({time.time()-t0:.0f}s)")
    else:
        idx = InvertedIndex.load(stem_on=not a.no_stem)
        for w in a.words:
            show(idx, w)
        if len(a.words) == 2:
            t1, t2 = (preprocess(w, stem_on=idx.stem_on)[0] for w in a.words)
            p1, p2 = idx.get_postings(t1, "abstract")[0], idx.get_postings(t2, "abstract")[0]
            both = intersect(p1, p2)
            print(f"\nAND merge in abstract zone: {len(p1):,} AND {len(p2):,} -> {len(both):,} docs")
