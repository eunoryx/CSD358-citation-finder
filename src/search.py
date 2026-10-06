"""
Command-line citation finder.

  python -m src.search --abstract "We study ..." --title "..." --year 2015
  python -m src.search --abstract "..." --year 2015 --verbose     # show the IR internals (for the demo video)

Prints the top-K papers the query paper should probably cite, each with its score
breakdown (text cosine, g(d), co-citation) and the query terms contributing most.
Weights default to the values tuned on the dev set (results/tuned_params.json).
"""
import argparse
import os
import time

import numpy as np

from src.explain import explain
from src.rank import Config, build_ranker

PARAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tuned_params.json")


def default_config():
    if os.path.exists(PARAMS):
        from src.eval import load_params
        return load_params()
    return Config(g_weight=0.1, cc_weight=0.1)


def show_internals(ranker, title, abstract, cfg, top_terms=8):
    """--verbose: preprocessing output, ltc query weights, postings and lnc doc weights."""
    terms = ranker.query_terms(title, abstract, cfg.use_title)
    print("=" * 80)
    print(f"[1] preprocessing -> {len(terms)} index terms ({len(set(terms))} distinct):")
    print("    " + " ".join(terms[:40]) + (" ..." if len(terms) > 40 else ""))
    for zone, zw in ranker.zone_weights(cfg).items():
        qw = ranker.query_weights(terms, zone)
        kept = {t: w for t, w in qw.items() if ranker.idf[zone][t] >= cfg.idf_threshold}
        print(f"\n[2] zone '{zone}' (zone weight {zw}): ltc query vector, {len(kept)}/{len(qw)} terms kept "
              f"by index elimination (idf >= {cfg.idf_threshold})")
        print(f"    {'term':14s} {'tf_q':>4s} {'df':>7s} {'idf':>6s} {'w_q (ltc)':>9s}  first postings (docID:tf -> lnc w_d)")
        for t, w in sorted(kept.items(), key=lambda kv: -kv[1])[:top_terms]:
            ids, tfs = ranker.index.postings[zone][t]
            wd = ranker.wnorm[zone][t]
            post = ", ".join(f"{d}:{tf}->{x:.3f}" for d, tf, x in list(zip(ids, tfs, wd))[:3])
            print(f"    {t:14s} {terms.count(t):4d} {len(ids):7,d} {ranker.idf[zone][t]:6.2f} {w:9.3f}  [{post}, ...]")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Find papers that a paper should cite.")
    ap.add_argument("--abstract", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--year", type=int, default=None, help="only return papers published before this year")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--verbose", action="store_true", help="print postings, tf-idf weights and score parts")
    ap.add_argument("--no-stem", action="store_true")
    ap.add_argument("--bm25", action="store_true")
    ap.add_argument("--title-w", type=float)
    ap.add_argument("--idf-threshold", type=float)
    ap.add_argument("--g-weight", type=float)
    ap.add_argument("--g-kind", choices=["citations", "pagerank"])
    ap.add_argument("--cc-weight", type=float)
    ap.add_argument("--cc-top-n", type=int)
    ap.add_argument("--champion-r", type=int, help="score only the union of champion lists of this size")
    a = ap.parse_args()

    cfg = default_config()
    overrides = {"title_w": a.title_w, "idf_threshold": a.idf_threshold, "g_weight": a.g_weight,
                 "g_kind": a.g_kind, "cc_weight": a.cc_weight, "cc_top_n": a.cc_top_n, "champion_r": a.champion_r}
    cfg = cfg.but(**{k: v for k, v in overrides.items() if v is not None})
    if a.bm25:
        cfg = cfg.but(model="bm25")
    if a.year is None:
        cfg = cfg.but(year_filter=False)

    t0 = time.time()
    ranker = build_ranker(stem_on=not a.no_stem)
    print(f"loaded index of {ranker.index.N:,} papers in {time.time()-t0:.1f}s | config: {cfg}")
    if a.verbose:
        show_internals(ranker, a.title, a.abstract, cfg)

    t0 = time.perf_counter()
    results = ranker.search(a.title, a.abstract, a.year, cfg, k=a.k)
    print(f"\nsearch took {1000*(time.perf_counter()-t0):.1f} ms\n")

    for rank, (d, score) in enumerate(results, 1):
        doc = ranker.index.docs[d]
        e = explain(ranker, a.title, a.abstract, a.year, d, cfg)
        print(f"{rank:2d}. [{score:.3f}] ({doc['year']}) {doc['title']}")
        print(f"      {doc['venue'][:60]} | AMiner id {doc['id']}")
        parts = [f"text {e['text_score']:.3f}"]
        if "g" in e:
            parts.append(f"g(d) {e['g']:.3f}")
        if "cocite" in e:
            n, top, b = e["cocite"]
            parts.append(f"co-citation {b:.3f} (cited by {n}/{top} top results)")
        print("      score = " + " + ".join(parts))
        if e["terms"]:
            total = e["text_score"] or 1
            print("      why: " + ", ".join(f"{t} ({c/total:.0%})" for t, c in e["terms"]))
        print()


if __name__ == "__main__":
    main()
