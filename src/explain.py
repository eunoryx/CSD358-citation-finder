"""
Stage 9: explanations + faithfulness check.

EXPLANATION. The cosine score is a sum over shared terms, so it decomposes exactly:
    zone_score(q, d) = sum_t  sum_zone  zone_w * w_tq * w_td
Each term's share of that sum is its contribution. We report the top terms.
The g(d) and co-citation parts of the net score are reported as separate lines,
so the explanation accounts for the WHOLE score, not just the text part.

FAITHFULNESS. An explanation is faithful if the terms it names really caused the
ranking. Test: delete the top-m explanation terms from the query (every occurrence),
re-run the full search and see how far the paper falls. Control: delete m OTHER
terms that the paper also matches (chosen at random). If explanations are faithful,
removing the named terms should hurt the rank much more than removing random ones.
"""
import random

import numpy as np

from src.preprocess import preprocess


def term_contributions(ranker, terms, doc, cfg):
    """{term: contribution to doc's zone-weighted cosine}, largest first."""
    contrib = {}
    for zone, zw in ranker.zone_weights(cfg).items():
        if zw == 0:
            continue
        for t, wq in ranker.query_weights(terms, zone).items():
            if ranker.idf[zone][t] < cfg.idf_threshold:
                continue
            ids, _ = ranker.index.postings[zone][t]
            pos = np.searchsorted(ids, doc)
            if pos < len(ids) and ids[pos] == doc:
                contrib[t] = contrib.get(t, 0.0) + zw * wq * float(ranker.wnorm[zone][t][pos])
    return dict(sorted(contrib.items(), key=lambda kv: -kv[1]))


def explain(ranker, title, abstract, year, doc, cfg, exclude=(), top=5):
    """Full breakdown of one result's score: text terms, g(d), co-citation."""
    terms = ranker.query_terms(title, abstract, cfg.use_title)
    contrib = term_contributions(ranker, terms, doc, cfg)
    out = {"terms": list(contrib.items())[:top], "text_score": sum(contrib.values())}
    if cfg.g_weight > 0:
        out["g"] = cfg.g_weight * float(ranker.quality.g(cfg.g_kind, year, np.asarray(list(exclude)))[doc])
    if cfg.cc_weight > 0:
        res = ranker.search(title, abstract, year, cfg.but(cc_weight=0), k=cfg.cc_top_n, exclude=exclude)
        top = [d for d, _ in res]
        n_citing = int(ranker.cocite.counts(top)[doc])
        out["cocite"] = (n_citing, len(top), cfg.cc_weight * n_citing / max(len(top), 1))
    return out


def remove_terms(text, removed, stem_on):
    """Rebuild the query text without any word whose index term is in `removed`."""
    words = text.split()
    return " ".join(w for w in words if not (set(preprocess(w, stem_on=stem_on)) & removed))


def rank_of(ranker, title, abstract, year, doc, cfg, exclude):
    """1-based rank of `doc` under the full pipeline (len(candidates)+1 if it is not a candidate)."""
    _, score, valid = ranker.search(title, abstract, year, cfg, k=1, exclude=exclude, return_scores=True)
    if not valid[doc]:
        return int(valid.sum()) + 1
    return int((score[valid] > score[doc]).sum()) + 1


def faithfulness(ranker, queries, cfg, m=3, results_per_query=10, seed=0):
    """For each query, each of its top results: rank after deleting the m explanation
    terms vs. after deleting m random other matching terms. Returns per-result rows."""
    rng = random.Random(seed)
    stem_on = ranker.index.stem_on
    rows = []
    for q in queries:
        res = ranker.search(q["title"], q["abstract"], q["year"], cfg, k=results_per_query, exclude=q["exclude"])
        terms = ranker.query_terms(q["title"], q["abstract"], cfg.use_title)
        for orig_rank, (doc, _) in enumerate(res, 1):
            contrib = list(term_contributions(ranker, terms, doc, cfg))
            if len(contrib) < 2 * m:
                continue  # not enough matching terms for a fair control
            top_terms = set(contrib[:m])
            rand_terms = set(rng.sample(contrib[m:], m))
            row = {"qid": q["id"], "doc": doc, "orig_rank": orig_rank, "n_matching_terms": len(contrib)}
            for name, removed in (("explained", top_terms), ("random", rand_terms)):
                t = remove_terms(q["title"], removed, stem_on)
                a = remove_terms(q["abstract"], removed, stem_on)
                row[f"rank_{name}"] = rank_of(ranker, t, a, q["year"], doc, cfg, q["exclude"])
            rows.append(row)
    return rows
