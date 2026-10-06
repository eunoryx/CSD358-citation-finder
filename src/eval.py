"""
Evaluation: hide a paper's references, search with its title/abstract/year, and
check how many of its true references we find.

Protocol
  - Query pool: seed papers (SIGIR/CIKM/WSDM/ECIR/ICTIR/WWW/KDD 2012-2024) that have
    >= 10 references resolving inside the corpus. Fixed seed (42).
  - Disjoint DEV set (200 queries) for tuning weights, TEST set (500) for every
    number we report. Tuning on the test set would overfit the reported numbers.
  - Relevant docs for a query = its references inside the corpus (binary relevance).
  - Leakage: the query paper and any duplicate of it (same normalised title, e.g.
    an arXiv copy) are excluded from the results, from g(d) counts and from co-citation.
  - Recall is CONSERVATIVE: a paper cites only some of the relevant work, so a
    "wrong" result may still be a perfectly good citation.

Metrics (k = 5, 10, 20, 50):
  P@k      = relevant in top k / k
  R@k      = relevant in top k / all relevant
  MRR      = 1 / rank of the first relevant result (0 if none in the top 100)
  NDCG@k   = DCG@k / ideal DCG@k with DCG = sum_i rel_i / log2(i + 1)

Commands (each writes CSV/PNG files into results/):
  python -m src.eval tune          # tune weights on DEV  -> results/tuned_params.json
  python -m src.eval ladder        # ablation ladder on TEST
  python -m src.eval idf           # index elimination: idf threshold vs speed vs quality
  python -m src.eval champion      # champion list size vs speed vs recall
  python -m src.eval faithfulness  # explanation faithfulness test
  python -m src.eval all
"""
import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict

import numpy as np

from src.data import SUBSET, load_jsonl
from src.rank import Config, build_ranker

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
KS = (5, 10, 20, 50)
SEED = 42
N_DEV, N_TEST = 200, 500
PARAMS_FILE = os.path.join(RESULTS, "tuned_params.json")


# ---------------------------------------------------------------- queries
def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def make_queries(ranker, subset_path=SUBSET, min_refs=10):
    """Returns (dev, test) lists of query dicts."""
    papers = load_jsonl(subset_path)
    id2doc = {d["id"]: i for i, d in enumerate(ranker.index.docs)}
    by_title = defaultdict(list)
    for i, d in enumerate(ranker.index.docs):
        by_title[norm_title(d["title"])].append(i)
    pool = []
    for p in papers:
        if not p.get("is_seed"):
            continue
        me = id2doc[p["id"]]
        family = set(by_title[norm_title(p["title"])]) | {me}   # the paper + its duplicates
        relevant = {id2doc[r] for r in p["references"] if r in id2doc} - family
        if len(relevant) >= min_refs:
            pool.append({"id": p["id"], "doc": me, "title": p["title"], "abstract": p["abstract"],
                         "year": p["year"], "relevant": relevant, "exclude": np.array(sorted(family))})
    pool.sort(key=lambda q: q["id"])
    random.Random(SEED).shuffle(pool)
    return pool[:N_DEV], pool[N_DEV:N_DEV + N_TEST], len(pool)


# ---------------------------------------------------------------- metrics
def metrics(ranked, relevant):
    m = {}
    hits = [d in relevant for d in ranked]
    for k in KS:
        h = sum(hits[:k])
        m[f"P@{k}"] = h / k
        m[f"R@{k}"] = h / len(relevant)
    first = next((i for i, x in enumerate(hits) if x), None)
    m["MRR"] = 0.0 if first is None else 1.0 / (first + 1)
    for k in (10, 50):
        dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(hits[:k]) if x)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        m[f"NDCG@{k}"] = dcg / idcg
    return m


METRIC_NAMES = [f"P@{k}" for k in KS] + [f"R@{k}" for k in KS] + ["MRR", "NDCG@10", "NDCG@50"]


def run(ranker, queries, cfg, k=100):
    """Per-query metrics + timing for one config."""
    per_q, times = [], []
    for q in queries:
        t0 = time.perf_counter()
        res = ranker.search(q["title"], q["abstract"], q["year"], cfg, k=k, exclude=q["exclude"])
        times.append(time.perf_counter() - t0)
        per_q.append(metrics([d for d, _ in res], q["relevant"]))
    mean = {name: statistics.fmean(m[name] for m in per_q) for name in METRIC_NAMES}
    mean["ms_per_query"] = 1000 * statistics.median(times)
    return mean, per_q


def paired_bootstrap(a, b, n=2000, seed=0):
    """95% CI of mean(b - a) over queries (paired: same queries in both systems)."""
    rng = np.random.default_rng(seed)
    diff = np.array(b) - np.array(a)
    boots = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def write_csv(path, rows):
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {os.path.relpath(path)}")


# ---------------------------------------------------------------- tuning (DEV only)
def tune(ranker, dev):
    """Greedy tuning along the ladder, objective = NDCG@50 on DEV."""
    obj = "NDCG@50"
    log = []

    def score(cfg, what):
        m, _ = run(ranker, dev, cfg)
        log.append({"step": what, **{k: v for k, v in vars(cfg).items()}, obj: m[obj], "R@50": m["R@50"],
                    "ms_per_query": m["ms_per_query"]})
        print(f"  {what:38s} {obj}={m[obj]:.4f} R@50={m['R@50']:.4f} {m['ms_per_query']:.1f}ms")
        return m[obj]

    print("tuning title zone weight (year filter off)")
    best_tw = max((0.1, 0.2, 0.3, 0.4, 0.5),
                  key=lambda w: score(Config(title_w=w, year_filter=False), f"title_w={w}"))

    print("choosing idf threshold (largest within 1% of no elimination)")
    base = Config(title_w=best_tw)
    full = score(base, "idf>=0")
    best_idf = 0.0
    for th in (0.25, 0.5, 0.75, 1.0, 1.5):
        if score(base.but(idf_threshold=th), f"idf>={th}") >= 0.99 * full:
            best_idf = th
    base = base.but(idf_threshold=best_idf)

    print("tuning g(d) weight (in-corpus citations)")
    best_g = max((0.02, 0.05, 0.1, 0.2, 0.3), key=lambda w: score(base.but(g_weight=w), f"g_weight={w}"))
    base = base.but(g_weight=best_g)

    print("tuning co-citation (top-N, weight)")
    grid = [(n, w) for n in (5, 10, 20, 50) for w in (0.05, 0.1, 0.2, 0.4)]
    best_n, best_cc = max(grid, key=lambda nw: score(base.but(cc_top_n=nw[0], cc_weight=nw[1]),
                                                       f"cc_top_n={nw[0]} cc_weight={nw[1]}"))
    params = {"title_w": best_tw, "idf_threshold": best_idf, "g_weight": best_g,
              "cc_top_n": best_n, "cc_weight": best_cc, "tuned_on": f"{len(dev)} dev queries, objective {obj}"}
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    write_csv(os.path.join(RESULTS, "tuning_log_dev.csv"), log)
    print("  tuned:", params)
    return params


def load_params():
    with open(PARAMS_FILE) as f:
        p = json.load(f)
    return Config(title_w=p["title_w"], idf_threshold=p["idf_threshold"], g_weight=p["g_weight"],
                  cc_top_n=p["cc_top_n"], cc_weight=p["cc_weight"],
                  bm25_k1=p.get("bm25_k1", 1.2), bm25_b=p.get("bm25_b", 0.75))


def tune_bm25(ranker, dev):
    """Fair comparison: give BM25 its own k1/b tuning on DEV (same objective), with the
    tuned zone weight and year filter (= ladder step 6 setting)."""
    base = load_params().but(model="bm25", g_weight=0, cc_weight=0)
    log = []
    for k1 in (0.6, 0.9, 1.2, 1.6, 2.0, 3.0, 5.0, 8.0, 12.0):
        for b in (0.3, 0.5, 0.75, 0.9, 1.0):
            m, _ = run(ranker, dev, base.but(bm25_k1=k1, bm25_b=b))
            log.append({"bm25_k1": k1, "bm25_b": b, "NDCG@50": m["NDCG@50"], "R@50": m["R@50"]})
            print(f"  k1={k1} b={b}: NDCG@50={m['NDCG@50']:.4f} R@50={m['R@50']:.4f}")
    best = max(log, key=lambda r: r["NDCG@50"])
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    params.update(bm25_k1=best["bm25_k1"], bm25_b=best["bm25_b"])
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    write_csv(os.path.join(RESULTS, "tuning_bm25_dev.csv"), log)
    print("  tuned BM25:", best)


# ---------------------------------------------------------------- experiments (TEST)
def ladder(ranker, test, nostem_ranker=None):
    full = load_params()
    steps = [
        ("1 baseline: tf-idf, abstract only", full.but(use_title=False, year_filter=False, g_weight=0, cc_weight=0)),
        ("2 + title zone", full.but(year_filter=False, g_weight=0, cc_weight=0)),
        ("3 + year filter", full.but(g_weight=0, cc_weight=0)),
        ("4 + g(d) citations", full.but(cc_weight=0)),
        ("5 + co-citation (full system)", full),
        ("6 BM25 (title zone + year filter)", full.but(model="bm25", g_weight=0, cc_weight=0)),
        ("7 BM25 + g(d) + co-citation", full.but(model="bm25")),
        ("x full system, g(d) = PageRank", full.but(g_kind="pagerank")),
        ("x full system, g(d) = AMiner n_citation (LEAKS)", full.but(g_kind="aminer")),
    ]
    rows, per_query = [], {}
    for name, cfg in steps:
        m, pq = run(ranker, test, cfg)
        per_query[name] = pq
        rows.append({"step": name, **m})
        print(f"  {name:48s} R@50={m['R@50']:.4f} NDCG@50={m['NDCG@50']:.4f} MRR={m['MRR']:.4f} {m['ms_per_query']:.1f}ms")
    if nostem_ranker is not None:
        name = "x full system, NO stemming"
        m, pq = run(nostem_ranker, test, full)
        per_query[name] = pq
        rows.append({"step": name, **m})
        print(f"  {name:48s} R@50={m['R@50']:.4f} NDCG@50={m['NDCG@50']:.4f} MRR={m['MRR']:.4f}")

    # paired bootstrap: each step vs the one it builds on
    compare = {"2 + title zone": "1 baseline: tf-idf, abstract only", "3 + year filter": "2 + title zone",
               "4 + g(d) citations": "3 + year filter", "5 + co-citation (full system)": "4 + g(d) citations",
               "6 BM25 (title zone + year filter)": "3 + year filter",
               "7 BM25 + g(d) + co-citation": "5 + co-citation (full system)",
               "x full system, g(d) = PageRank": "5 + co-citation (full system)",
               "x full system, g(d) = AMiner n_citation (LEAKS)": "5 + co-citation (full system)",
               "x full system, NO stemming": "5 + co-citation (full system)"}
    for r in rows:
        ref = compare.get(r["step"])
        if not ref:
            continue
        for metric in ("R@50", "NDCG@50"):
            d, lo, hi = paired_bootstrap([m[metric] for m in per_query[ref]], [m[metric] for m in per_query[r["step"]]])
            r[f"vs"] = ref.split(" ")[0]
            r[f"d_{metric}"] = d
            r[f"d_{metric}_ci95"] = f"[{lo:+.4f}, {hi:+.4f}]"
    write_csv(os.path.join(RESULTS, "ablation_ladder.csv"), rows)
    # also the unreachable part of the ground truth (refs not older than the query)
    years = ranker.years
    unreachable = statistics.fmean(sum(years[d] >= q["year"] for d in q["relevant"]) / len(q["relevant"]) for q in test)
    print(f"  share of true references NOT older than the query (unreachable with year filter): {unreachable:.2%}")
    from src.plots import plot_ladder, plot_recall_at_k
    plot_ladder(rows, os.path.join(RESULTS, "ablation_ladder.png"))
    plot_recall_at_k(rows, os.path.join(RESULTS, "recall_at_k.png"))
    return rows


def idf_sweep(ranker, queries, tag):
    full = load_params().but(cc_weight=0)   # text + year + g(d): elimination acts on the text part
    rows = []
    for th in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        cfg = full.but(idf_threshold=th)
        kept = []
        for q in queries[:100]:
            terms = set(ranker.query_terms(q["title"], q["abstract"]))
            kept.append(sum(ranker.idf["abstract"].get(t, 99) >= th for t in terms) / max(len(terms), 1))
        m, _ = run(ranker, queries, cfg)
        rows.append({"idf_threshold": th, "max_df_kept": int(ranker.index.N / 10 ** th),
                     "query_terms_kept": statistics.fmean(kept), **m})
        print(f"  idf>={th:<4} keeps {rows[-1]['query_terms_kept']:.0%} of query terms  R@50={m['R@50']:.4f} "
              f"NDCG@50={m['NDCG@50']:.4f}  {m['ms_per_query']:.2f}ms")
    write_csv(os.path.join(RESULTS, f"idf_threshold_{tag}.csv"), rows)
    from src.plots import plot_tradeoff
    plot_tradeoff(rows, "idf_threshold", "idf threshold", os.path.join(RESULTS, f"idf_threshold_{tag}.png"),
                  title="Index elimination: skipping low-idf query terms")
    return rows


def champion_sweep(ranker, test):
    """Champion lists vs exhaustive scoring on the step-3 config (text + zones + year filter).
    Time = median ms of the TEXT SCORING step only (g(d)/co-citation would add the same
    constant to every variant). Cost is also counted as postings touched per query,
    which does not depend on the machine or on numpy."""
    cfg = load_params().but(g_weight=0, cc_weight=0)
    t0 = time.perf_counter()
    ranker.precompute_champions()
    print(f"  precomputing champion lists for all terms: {time.perf_counter()-t0:.1f}s (one-off, offline)")

    def measure(c, label, r_value):
        times, touched, overlap, ncand, per_q = [], [], [], [], []
        for q in test:
            terms = ranker.query_terms(q["title"], q["abstract"], c.use_title)
            t0 = time.perf_counter()
            if c.champion_r > 0:
                text, n = ranker.cosine_scores_champion(terms, c)
            else:
                text = ranker.cosine_scores(terms, c)
                n = sum(len(ranker.index.postings[z][t][0]) for z, zw in ranker.zone_weights(c).items() if zw > 0
                        for t in ranker.query_weights(terms, z) if ranker.idf[z][t] >= c.idf_threshold)
            times.append(time.perf_counter() - t0)
            touched.append(n)
            ncand.append(int((text > 0).sum()))
            res = ranker.search(q["title"], q["abstract"], q["year"], c, k=100, exclude=q["exclude"])
            ranked = [d for d, _ in res]
            per_q.append(metrics(ranked, q["relevant"]))
            if label != "exhaustive":
                overlap.append(len(set(ranked[:50]) & set(exact_top[q["id"]])) / max(len(exact_top[q["id"]]), 1))
            else:
                exact_top[q["id"]] = ranked[:50]
        row = {"variant": label, "champion_r": r_value,
               "text_scoring_ms": 1000 * statistics.median(times),
               "postings_touched": statistics.fmean(touched), "candidates": statistics.fmean(ncand),
               **{m: statistics.fmean(x[m] for x in per_q) for m in ("R@20", "R@50", "NDCG@50", "MRR")},
               "overlap@50_with_exhaustive": statistics.fmean(overlap) if overlap else 1.0}
        print(f"  {label:10s} r={str(r_value):5s} {row['text_scoring_ms']:6.2f}ms  touched={row['postings_touched']:9.0f} "
              f"cands={row['candidates']:7.0f}  R@50={row['R@50']:.4f}  overlap@50={row['overlap@50_with_exhaustive']:.3f}")
        return row

    exact_top = {}
    rows = [measure(cfg, "exhaustive", "all")]
    for exact in (False, True):
        for r in (5, 10, 25, 50, 100, 250, 500, 1000):
            rows.append(measure(cfg.but(champion_r=r, champion_exact=exact), "rescored" if exact else "approx", r))
    write_csv(os.path.join(RESULTS, "champion_lists.csv"), rows)
    from src.plots import plot_champion
    plot_champion(rows, os.path.join(RESULTS, "champion_lists.png"))
    return rows


def faithfulness_test(ranker, test, n_queries=100):
    from src.explain import faithfulness
    full = load_params()
    out = []
    for name, cfg in (("text only (step 3)", full.but(g_weight=0, cc_weight=0)), ("full system (step 5)", full)):
        t0 = time.perf_counter()
        rows = faithfulness(ranker, test[:n_queries], cfg)
        a = np.array([r["rank_explained"] for r in rows])
        b = np.array([r["rank_random"] for r in rows])
        o = np.array([r["orig_rank"] for r in rows])
        summ = {"config": name, "results_tested": len(rows),
                "median_rank_before": float(np.median(o)),
                "median_rank_after_removing_explanation_terms": float(np.median(a)),
                "median_rank_after_removing_random_matching_terms": float(np.median(b)),
                "pct_dropped_out_of_top10_explained": float((a > 10).mean()),
                "pct_dropped_out_of_top10_random": float((b > 10).mean()),
                "pct_explained_hurts_more_than_random": float((a > b).mean()),
                "pct_tie": float((a == b).mean())}
        out.append(summ)
        write_csv(os.path.join(RESULTS, f"faithfulness_rows_{'full' if 'full' in name else 'text'}.csv"), rows)
        print(f"  {name}: {len(rows)} results, median rank {summ['median_rank_before']:.0f} -> "
              f"{summ['median_rank_after_removing_explanation_terms']:.0f} (explanation terms removed) vs "
              f"{summ['median_rank_after_removing_random_matching_terms']:.0f} (random terms removed); "
              f"explained hurts more in {summ['pct_explained_hurts_more_than_random']:.0%} ({time.perf_counter()-t0:.0f}s)")
    write_csv(os.path.join(RESULTS, "faithfulness.csv"), out)
    from src.plots import plot_faithfulness
    plot_faithfulness(out, os.path.join(RESULTS, "faithfulness.png"))
    return out


def scale_run(ranker, test, subset_path):
    """Same tuned configs on a bigger corpus (appends one row per config to results/scaling.csv).
    The test queries are re-drawn from this corpus' pool with the same seed."""
    full = load_params()
    rows = []
    for name, cfg in (("3 + year filter", full.but(g_weight=0, cc_weight=0)),
                      ("5 full system", full)):
        m, _ = run(ranker, test, cfg)
        rows.append({"corpus": os.path.basename(subset_path), "docs": ranker.index.N, "step": name, **m})
        print(f"  {name:20s} R@50={m['R@50']:.4f} NDCG@50={m['NDCG@50']:.4f} MRR={m['MRR']:.4f} {m['ms_per_query']:.1f}ms")
    path = os.path.join(RESULTS, "scaling.csv")
    old = list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []
    old = [r for r in old if r["corpus"] != os.path.basename(subset_path)]
    write_csv(path, old + rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tune", "tune_bm25", "ladder", "idf", "champion", "faithfulness", "all", "scale"])
    ap.add_argument("--subset", default=SUBSET, help="corpus file (for the scaling experiment)")
    ap.add_argument("--index", default=None, help="index file built from --subset")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    ranker = build_ranker(stem_on=True, index_path=a.index)
    dev, test, pool = make_queries(ranker, a.subset)
    print(f"query pool: {pool:,} seed papers with >=10 resolved refs; dev={len(dev)}, test={len(test)} (seed {SEED})")
    print(f"  test: mean {statistics.fmean(len(q['relevant']) for q in test):.1f} relevant refs/query; "
          f"{sum(len(q['exclude']) > 1 for q in test)} queries have a duplicate copy in the corpus (excluded)")
    if a.what in ("tune", "all"):
        tune(ranker, dev)
    if a.what in ("tune_bm25", "all"):
        tune_bm25(ranker, dev)
    if a.what in ("ladder", "all"):
        nostem = build_ranker(stem_on=False)
        ladder(ranker, test, nostem)
    if a.what in ("idf", "all"):
        idf_sweep(ranker, test, "test")
    if a.what in ("champion", "all"):
        champion_sweep(ranker, test)
    if a.what in ("faithfulness", "all"):
        faithfulness_test(ranker, test)
    if a.what == "scale":
        scale_run(ranker, test, a.subset)
    print(f"done in {time.time()-t0:.0f}s")
