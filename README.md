# Citation Finder: vertical search for science (CSD358 hackathon)

Input: a paper's abstract (+ optional title and year). Output: a ranked list of papers it should probably cite,
with a per-term explanation of why each was ranked.

## Setup

Python 3.11+ (tested on 3.13). `pip install -r requirements.txt` (numpy, scikit-learn for the stop-word list,
nltk for the Porter stemmer, matplotlib for plots). Then download the data (below) and run:

```bash
python -m src.data subset            # 1. connected working subset -> data/subset.jsonl   (~3 min, 2 passes over 18 GB)
python -m src.index build            # 2. inverted index -> data/index_stem.pkl          (~8 s)
python -m src.index build --no-stem  #    (only needed for the stemming ablation)
python -m src.search --year 2016 --title "Neural ranking models for ad-hoc retrieval" \
       --abstract "We propose a deep neural network for ad-hoc document retrieval ..." --verbose
python -m src.eval all               # 3. tuning on dev + every experiment on test -> results/ (~15 min)
```

## Data

**Source:** AMiner, *DBLP-Citation-network V19* (7,289,177 papers, 85,014,559 citation relationships, released 2026-09-10).
- Dataset page: https://www.aminer.cn/aboutus/zh-CN/articles/655db2202ab17a072284bc0c
  (the old `https://www.aminer.org/citation` URL now returns "not found")
- File: `https://opendata.aminer.cn/dataset/DBLP-Citation-network-V19.zip` (5.96 GB zipped, JSON Lines inside)
- Downloaded: 2026-10-07
- **Terms:** AMiner states the data set is "designed for research purpose only". If you use it, you must cite:

> Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. *ArnetMiner: Extraction and Mining of
> Academic Social Networks.* In Proceedings of KDD 2008, pp. 990-998.

The data is derived from DBLP, ACM, MAG and other sources. We credit AMiner and DBLP.

### Download

```bash
mkdir -p data && cd data
curl -L -C - --retry 10 -o DBLP-Citation-network-V19.zip \
  https://opendata.aminer.cn/dataset/DBLP-Citation-network-V19.zip
python -c "import zipfile; zipfile.ZipFile('DBLP-Citation-network-V19.zip').extractall('.')"   # optional, ~18 GB
```

The loader can read straight from the `.zip`, so unzipping is optional.

### Record format (verified on the real file)

One JSON object per line. The fields we use are `id, title, abstract, year, venue, references (list of ids), n_citation`.

### Working subset

```bash
python -m src.data subset      # -> data/subset.jsonl
python -m src.data stats       # corpus size, references, how many resolve in the corpus
```

Random sampling gives almost no usable ground truth: in a random 3k sample, 52 of 23k references resolved.
So the subset is one **connected region of the citation graph**:
1. Seeds: papers from the main IR / web / data-mining conferences (SIGIR, CIKM, WSDM, ECIR, ICTIR, WWW, KDD)
   from 2012-2024. Forums, companion volumes, workshops and look-alike venues are excluded (see `src/data.py`).
2. Plus the papers those seeds cite most (cited by >= 2 seeds), up to `--max-papers` (default 50k).
3. Papers without an abstract or year are dropped.

Corpus stats (V19, default settings, built 2026-10-07):

| | |
|---|---|
| Full corpus: papers with abstract + year | 6,977,085 (5,710,768 have references) |
| Subset papers | 48,169 (19,918 seeds + 28,251 cited papers that have abstracts) |
| Papers with >= 1 reference | 45,378 (94.2%) |
| References resolving inside the subset | 778,367 / 1,258,540 (61.8%); seeds only: 73.1% |
| Papers with >= 10 resolved references (eval candidates) | 28,238 |

Known data quirk: 11k of the resolved references (~1.4%) point to a paper with a *later* year (preprint vs.
publication dates). The year filter (`year < query year`) will drop those, so they slightly cap recall.

## Search CLI

```bash
python -m src.search --abstract "..." [--title "..."] [--year 2015] [-k 10] [--verbose]
```

Each result prints its total score split into its parts and the query terms that contributed most, e.g.
(full output: `results/example_search_output.txt`):

```
 2. [0.351] (2003) Time-based Language Models.
      score = text 0.263 + g(d) 0.068 + co-citation 0.020 (cited by 1/10 top results)
      why: hoc (15%), model (12%), likelihood (11%), queri (11%), ad (11%)
```

`--verbose` (for the demo video) also prints the preprocessed query, and for each zone the ltc query weights
(tf, df, idf, w_q) of the top terms with the first entries of their postings lists and lnc document weights.
`python -m src.index show <word> [<word>]` prints a term's postings and runs the AND merge.
Every weight can be overridden: `--title-w --idf-threshold --g-weight --g-kind {citations,pagerank}
--cc-weight --cc-top-n --champion-r --bm25 --no-stem`. Defaults come from `results/tuned_params.json`.

## How it works (one line per lecture concept; code has the details next to each part)

| Stage | Concept | File |
|---|---|---|
| 1 | tokenize, case fold, stop words, Porter stemming (flag) | `src/preprocess.py` |
| 2 | inverted index: dictionary + docID-sorted postings (docID, tf), title / abstract zones, AND merge | `src/index.py` |
| 3 | lnc.ltc weighting, cosine, term-at-a-time accumulators, heap top-K, zone weights | `src/rank.py` |
| 4 | index elimination: skip query terms with idf < threshold | `src/rank.py` |
| 5 | parametric filter: only papers with year < query year | `src/rank.py` |
| 6 | static quality g(d) = log in-corpus citations, net score = cosine + w·g(d), leak-free | `src/quality.py` |
| 7 | champion lists (approximate, and union + exact rescoring) | `src/rank.py` |
| 8 | **co-citation boosting** (our novelty): pseudo-relevance feedback over citation links | `src/cocitation.py` |
| 9 | per-term score decomposition + faithfulness test | `src/explain.py` |
| 10 | BM25 comparison; PageRank as g(d) | `src/bm25.py`, `src/quality.py` |
| - | evaluation, tuning, plots | `src/eval.py`, `src/plots.py` |

## Evaluation

**Protocol** (`src/eval.py`). Query pool = seed papers (the IR/web/KDD conference papers) with at least 10
references inside the corpus: 11,841 papers. With a fixed seed (42) we draw a **dev set of 200** queries, used only
for tuning, and a disjoint **test set of 500**, used for every number below. For each query paper we hide it
(and any copy with the same normalised title) from the results, search with its title + abstract + year, and
compare with its true in-corpus references (on average 25.1 per test query). Metrics: P@k and R@k for
k = 5, 10, 20, 50, MRR (first relevant hit in the top 100) and NDCG@10/@50 (binary relevance). Δ columns are
paired bootstrap 95% confidence intervals over the 500 queries (2,000 resamples).

**Recall is conservative.** A paper cites only part of the relevant work, so a "wrong" result can still be a good
citation. Also, 4.3% of the true references have a year >= the query's year (preprint/publication date
mismatch), so they can never be found once the year filter is on.

**Leakage control.** Citation counts for g(d) are recomputed per query: we drop the edges coming from the held-out
paper (its own references would otherwise get +1 each), and only count citations from papers published before
the query year. The extra row with AMiner's global `n_citation` shows what leakage does: it scores higher (+0.016
R@50), because that count includes the query paper's own citations and citations from the future. We do not use it.

**Tuned weights** (greedy along the ladder on DEV, objective NDCG@50; `results/tuning_log_dev.csv`):
title zone weight 0.3 (abstract 0.7), idf threshold 0.25, g(d) weight 0.1, co-citation top-N 10 with weight 0.2.
BM25 got its own k1/b grid on DEV (`results/tuning_bm25_dev.csv`): k1 = 3.0, b = 0.9.

### Ablation ladder (500 test queries) - `results/ablation_ladder.csv`, `ablation_ladder.png`, `recall_at_k.png`

| Step | P@10 | R@10 | R@20 | R@50 | MRR | NDCG@10 | NDCG@50 | Δ R@50 [95% CI] | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| 1 baseline: tf-idf, abstract only | 0.179 | 0.083 | 0.117 | 0.180 | 0.498 | 0.213 | 0.191 | - | 5.5 |
| 2 + title zone | 0.202 | 0.093 | 0.133 | 0.204 | 0.539 | 0.237 | 0.215 | +0.025 [+0.0193, +0.0296] vs 1 | 6.2 |
| 3 + year filter | 0.251 | 0.115 | 0.161 | 0.243 | 0.644 | 0.300 | 0.262 | +0.038 [+0.0323, +0.0452] vs 2 | 5.2 |
| 4 + g(d) citations | 0.279 | 0.125 | 0.183 | 0.274 | 0.666 | 0.328 | 0.292 | +0.032 [+0.0244, +0.0384] vs 3 | 9.7 |
| 5 + co-citation (full system) | 0.337 | 0.149 | 0.222 | 0.331 | 0.669 | 0.376 | 0.341 | +0.057 [+0.0496, +0.0642] vs 4 | 12.2 |
| 6 BM25 (title zone + year filter) | 0.219 | 0.100 | 0.146 | 0.221 | 0.613 | 0.267 | 0.237 | -0.022 [-0.0270, -0.0172] vs 3 | 8.5 |
| 7 BM25 + g(d) + co-citation | 0.313 | 0.139 | 0.203 | 0.306 | 0.660 | 0.356 | 0.318 | -0.026 [-0.0323, -0.0193] vs 5 | 15.6 |
| x full system, g(d) = PageRank | 0.339 | 0.149 | 0.220 | 0.324 | 0.682 | 0.382 | 0.339 | -0.007 [-0.0130, -0.0015] vs 5 | 165.8 |
| x full system, g(d) = AMiner n_citation (LEAKS) | 0.362 | 0.160 | 0.233 | 0.348 | 0.706 | 0.406 | 0.361 | +0.016 [+0.0100, +0.0225] vs 5 | 8.1 |
| x full system, NO stemming | 0.329 | 0.144 | 0.220 | 0.327 | 0.678 | 0.371 | 0.337 | -0.004 [-0.0108, +0.0022] vs 5 | 11.2 |

Reading it:
- Every step of the ladder gives a significant gain (all CIs exclude 0). Overall: R@50 0.180 -> 0.331 (+84%),
  NDCG@50 0.191 -> 0.341, MRR 0.498 -> 0.669.
- **Co-citation boosting is the biggest single gain** (+0.057 R@50), even after g(d). It finds papers that the
  top text matches cite, even when their own abstract uses different words.
- BM25 (with tuned k1/b) is significantly worse than lnc.ltc here, both alone and with g(d) + co-citation. A likely
  reason: our query is a whole abstract, and ltc gives repeated query terms more weight (log tf) while our BM25 sums
  over distinct query terms. We did not test this further.
- PageRank as g(d) is no better than log citation count (-0.007 R@50, NDCG difference not significant) and is
  ~15x slower, because the leak-free PageRank has to be recomputed for every query.
- Stemming helps only slightly: removing it costs -0.004 R@50, and the CI includes 0, so the difference is not significant.
- ms/query for steps 4+ includes recomputing leak-free citation counts per query (~4 ms). A deployed system would
  compute them once.

### Index elimination (stage 4) - `results/idf_threshold_test.csv`, `idf_threshold_test.png`

Measured on test with text + zones + year + g(d). ms includes the constant ~4 ms g(d) step.

| idf threshold (log10 N/df) | longest list kept (df) | query terms kept | R@50 | NDCG@50 | ms/query |
|---|---|---|---|---|---|
| 0 (none) | 48,169 | 100% | 0.274 | 0.292 | 9.1 |
| 0.5 | 15,232 | 94% | 0.273 | 0.290 | 8.7 |
| 0.75 | 8,565 | 83% | 0.268 | 0.283 | 8.0 |
| 1.0 | 4,816 | 65% | 0.253 | 0.266 | 7.1 |
| 1.5 | 1,523 | 31% | 0.216 | 0.225 | 5.1 |
| 2.0 | 481 | 15% | 0.149 | 0.157 | 4.0 |

Low-idf terms can be skipped almost for free up to ~0.5-0.75. Beyond that, recall drops fast: long abstracts
need their mid-frequency terms. With the 1%-of-NDCG rule on dev we picked 0.25, which in practice removes almost nothing.

### Champion lists (stage 7) - `results/champion_lists.csv`, `champion_lists.png`

Text scoring only, step-3 config, 500 test queries. Exhaustive = 392k postings touched, **2.6 ms**, R@50 0.243.

| r | approx: postings touched | approx: ms | approx: R@50 | rescored: ms | rescored: R@50 | rescored: top-50 overlap with exhaustive |
|---|---|---|---|---|---|---|
| 10 | 1,401 | 0.64 | 0.061 | 6.1 | 0.117 | 0.24 |
| 50 | 6,514 | 0.69 | 0.096 | 20.4 | 0.207 | 0.63 |
| 100 | 12,317 | 0.75 | 0.120 | 34.4 | 0.230 | 0.81 |
| 250 | 27,713 | 0.82 | 0.151 | 64.4 | 0.242 | 0.96 |
| 1000 | 86,252 | 1.16 | 0.194 | 130.9 | 0.243 | 1.00 |

**Honest result: champion lists do not pay off on this corpus.** Approximate scoring (walk only the top-r postings)
is 2-4x faster but loses a lot of recall: a long query's relevant papers are mid-weight for many terms, not
top-weight for a few. Exact rescoring of the champion union needs r >= 250 to match exhaustive scoring, and is slower
here: at 48k docs, numpy scores all postings in 2.6 ms, so the per-candidate binary searches cost more than they save.
The tradeoff would change on a much larger, disk-based index.

### Explanation faithfulness (stage 9) - `results/faithfulness.csv`, `faithfulness.png`

For the top-10 results of 100 test queries: remove the top-3 explanation terms from the query (every occurrence),
re-run the full search, and record the result's new rank. Control: remove 3 random *other* terms that the paper also matches.

| config | results | median rank before | median rank after removing explanation terms | after removing random matching terms | fell out of top 10 (explained / random) | explained hurts more than random |
|---|---|---|---|---|---|---|
| text only (step 3) | 993 | 5 | 275 | 11 | 95% / 51% | 98% |
| full system (step 5) | 977 | 5 | 127 | 8 | 90% / 39% | 96% |

The explanations are faithful: the named terms cause the rank. In the full system the drop is smaller because g(d)
and co-citation also hold results up. The CLI reports those two parts as separate lines.

### Scaling

Same tuned weights, run on a larger region of the citation graph: the same 19,918 seed papers plus every paper they
cite at least once (`python -m src.data subset --max-papers 200000 --min-cited-by 1 --out data/subset_large.jsonl`;
only 128,237 such papers have an abstract, so the 200k cap is not reached). `results/scaling.csv`, `results/log_scaling*.txt`.

| corpus | papers | abstract postings | index build | index file | step | R@50 | NDCG@50 | MRR | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| default subset | 48,169 | 3.16M | 7.6 s | 69 MB | 3 + year filter | 0.243 | 0.262 | 0.644 | 5.4 |
| | | | | | 5 full system | 0.331 | 0.341 | 0.669 | 12.2 |
| large subset | 128,237 | 8.16M | 19.9 s | 174 MB | 3 + year filter | 0.213 | 0.237 | 0.622 | 15.9 |
| | | | | | 5 full system | 0.283 | 0.298 | 0.647 | 36.3 |

Build time and index size grow roughly linearly (2.7x docs -> 2.6x time and size), and query time grows ~3x.
Recall drops because there are more distractors per query. Co-citation still adds +0.07 R@50 on the larger corpus.
Caveat: the larger corpus has a bigger query pool (14,116), so its 500 test queries are a different random draw;
the two rows are not a paired comparison. The full 7M-paper corpus was not indexed (see limits below).

## What works / what is still planned

Works (all run on real data, numbers above): stages 1-10, the CLI with explanations and `--verbose` internals,
and the full evaluation with dev/test split, paired CIs, plots and leakage control.

Limits / planned:
- **Full corpus (7M papers).** `python -m src.data subset --full` writes all 6,977,085 papers with an abstract, but
  our in-memory Python index (dicts of numpy arrays) does not fit in 15 GB RAM at that size. It would need
  block-based index construction (BSBI/SPIMI with on-disk postings). Not done.
- Champion lists don't speed things up at our scale (see above). Tiered indexes or a g(d)-ordered champion list were not tried.
- Explanations decompose the lnc.ltc cosine. With `--bm25` the "why" terms are still the cosine terms, not BM25 contributions.
- BM25 sums over distinct query terms. A query-tf-aware BM25 variant was not tried.
- Known preprocessing quirk: stop words are removed *before* stemming, so "uses" -> `us` survives as a term.
- Query pool = conference seed papers 2012-2024 only. Results may differ for other fields.
