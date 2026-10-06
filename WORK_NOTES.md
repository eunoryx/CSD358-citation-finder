# Work notes: which component lives where

Fill in the Owner column so each teammate explains their part on video. The "Demo" column lists a command that
shows the concept live.

| Component (lecture concept) | File | Demo command | Owner |
|---|---|---|---|
| Streaming loader, connected subset, corpus stats | `src/data.py` | `python -m src.data stats` | |
| Preprocessing: tokenize, case fold, stop words, Porter (flag) | `src/preprocess.py` | `python -m src.preprocess` | |
| Inverted index: dictionary + postings, title/abstract zones, AND merge | `src/index.py` | `python -m src.index show query expansion` | |
| lnc.ltc weights, cosine, accumulators, heap top-K, zone weights | `src/rank.py` (`Ranker.cosine_scores`, `top_k`) | `python -m src.search ... --verbose` | |
| Index elimination (idf threshold) | `src/rank.py` (`cosine_scores`) | `python -m src.eval idf` | |
| Year (parametric) filter | `src/rank.py` (`search`, step 2) | compare `--year 2010` vs no `--year` | |
| Static quality g(d), net score, leakage control | `src/quality.py` | `--g-weight 0` vs default | |
| Champion lists | `src/rank.py` (`champions`, `cosine_scores_champion`) | `python -m src.eval champion` / `--champion-r 100` | |
| Co-citation boosting (novelty) | `src/cocitation.py` | `--cc-weight 0` vs default (watch "cited by k/10") | |
| Explanations + faithfulness | `src/explain.py` | CLI "why:" lines; `python -m src.eval faithfulness` | |
| BM25, PageRank | `src/bm25.py`, `src/quality.py` (`pagerank`) | `--bm25`, `--g-kind pagerank` | |
| Evaluation, tuning, significance | `src/eval.py` | `python -m src.eval ladder` | |
| Plots | `src/plots.py` | files in `results/` | |
| CLI | `src/search.py` | `python -m src.search --help` | |

## Where the report numbers are

- Corpus stats: README "Working subset", `results/subset_build_log.txt`
- Preprocessing: `results/preprocess_stats.csv`; index: `results/index_stats.csv`
- Ablation ladder: `results/ablation_ladder.csv` + `.png`, `results/recall_at_k.png`
- Index elimination: `results/idf_threshold_test.csv` + `.png`
- Champion lists: `results/champion_lists.csv` + `.png`
- Faithfulness: `results/faithfulness.csv` + `.png` (per-result rows: `faithfulness_rows_*.csv`)
- Tuning (dev only): `results/tuning_log_dev.csv`, `results/tuning_bm25_dev.csv`, `results/tuned_params.json`
- Scaling: `results/scaling.csv`, `results/log_scaling.txt`
- Raw console logs of every run: `results/log_*.txt`
