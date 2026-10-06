"""
Stage 10 extra: Okapi BM25, as a comparison to lnc.ltc cosine.

For each zone (title / abstract) separately:
    BM25(q, d) = sum over distinct query terms t present in d of
                 idf_t * tf_td * (k1 + 1) / (tf_td + k1 * (1 - b + b * len_d / avg_len))
    idf_t = ln( (N - df_t + 0.5) / (df_t + 0.5) + 1 )

  k1 (default 1.2, tuned on dev) controls tf saturation: the 10th occurrence of a term adds much less than the 1st.
  b  (default 0.75, tuned on dev) controls document length normalisation: long abstracts are penalised.
Zones are combined like the cosine model: title_w * BM25_title + (1 - title_w) * BM25_abstract.
The same index elimination threshold is applied (on log10 N/df, so thresholds are comparable).

BM25 scores are not in [0, 1] like cosine. So that the SAME g_weight / cc_weight can be
added on top, we divide the BM25 score by the query's maximum BM25 score (top doc = 1).
"""
import math

import numpy as np

from src.index import ZONES


class BM25:
    def __init__(self, ranker):
        self.r = ranker
        idx = ranker.index
        self.len = {}
        self.avg = {}
        self.idf = {}
        for zone in ZONES:
            L = np.zeros(idx.N, dtype=np.float64)
            for ids, tfs in idx.postings[zone].values():
                np.add.at(L, ids, tfs)        # document length = number of index terms in the zone
            self.len[zone] = L
            self.avg[zone] = L.mean()
            self.idf[zone] = {t: math.log((idx.N - len(ids) + 0.5) / (len(ids) + 0.5) + 1)
                              for t, (ids, _) in idx.postings[zone].items()}

    def scores(self, terms, cfg):
        idx = self.r.index
        acc = np.zeros(idx.N, dtype=np.float32)
        for zone, zw in self.r.zone_weights(cfg).items():
            if zw == 0:
                continue
            for t in set(terms):
                if t not in idx.postings[zone] or self.r.idf[zone][t] < cfg.idf_threshold:
                    continue
                ids, tfs = idx.postings[zone][t]
                tf = tfs.astype(np.float32)
                k1, b = cfg.bm25_k1, cfg.bm25_b
                norm = k1 * (1 - b + b * self.len[zone][ids] / self.avg[zone])
                acc[ids] += zw * self.idf[zone][t] * tf * (k1 + 1) / (tf + norm)
        m = acc.max()
        return acc / m if m > 0 else acc
