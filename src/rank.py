"""
Stages 3-5: vector-space ranking.

STAGE 3 - lnc.ltc weighting + cosine similarity + heap top-K + zone weights
  SMART notation "ddd.qqq" = document weighting . query weighting, each 3 letters:
    l = logarithmic tf:   1 + log10(tf)        (tf=0 -> 0)
    n = no idf for documents  (idf is applied once, on the query side)
    t = idf for the query:  log10(N / df)
    c = cosine normalisation (divide by the vector's Euclidean length)
  So a document term weight is   w_td = (1 + log10 tf_td) / ||d||
  and a query term weight is     w_tq = (1 + log10 tf_tq) * idf_t / ||q||
  cosine(q, d) = sum over shared terms of w_tq * w_td.

  We score TERM-AT-A-TIME with an accumulator array: for each query term, walk its
  postings list and add w_tq * w_td to acc[docID]. Docs that share no term with the
  query keep acc = 0 and are never candidates.

  Zones: we compute a cosine per zone (title, abstract) and combine
     zone_score = title_w * cos_title + abstract_w * cos_abstract.

  Top-K uses a heap (heapq.nlargest keeps a min-heap of size K): O(n log K)
  instead of sorting all n candidates.

STAGE 4 - index elimination for long queries
  A query here is a whole abstract (~100 terms). Low-idf terms ("model", "data")
  have huge postings lists but add little to the score. We only walk postings of
  terms with idf >= idf_threshold. The query vector norm still uses ALL query
  terms, so a kept term has the same weight it would have without elimination:
  the scores are an approximation from below of the full cosine.

STAGE 5 - parametric (year) filter
  Only papers published strictly before the query paper's year can be cited, so
  docs with year >= query year are removed from the candidate set.

Stages 6 (static quality g(d)) and 8 (co-citation) plug in through `search()`;
their maths lives in src/quality.py and src/cocitation.py.
"""
import collections
import heapq
import math
from dataclasses import dataclass, replace

import numpy as np

from src.index import ZONES
from src.preprocess import preprocess


@dataclass
class Config:
    """All the knobs of one ranking run. The eval ablation ladder toggles these."""
    use_title: bool = True        # query title + title zone (False = abstract-only baseline)
    title_w: float = 0.3          # zone weights: score = title_w*cos_title + (1-title_w)*cos_abstract
    idf_threshold: float = 0.0    # stage 4: skip query terms with idf below this (0 = no elimination)
    year_filter: bool = True      # stage 5
    g_weight: float = 0.0         # stage 6: net = cosine + g_weight * g(d)
    g_kind: str = "citations"     # "citations" (log in-corpus count), "pagerank", or "aminer" (leaky, for comparison)
    cc_weight: float = 0.0        # stage 8: co-citation boost weight
    cc_top_n: int = 20            # stage 8: how many top results' reference lists to pool
    model: str = "cosine"         # "cosine" (lnc.ltc) or "bm25" (stage 10 comparison)
    champion_r: int = 0           # stage 7: >0 = use champion lists of size r
    champion_exact: bool = False  # stage 7: False = score from champion postings only; True = exact rescoring of the union
    bm25_k1: float = 1.2          # stage 10: BM25 tf saturation
    bm25_b: float = 0.75          # stage 10: BM25 length normalisation

    def but(self, **kw):
        return replace(self, **kw)


class Ranker:
    def __init__(self, index):
        self.index = index
        N = index.N
        self.years = np.array([d["year"] for d in index.docs], dtype=np.int32)
        self.idf = {}    # zone -> {term: idf}
        self.wnorm = {}  # zone -> {term: array of w_td = (1+log10 tf)/||d|| aligned with the postings}
        for zone in ZONES:
            plist = index.postings[zone]
            # idf_t = log10(N / df_t)
            self.idf[zone] = {t: math.log10(N / len(ids)) for t, (ids, _) in plist.items()}
            # lnc document weights: first the squared lengths ||d||^2 = sum_t (1 + log10 tf)^2 ...
            sq = np.zeros(N, dtype=np.float64)
            for ids, tfs in plist.values():
                np.add.at(sq, ids, (1 + np.log10(tfs)) ** 2)
            norm = np.sqrt(sq)
            norm[norm == 0] = 1.0  # docs with an empty zone: avoid division by zero
            # ... then each posting's normalised weight.
            self.wnorm[zone] = {t: ((1 + np.log10(tfs)) / norm[ids]).astype(np.float32)
                                for t, (ids, tfs) in plist.items()}
        self._champions = {}  # stage 7 cache: zone -> {term: docIDs sorted by weight, descending}
        self.bm25 = None      # built lazily by src/bm25.py
        self.quality = None   # set by src/quality.py
        self.cocite = None    # set by src/cocitation.py

    # ---- query side ------------------------------------------------------
    def query_terms(self, title, abstract, use_title=True):
        text = (title + " " + abstract) if use_title else abstract
        return preprocess(text, stem_on=self.index.stem_on)

    def query_weights(self, terms, zone):
        """ltc query vector for one zone: {term: (1 + log10 tf_q) * idf_t / ||q||}.
        Terms that never occur in this zone have no idf and get weight 0 (dropped)."""
        tf = collections.Counter(terms)
        idf = self.idf[zone]
        w = {t: (1 + math.log10(c)) * idf[t] for t, c in tf.items() if t in idf}
        length = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return {t: x / length for t, x in w.items()}

    def zone_weights(self, cfg):
        if not cfg.use_title:
            return {"abstract": 1.0}
        return {"title": cfg.title_w, "abstract": 1.0 - cfg.title_w}

    # ---- stage 3 + 4: cosine scores for all docs (term-at-a-time) ------
    def cosine_scores(self, terms, cfg):
        """Returns an array of zone-weighted cosine scores for every docID."""
        acc = np.zeros(self.index.N, dtype=np.float32)
        for zone, zw in self.zone_weights(cfg).items():
            if zw == 0:
                continue
            for t, wq in self.query_weights(terms, zone).items():
                if self.idf[zone][t] < cfg.idf_threshold:   # stage 4: index elimination
                    continue
                ids, _ = self.index.postings[zone][t]
                acc[ids] += zw * wq * self.wnorm[zone][t]  # each docID appears once per list, so += is safe
        return acc

    # ---- stage 7: champion lists ---------------------------------------
    def champions(self, zone, term):
        """Postings of `term` re-ordered by w_td, highest first: (docIDs, weights).
        The champion list of size r is the first r entries: the r docs where this
        term weighs the most. Computed once per term (offline in a real system)."""
        cache = self._champions.setdefault(zone, {})
        if term not in cache:
            ids, _ = self.index.postings[zone][term]
            w = self.wnorm[zone][term]
            order = np.argsort(-w, kind="stable")
            cache[term] = (ids[order], w[order])
        return cache[term]

    def precompute_champions(self):
        for zone in ZONES:
            for t in self.index.postings[zone]:
                self.champions(zone, t)

    def cosine_scores_champion(self, terms, cfg):
        """Champion-list scoring. Two variants:

        champion_exact=False (textbook "approximate" scoring): walk only the first r
          postings of each query term's champion list. Cost = r postings per term
          instead of df. A doc's score only counts the terms for which it is a champion.
        champion_exact=True: the union of the champion lists is the CANDIDATE set,
          then we compute the exact cosine for those candidates (binary search into
          each full postings list). Better scores, but more work per candidate.

        Returns (scores for all docs, postings touched)."""
        acc = np.zeros(self.index.N, dtype=np.float32)
        touched = 0
        zws = self.zone_weights(cfg)
        qws = {z: {t: w for t, w in self.query_weights(terms, z).items() if self.idf[z][t] >= cfg.idf_threshold}
               for z, zw in zws.items() if zw > 0}
        r = cfg.champion_r
        if not cfg.champion_exact:
            for zone, qw in qws.items():
                for t, wq in qw.items():
                    ids, w = self.champions(zone, t)
                    acc[ids[:r]] += zws[zone] * wq * w[:r]
                    touched += min(r, len(ids))
            return acc, touched
        parts = [self.champions(z, t)[0][:r] for z, qw in qws.items() for t in qw]
        if not parts:
            return acc, 0
        cand = np.unique(np.concatenate(parts))
        sub = np.zeros(len(cand), dtype=np.float32)
        for zone, qw in qws.items():
            for t, wq in qw.items():
                ids, _ = self.index.postings[zone][t]
                pos = np.searchsorted(ids, cand)          # where each candidate would be in this list
                pos[pos == len(ids)] = 0
                hit = ids[pos] == cand                    # candidate actually contains the term
                sub[hit] += zws[zone] * wq * self.wnorm[zone][t][pos[hit]]
                touched += len(cand)
        acc[cand] = sub
        return acc, touched

    # ---- full pipeline ---------------------------------------------------
    def search(self, title, abstract, year=None, cfg=Config(), k=100, exclude=(), return_scores=False):
        """Rank docs for a query paper. `exclude` = docIDs never to return (the
        held-out query paper itself and its duplicates during evaluation).

        Returns a list of (docID, score) of length <= k, best first
        (and, if return_scores, the full score array used for ranking)."""
        terms = self.query_terms(title, abstract, cfg.use_title)
        N = self.index.N

        # 1. text match score (stage 3/4, or BM25 for stage 10, or champion lists for stage 7)
        if cfg.champion_r > 0:
            text, _ = self.cosine_scores_champion(terms, cfg)
        elif cfg.model == "bm25":
            text = self.bm25.scores(terms, cfg)
        else:
            text = self.cosine_scores(terms, cfg)

        # 2. candidate set = docs that matched at least one query term,
        #    minus excluded docs, minus docs not older than the query (stage 5)
        valid = text > 0
        if cfg.year_filter and year is not None:
            valid &= self.years < year
        exclude = np.asarray(list(exclude), dtype=np.int64)
        valid[exclude] = False

        # 3. stage 6: net score = text score + g_weight * g(d)   (only for candidates)
        score = text.copy()
        if cfg.g_weight > 0:
            score += cfg.g_weight * self.quality.g(cfg.g_kind, year, exclude)

        # 4. stage 8: co-citation boost from the reference lists of the top-N results
        if cfg.cc_weight > 0:
            top_n = self.top_k(score, valid, cfg.cc_top_n)
            boost = self.cocite.boost([d for d, _ in top_n])
            score = score + cfg.cc_weight * boost
            valid_cc = boost > 0                       # papers cited by the top results become candidates too
            if cfg.year_filter and year is not None:
                valid_cc &= self.years < year
            valid_cc[exclude] = False
            valid = valid | valid_cc

        results = self.top_k(score, valid, k)
        return (results, score, valid) if return_scores else results

    @staticmethod
    def top_k(score, valid, k):
        """Heap-based top-K over the candidate set (heapq.nlargest = size-k min-heap)."""
        cand = np.flatnonzero(valid)
        best = heapq.nlargest(k, zip(score[cand].tolist(), cand.tolist()))
        return [(d, s) for s, d in best]


def build_ranker(stem_on=True, index_path=None):
    """Load the saved index and attach every scoring component (one place, used by eval and CLI)."""
    from src.bm25 import BM25
    from src.cocitation import CoCitation
    from src.index import InvertedIndex
    from src.quality import Quality

    index = InvertedIndex.load(stem_on=stem_on, path=index_path)
    r = Ranker(index)
    r.quality = Quality(index)
    r.cocite = CoCitation(index)
    r.bm25 = BM25(r)
    return r
