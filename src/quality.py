"""
Stage 6 (+ stage 10 extra): static, query-independent quality g(d).

Net score (lecture: "net score = cosine + g(d)"):
    net(q, d) = cosine(q, d) + g_weight * g(d)
g(d) is in [0, 1] so g_weight directly controls how much popularity can override text match.

g_kind = "citations":  g(d) = log10(1 + c_d) / log10(1 + max_c)
    c_d = number of papers IN OUR CORPUS that cite d. The log damps the huge range
    (a few papers have thousands of citations; most have a handful).

g_kind = "pagerank":   g(d) = log10(1 + N * PR_d) / max  over the citation graph
    PageRank: a paper is important if important papers cite it.
    PR = (1 - a)/N + a * sum over citing papers p of PR_p / outdeg_p   (a = 0.85),
    plus the rank of papers with no out-links spread evenly ("dangling" nodes).

g_kind = "aminer":     g(d) uses AMiner's global n_citation field. ONLY for comparison:
    it counts citations from the whole world up to 2026, including the held-out
    query paper and papers written AFTER it -> leaks the answer during evaluation.

LEAKAGE CONTROL (important for honest evaluation)
  When we evaluate on a held-out paper q, its references are the right answers.
  If q's own citations were counted in c_d, every true reference would get +1
  citation "for free". So for every query we:
    1. subtract the citation edges coming from q (and its duplicates), and
    2. only count citations from papers published BEFORE q's year (time-aware):
       at the time q was written, later citations did not exist yet.
  (2) already implies (1) since q's year is not < q's year; we still do (1)
  explicitly so the CLI, which may get no year, is also leak-free.
"""
import numpy as np


class Quality:
    def __init__(self, index):
        self.N = index.N
        id2doc = {d["id"]: i for i, d in enumerate(index.docs)}
        src, dst = [], []
        for i, d in enumerate(index.docs):
            for r in set(d["references"]):
                j = id2doc.get(r)
                if j is not None and j != i:
                    src.append(i)
                    dst.append(j)
        # citation edges inside the corpus: paper src cites paper dst
        self.src = np.array(src, dtype=np.int32)
        self.dst = np.array(dst, dtype=np.int32)
        self.years = np.array([d["year"] for d in index.docs], dtype=np.int32)
        self.src_year = self.years[self.src]
        self.aminer = np.array([d["n_citation"] for d in index.docs], dtype=np.float64)
        self._cache = {}

    def edge_mask(self, year, exclude):
        """Which citation edges we are allowed to count for this query."""
        keep = np.ones(len(self.src), dtype=bool)
        if year is not None:
            keep &= self.src_year < year          # time-aware: only citations that existed before the query
        if len(exclude):
            keep &= ~np.isin(self.src, exclude)   # subtract the held-out paper's own references
        return keep

    def citation_counts(self, year=None, exclude=()):
        keep = self.edge_mask(year, np.asarray(exclude))
        return np.bincount(self.dst[keep], minlength=self.N).astype(np.float64)

    def pagerank(self, year=None, exclude=(), alpha=0.85, iters=50, tol=1e-9):
        """Power iteration on the citation graph (only allowed edges)."""
        keep = self.edge_mask(year, np.asarray(exclude))
        src, dst = self.src[keep], self.dst[keep]
        N = self.N
        outdeg = np.bincount(src, minlength=N).astype(np.float64)
        dangling = outdeg == 0
        pr = np.full(N, 1.0 / N)
        for _ in range(iters):
            share = pr[src] / outdeg[src]                    # each citing paper splits its rank over its references
            new = np.bincount(dst, weights=share, minlength=N)
            new = alpha * (new + pr[dangling].sum() / N) + (1 - alpha) / N
            if np.abs(new - pr).sum() < tol:
                pr = new
                break
            pr = new
        return pr

    def g(self, kind="citations", year=None, exclude=()):
        """g(d) for all docs, scaled to [0, 1]. Cached per (kind, year, excluded docs)."""
        key = (kind, year, tuple(sorted(int(x) for x in exclude)))
        if key in self._cache:
            return self._cache[key]
        if kind == "citations":
            raw = np.log10(1 + self.citation_counts(year, exclude))
        elif kind == "pagerank":
            raw = np.log10(1 + self.N * self.pagerank(year, exclude))
        elif kind == "aminer":
            raw = np.log10(1 + self.aminer)
        else:
            raise ValueError(kind)
        g = (raw / raw.max() if raw.max() > 0 else raw).astype(np.float32)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = g
        return g
