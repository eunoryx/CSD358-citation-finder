"""
Stage 8 (our novelty): co-citation boosting = pseudo-relevance feedback over the citation graph.

Idea: the top-N text matches are probably on-topic. Papers that MANY of them cite
are likely the "standard references" of that topic, even when their own abstract
shares few words with the query (e.g. a classic paper with old terminology).

  1. run the normal ranking, take the top-N results (default N = 20)
  2. pool their reference lists (only references inside our corpus)
  3. cocite(d) = number of the top-N papers that cite d,  in [0, N]
  4. final(q, d) = net(q, d) + cc_weight * cocite(d) / N

Like classic pseudo-relevance feedback (Rocchio with the top results assumed
relevant), but the feedback signal is links, not terms. Papers that the top
results cite are added to the candidate set even if they share no query term.

No leakage: we only use reference lists of the RETRIEVED papers, which are all
older than the query (year filter) and never the held-out query paper itself.
"""
import numpy as np


class CoCitation:
    def __init__(self, index):
        self.N = index.N
        id2doc = {d["id"]: i for i, d in enumerate(index.docs)}
        # refs[d] = docIDs (inside the corpus) that paper d cites
        self.refs = [np.array(sorted({id2doc[r] for r in d["references"] if r in id2doc}), dtype=np.int32)
                     for d in index.docs]

    def counts(self, top_docs):
        """cocite(d) for every doc: how many of `top_docs` have d in their reference list."""
        if not top_docs:
            return np.zeros(self.N, dtype=np.float32)
        pooled = np.concatenate([self.refs[d] for d in top_docs])
        return np.bincount(pooled, minlength=self.N).astype(np.float32)

    def boost(self, top_docs):
        """Normalised to [0, 1] by the number of pooled papers."""
        return self.counts(top_docs) / max(len(top_docs), 1)
