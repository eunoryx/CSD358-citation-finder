"""
Stage 1: text preprocessing (the "linguistic modules" before indexing).

Pipeline, applied identically to documents AND queries (if they were processed
differently, query terms would not match index terms):

  raw text
    -> normalise   : unescape HTML (&amp; -> &), strip accents (naive -> naive)
    -> tokenize    : split into runs of letters/digits
    -> case fold   : lowercase everything ("Retrieval" == "retrieval")
    -> drop junk   : tokens of length 1 and pure numbers carry little meaning
    -> stop words  : remove very frequent function words ("the", "of", "we")
    -> stemming    : Porter stemmer conflates variants
                     ("retrieval", "retrieve", "retrieving" -> "retriev")

Stemming is a flag (stem=True/False) so we can ablate it: it raises recall
(more variants match) but can hurt precision ("university"/"universe" -> "univers").

Run `python -m src.preprocess` to print statistics on the working subset.
"""
import functools
import html
import re
import unicodedata

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # a fixed list of 318 English stop words

# Extra stop words that are frequent in scientific abstracts but say nothing about the topic.
# Measured on the 48k subset: each appears in 5-47% of abstracts
# (e.g. 'based' 47%, 'paper' 47%, 'propose' 42%, 'novel' 23%), so they barely discriminate.
ABSTRACT_STOP_WORDS = {
    "paper", "propose", "proposed", "approach", "method", "methods", "results", "result",
    "show", "shows", "based", "using", "use", "used", "new", "novel", "existing",
    "work", "present", "study", "experiments", "experimental", "demonstrate", "effectiveness",
}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | ABSTRACT_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Porter's original 1980 algorithm (nltk's default mode adds extra rules; we want the textbook one).
_porter = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)


@functools.lru_cache(maxsize=500_000)
def stem(token):
    """Cached: the same word appears thousands of times, so stem each distinct word once."""
    return _porter.stem(token)


def normalise(text):
    """HTML unescape + strip accents, so 'na&iuml;ve' and 'naïve' both become 'naive'."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text):
    """Case fold, then split on anything that is not a letter or digit.
    'Learning-to-Rank (LTR)' -> ['learning', 'to', 'rank', 'ltr']"""
    return TOKEN_RE.findall(normalise(text).lower())


def preprocess(text, stem_on=True, remove_stop=True):
    """Full pipeline: text -> list of index terms (order kept, duplicates kept, so tf can be counted)."""
    terms = []
    for tok in tokenize(text):
        if len(tok) < 2 or tok.isdigit():
            continue
        if remove_stop and tok in STOP_WORDS:
            continue
        terms.append(stem(tok) if stem_on else tok)
    return terms


if __name__ == "__main__":
    import collections
    import csv
    import os
    import time

    from src.data import load_jsonl

    papers = load_jsonl()
    print(f"{len(papers):,} papers")

    example = papers[0]["title"] + ". " + papers[0]["abstract"][:200]
    print("\nExample:", example)
    print(" tokens          :", tokenize(example)[:25])
    print(" -stop, no stem  :", preprocess(example, stem_on=False)[:25])
    print(" -stop, + Porter :", preprocess(example, stem_on=True)[:25])

    rows = []
    for name, kwargs in [("tokenize only", dict(stem_on=False, remove_stop=False)),
                         ("+ stop words", dict(stem_on=False, remove_stop=True)),
                         ("+ stop words + Porter", dict(stem_on=True, remove_stop=True))]:
        t0 = time.time()
        vocab = collections.Counter()  # document frequency of each term
        n_tokens = 0
        for p in papers:
            terms = preprocess(p["title"] + " " + p["abstract"], **kwargs)
            n_tokens += len(terms)
            vocab.update(set(terms))
        secs = time.time() - t0
        singletons = sum(1 for c in vocab.values() if c == 1)
        rows.append([name, n_tokens, round(n_tokens / len(papers), 1), len(vocab), singletons, round(secs, 1)])
        print(f"\n{name}: {n_tokens:,} tokens ({n_tokens/len(papers):.1f}/doc), vocab {len(vocab):,} "
              f"({singletons:,} appear in only 1 doc), {secs:.1f}s")
        print("  highest-df terms:", [f"{t}:{c}" for t, c in vocab.most_common(15)])

    os.makedirs("results", exist_ok=True)
    with open("results/preprocess_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "tokens", "tokens_per_doc", "vocab_size", "df1_terms", "seconds"])
        w.writerows(rows)
    print("\nsaved results/preprocess_stats.csv")
