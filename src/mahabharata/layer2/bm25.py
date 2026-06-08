"""Layer 2 Phase C / Step 5 — BM25 lexical retrieval.

Indexes the fluent English translations of all corpus verses with the
classical Okapi BM25 formula (rank_bm25, ~200 LOC pure Python — no
torch / transformers / sentence-transformers). Used by the retriever's
`concept` mode to answer queries that have no structural anchor and
no facet keyword.

Why BM25 here
-------------
Layer 2 Choice 1 (route to the cheapest mode that works) sets the
pattern: only invoke the more expensive lane when cheaper lanes can't
answer. BM25 is the cheapest of the free-text lanes — pure Python,
sub-millisecond per query at our corpus size, no model deps.

It's also the right anchor for measuring whatever comes after.
Architecture doc Choice 5/6 of Layer 2 commits to dense retrieval
(bge-small-en-v1.5) + RRF fusion as the next steps; both decisions
need a baseline to be measured against, and BM25 is the standard IR
baseline for exactly that role.

Tokenization
------------
Lowercase + extract alphanumeric runs + drop English stopwords. No
stemming — translations are short (~25 tokens median) and stemming
gains are marginal at this corpus size; revisit if eval flags
morphological misses.

The stopword list was added after the initial no-stopwords build
showed mean recall@10 = 0.0 on concept eval items. The diagnosis:
at this corpus size, IDF alone failed to suppress short common
words. A query like "Why should we not mourn the dead?" was
dominated by docs sharing many of "why/should/we/not/the" rather
than the meaningful "mourn"/"dead". Removing the stopwords from
both index and query restores BM25 to scoring on content terms.

Single-pass build
-----------------
Index built in memory at retriever construction time. For 73,820
verses × ~25 tokens each, build is ~1-2 seconds and a few tens of
MB of RAM. No on-disk caching yet — if startup latency becomes
annoying we can pickle the BM25Okapi instance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short, hand-curated English stopword list. Larger lists (NLTK's
# ~180-word list) over-prune for this corpus — terms like "all",
# "now", "very" carry retrieval signal here. This is the minimal
# core of true function words plus question/auxiliary words that
# polluted the initial concept eval.
STOPWORDS = frozenset({
    "a", "an", "the",
    "and", "or", "but", "if", "then", "so", "as",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "into", "onto", "upon", "about",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "do", "does", "did", "done",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    "this", "that", "these", "those",
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how",
    "not", "no", "nor",
    "there", "here",
})


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric-run split + stopword removal.

    Used identically for both index and query — keeps DF counts
    consistent with what the scorer actually sees.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class BM25Hit:
    uid: str
    score: float


class BM25Retriever:
    """Wraps rank_bm25.BM25Okapi with parallel uid lookup.

    The list `uids` and the list `texts` must be parallel — uids[i]
    identifies the document texts[i]. Internal state is the tokenized
    corpus and a fitted BM25Okapi instance.
    """

    def __init__(self, uids: list[str], texts: list[str]):
        if len(uids) != len(texts):
            raise ValueError(
                f"uids ({len(uids)}) and texts ({len(texts)}) "
                "must be parallel lists"
            )
        self.uids = list(uids)
        self.tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, *, top_k: int = 100) -> list[BM25Hit]:
        """Return up to top_k hits sorted by descending BM25 score.

        Hits with score <= 0 are filtered (no query token matched).
        Empty query (no tokens after tokenization) returns [] without
        scoring the corpus.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            ((float(s), i) for i, s in enumerate(scores) if s > 0.0),
            reverse=True,
        )[:top_k]
        return [BM25Hit(uid=self.uids[i], score=s) for s, i in ranked]
