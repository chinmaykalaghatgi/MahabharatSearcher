"""Layer 2 Phase C / Step 6 — Dense bi-encoder retrieval.

Loads a precomputed (N, D) embedding matrix + parallel UID list and
serves cosine-similarity queries against it. Embeddings are produced
offline by `mbh-build-embeddings`; this module is the runtime side.

Why dense after BM25
--------------------
BM25 (Step 5) failed all 15 concept eval items at recall@10 = 0.000.
Diagnosis: the curated known_goods talk about "grieve" when the query
says "mourn", "slay" when the query says "fight", "soul/garments" when
the query says "death" — vocabulary mismatch is structural, not
something IDF or stopword tuning can fix. Dense retrieval places
semantically similar phrases close in vector space without requiring
shared tokens, which is exactly what these concept queries need.

Index format
------------
Stored as a single `embeddings.npy` of shape (N, D), already
L2-normalized so cosine similarity = inner product. Parallel
`uids.txt` is one UID per line in the same order as matrix rows.
Brute-force argsort (per architecture doc Choice 5) — at N=73,820
× D=384 the matmul is single-digit ms on CPU, no FAISS needed.

Model loading
-------------
The SentenceTransformer model is loaded lazily on first query, not
at retriever construction. Loading takes ~1-2 seconds for
bge-small-en-v1.5 and would penalize one-shot CLI invocations that
never issue a concept query (e.g. `mbh-query B6_C24_S11`). The eval
harness amortizes the load over 15 concept items.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class DenseHit:
    uid: str
    score: float


class DenseRetriever:
    def __init__(
        self,
        uids: list[str],
        embeddings: np.ndarray,
        *,
        model_name: str = DEFAULT_MODEL,
    ):
        if len(uids) != embeddings.shape[0]:
            raise ValueError(
                f"uids ({len(uids)}) and embeddings ({embeddings.shape[0]}) "
                "must have parallel length"
            )
        self.uids = list(uids)
        self.embeddings = embeddings  # (N, D), expected L2-normalized
        self.model_name = model_name
        self._model = None  # lazy

    @property
    def model(self):
        if self._model is None:
            # Imported lazily so the dep is only required when the
            # dense path is actually invoked. Lets the rest of the
            # retriever load even if sentence-transformers isn't
            # installed yet (useful during partial-build dev cycles).
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def search(self, query: str, *, top_k: int = 100) -> list[DenseHit]:
        """Encode query, cosine-rank against the embedding matrix."""
        if not query.strip():
            return []
        q_emb = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Both sides L2-normalized → cosine sim == dot product
        scores = self.embeddings @ q_emb  # shape (N,)

        # Top-k by score. argpartition is O(N) vs O(N log N) for sort.
        if top_k >= len(scores):
            ranked_idx = np.argsort(-scores)
        else:
            partition = np.argpartition(-scores, top_k)[:top_k]
            ranked_idx = partition[np.argsort(-scores[partition])]

        return [
            DenseHit(uid=self.uids[i], score=float(scores[i]))
            for i in ranked_idx
        ]

    @classmethod
    def from_paths(
        cls,
        *,
        embeddings_path: Path,
        uids_path: Path,
        model_name: str = DEFAULT_MODEL,
    ) -> "DenseRetriever":
        embeddings = np.load(embeddings_path)
        with open(uids_path) as f:
            uids = [line.strip() for line in f if line.strip()]
        return cls(uids, embeddings, model_name=model_name)
