"""Chapter-level dense retrieval with chunk max-pooling.

The chapter index (`mbh-build-chapter-embeddings`) embeds each chapter
summary as several <=N-char chunks rather than one truncated vector. At
query time we score every chunk, then max-pool back to the chapter: a
chapter ranks by its single best-matching chunk. This recovers
mid-chapter scenes that a single whole-summary vector lost to bge's
512-token truncation (the L3-eval failure: chapter context recall ~0.07).

`search` returns distinct chapters (not chunks), each carrying the text
of its best-matching chunk — that chunk, not a head-truncated summary, is
what Layer 3 shows the model, so the localized scene is actually present
in the context window.

Brute-force matmul + per-chapter max, same scale rationale as
`layer2.dense` (architecture doc Choice 5): a few thousand chunks is a
single-digit-ms matmul on CPU, no ANN index needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class ChapterHit:
    chapter_uid: str
    score: float
    chunk_idx: int
    chunk_text: str


class ChapterRetriever:
    def __init__(
        self,
        chunk_chapter_uids: list[str],
        chunk_texts: list[str],
        embeddings: np.ndarray,
        *,
        chunk_idxs: list[int] | None = None,
        model_name: str = DEFAULT_MODEL,
    ):
        n = embeddings.shape[0]
        if not (len(chunk_chapter_uids) == len(chunk_texts) == n):
            raise ValueError(
                "chunk uids, texts, and embeddings must be row-parallel: "
                f"{len(chunk_chapter_uids)}, {len(chunk_texts)}, {n}"
            )
        self.chunk_chapter_uids = chunk_chapter_uids
        self.chunk_texts = chunk_texts
        self.chunk_idxs = chunk_idxs or [0] * n
        self.embeddings = embeddings  # (N_chunks, D), L2-normalized
        self.model_name = model_name
        self._model = None  # lazy

        # chapter_uid -> list of chunk row indices, for max-pooling.
        self._rows_by_chapter: dict[str, list[int]] = {}
        for i, uid in enumerate(chunk_chapter_uids):
            self._rows_by_chapter.setdefault(uid, []).append(i)

    @property
    def n_chapters(self) -> int:
        return len(self._rows_by_chapter)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def set_model(self, model) -> None:
        """Share an already-loaded SentenceTransformer (same model_name).

        Layer 3 always loads the verse bge model too; sharing avoids a
        second copy in RAM and a second load.
        """
        self._model = model

    def search(self, query: str, *, top_k: int = 3) -> list[ChapterHit]:
        if not query.strip():
            return []
        q_emb = self.model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
        scores = self.embeddings @ q_emb  # (N_chunks,)

        hits: list[ChapterHit] = []
        for uid, rows in self._rows_by_chapter.items():
            best = max(rows, key=lambda i: scores[i])
            hits.append(
                ChapterHit(
                    chapter_uid=uid,
                    score=float(scores[best]),
                    chunk_idx=self.chunk_idxs[best],
                    chunk_text=self.chunk_texts[best],
                )
            )
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    @classmethod
    def from_paths(
        cls,
        *,
        embeddings_path: Path,
        chunks_path: Path,
        model_name: str = DEFAULT_MODEL,
    ) -> "ChapterRetriever":
        embeddings = np.load(embeddings_path)
        chapter_uids: list[str] = []
        texts: list[str] = []
        chunk_idxs: list[int] = []
        with open(chunks_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                chapter_uids.append(rec["chapter_uid"])
                texts.append(rec["text"])
                chunk_idxs.append(rec.get("chunk_idx", 0))
        return cls(
            chapter_uids,
            texts,
            embeddings,
            chunk_idxs=chunk_idxs,
            model_name=model_name,
        )
