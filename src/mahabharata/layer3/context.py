"""Layer 3 context assembly — coarse-to-fine retrieval for synthesis.

Implements architecture-doc Layer 3 Choice 3: chapter-localize, then
verse-gather. Two stages, both reusing existing Layer 2 artifacts:

  Stage 1 (coarse): rank the 1,995 chapter summaries against the query
  with the chapter-dense index. This is the scene-localizer the canary
  arc validated (concept_003 -> its chapter at rank #1) — it recovers
  scenes that verse-dense buries at rank 7,000-53,000.

  Stage 2 (fine): pull the top verse-dense hits (with full translations)
  for the directly-quotable ground truth the model must cite.

The bundle hands the synthesizer both: chapter summaries for *where in
the epic this happens* and verses for *the exact lines to cite*. Neither
alone suffices — chapter-only loses verse citation, verse-only loses the
scene (the whole canary lesson).

This module does no generation; it only assembles context. Keeping
retrieval and synthesis separate keeps the context bundle inspectable
(you can print exactly what the model was shown) and testable without a
model in the loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mahabharata.layer2.chapter_dense import ChapterRetriever
from mahabharata.layer2.retriever import Retriever, Verse


@dataclass
class ChapterContext:
    chapter_uid: str
    parva: str
    book: int
    chapter: int
    verse_count: int
    summary: str  # the best-matching chunk of the chapter summary
    score: float


@dataclass
class ContextBundle:
    """Everything the synthesizer is shown, plus provenance."""

    query: str
    chapters: list[ChapterContext] = field(default_factory=list)
    verses: list[Verse] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chapters and not self.verses


class ContextBuilder:
    """Assembles a ContextBundle for a query.

    Composes a Layer 2 `Retriever` (verse-level concept search + corpus
    hydration) with a `ChapterRetriever` (chunked scene localization) and
    the Step 6 chapter summaries.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        chapter_dense: ChapterRetriever,
        chapter_summaries: dict[str, dict],
    ):
        self.retriever = retriever
        self.chapter_dense = chapter_dense
        self.chapter_summaries = chapter_summaries
        self._model_shared = False

    @classmethod
    def from_paths(
        cls,
        *,
        entities_path: Path,
        themes_path: Path,
        char_index_path: Path,
        group_index_path: Path,
        theme_index_path: Path,
        raw_path: Path,
        dense_embeddings_path: Path,
        dense_uids_path: Path,
        chapter_embeddings_path: Path,
        chapter_chunks_path: Path,
        chapter_summaries_path: Path,
    ) -> "ContextBuilder":
        retriever = Retriever.from_paths(
            entities_path=entities_path,
            themes_path=themes_path,
            char_index_path=char_index_path,
            group_index_path=group_index_path,
            theme_index_path=theme_index_path,
            raw_path=raw_path,
            dense_embeddings_path=dense_embeddings_path,
            dense_uids_path=dense_uids_path,
        )
        chapter_dense = ChapterRetriever.from_paths(
            embeddings_path=chapter_embeddings_path,
            chunks_path=chapter_chunks_path,
        )
        chapter_summaries: dict[str, dict] = {}
        with open(chapter_summaries_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                chapter_summaries[rec["chapter_uid"]] = rec
        return cls(
            retriever=retriever,
            chapter_dense=chapter_dense,
            chapter_summaries=chapter_summaries,
        )

    def _share_model_once(self) -> None:
        """Give the chapter retriever the verse bge model (same model_name).

        Layer 3 always issues a verse concept query too, so the verse
        model loads anyway — sharing avoids a second 130 MB copy + load.
        """
        if self._model_shared:
            return
        dense = self.retriever.dense
        if dense is not None:
            self.chapter_dense.set_model(dense.model)
        self._model_shared = True

    def build(
        self,
        query: str,
        *,
        n_chapters: int = 3,
        n_verses: int = 8,
    ) -> ContextBundle:
        notes: list[str] = []
        self._share_model_once()

        # Stage 1 — coarse chapter localization (chunk max-pool). The hit
        # carries the best-matching chunk text, so for a long chapter the
        # model sees the localized scene, not a head-truncated summary.
        chapters: list[ChapterContext] = []
        for hit in self.chapter_dense.search(query, top_k=n_chapters):
            rec = self.chapter_summaries.get(hit.chapter_uid, {})
            chapters.append(
                ChapterContext(
                    chapter_uid=hit.chapter_uid,
                    parva=rec.get("parva", ""),
                    book=rec.get("book", 0),
                    chapter=rec.get("chapter", 0),
                    verse_count=rec.get("verse_count", 0),
                    summary=hit.chunk_text,
                    score=hit.score,
                )
            )

        # Stage 2 — fine verse gathering. Reuse the verse concept lane.
        resp = self.retriever.search(query, limit=n_verses)
        verses = resp.results
        notes.extend(resp.notes)

        return ContextBundle(
            query=query,
            chapters=chapters,
            verses=verses,
            notes=notes,
        )
