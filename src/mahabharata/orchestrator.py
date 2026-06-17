"""Unified front door over the whole stack.

Turns the toolbox (separate `mbh-query` lookup + `mbh-ask` synthesis)
into one system: classify any query once, dispatch to the lane its shape
needs, and return a single coherent response. This is the "agentic
routing" the project set out to learn — and the integration layer that
lets the system be *used* end-to-end (so real failures, not hand-picked
canaries, drive what to fix next).

Dispatch policy (deliberately simple — conditional-invocation tuning is
an open question, architecture-doc Layer 3 Choice 2)::

    structural_uid / structural_slice / facet / lexical  -> lookup
        (return verses directly; no model, deterministic, instant)
    concept                                              -> synthesis
        (assemble chapter + verse context, generate a grounded answer)

"All concept queries get synthesized" is the dumb-but-honest v1 rule: we
do not yet try to split concept-lookup from concept-reasoning (the same
ambiguity that forced quote-gating). Synthesis is gated behind a flag so
a caller can always force pure lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mahabharata.layer2.router import classify
from mahabharata.layer2.retriever import Verse
from mahabharata.layer3.context import ChapterContext, ContextBuilder
from mahabharata.layer3.synthesize import Synthesizer

# Router modes that route to synthesis rather than a direct verse lookup.
SYNTH_MODES = {"concept"}


@dataclass
class OrchestratorResponse:
    query: str
    mode: str               # router mode (structural_uid/.../concept)
    kind: str               # "lookup" | "synthesis"
    verses: list[Verse]
    total: int
    answer: str | None = None
    cited_uids: list[str] = field(default_factory=list)
    abstained: bool = False
    chapters: list[ChapterContext] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, *, builder: ContextBuilder, synth: Synthesizer):
        self.builder = builder
        self.synth = synth

    @classmethod
    def from_paths(
        cls, *, synth_model: str | None = None, stream: bool = True
    ) -> "Orchestrator":
        from mahabharata.common import paths

        builder = ContextBuilder.from_paths(
            entities_path=paths.ENTITIES_PATH,
            themes_path=paths.THEMES_PATH,
            char_index_path=paths.CHAR_INDEX_PATH,
            group_index_path=paths.GROUP_INDEX_PATH,
            theme_index_path=paths.THEME_INDEX_PATH,
            raw_path=paths.RAW_PATH,
            dense_embeddings_path=paths.DENSE_EMBEDDINGS_PATH,
            dense_uids_path=paths.DENSE_UIDS_PATH,
            chapter_embeddings_path=paths.CHAPTER_DENSE_EMBEDDINGS_PATH,
            chapter_chunks_path=paths.CHAPTER_DENSE_CHUNKS_PATH,
            chapter_summaries_path=paths.CHAPTER_SUMMARIES_PATH,
        )
        synth_kwargs: dict = {"stream_to_stderr": stream}
        if synth_model:
            synth_kwargs["model"] = synth_model
        synth = Synthesizer(**synth_kwargs)
        return cls(builder=builder, synth=synth)

    def answer(
        self,
        query: str,
        *,
        limit: int = 10,
        synthesize: bool = True,
        n_chapters: int = 3,
        n_verses: int = 8,
    ) -> OrchestratorResponse:
        plan = classify(query, self.builder.retriever.gazetteer)

        if plan.mode in SYNTH_MODES and synthesize:
            bundle = self.builder.build(
                query, n_chapters=n_chapters, n_verses=n_verses
            )
            result = self.synth.answer(bundle)
            return OrchestratorResponse(
                query=query,
                mode=plan.mode,
                kind="synthesis",
                verses=bundle.verses,
                total=len(bundle.verses),
                answer=result.answer,
                cited_uids=result.cited_uids,
                abstained=result.abstained,
                chapters=bundle.chapters,
                notes=bundle.notes,
            )

        resp = self.builder.retriever.search(query, limit=limit)
        return OrchestratorResponse(
            query=query,
            mode=resp.mode,
            kind="lookup",
            verses=resp.results,
            total=resp.total,
            notes=resp.notes,
        )
