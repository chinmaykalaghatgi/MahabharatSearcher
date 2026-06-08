"""Layer 2 retriever — Phase A + C Step 5.

Loads Layer 1 artifacts once, then serves queries via:

  - direct UID lookup          ("B6_C27_S29")
  - book / chapter slice       ("B6", "B6_C27")
  - facet index intersection   ("Krishna dharma", "Arjuna Bhishma")
  - lexical (BM25)             quoted queries (`"iron mace"`)
  - concept (dense bi-encoder) free-text / paraphrase queries

Route-by-shape, not fusion
--------------------------
Lexical (exact-keyword) and concept (paraphrase) queries go to the
single lane that wins on that shape: BM25 for lexical, dense for
concept. A 2026-06-08 eval showed BM25 beats dense ~4x on lexical and
dense beats BM25 ~4x on concept, and that equal-weight RRF fusion
*regressed both shapes* by diluting the correct lane. The fusion module
(`mahabharata.layer2.fusion`) is kept for a future genuine "mixed"
route but is not on the default path. If the dense index isn't loaded,
concept degrades to BM25.

Loading strategy
----------------
The raw corpus is loaded into an in-memory `uid -> record` dict once
at construction. At 73,820 verses this is a small hit and saves us
from per-query file scans. Tests and repeated CLI invocations pay
the load cost once; interactive sessions amortize it across many
queries.

Group expansion
---------------
Per Layer 1 Choice 3 (atomic group tagging, expand at query time),
a group facet match unions its own tagged verses with the verses of
every member character in `entities.json`. This gives queries like
"Pandavas" sensible recall without Layer 1 having fabricated
per-member tags on collective references.

Intersection vs union fallback
------------------------------
Multi-facet queries intersect by default — "Arjuna dharma" wants
verses tagged with both. If the intersection is empty we fall back
to the union and attach a note, so the user sees something useful
instead of a blank screen. This is a v1 convenience; fancier fallback
policies wait for eval evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from mahabharata.common.corpus_loader import stream_corpus
from mahabharata.common.sections import parse_sections
from mahabharata.layer2.bm25 import BM25Retriever, tokenize as bm25_tokenize
from mahabharata.layer2.dense import DenseRetriever
from mahabharata.layer2.router import Gazetteer, QueryPlan, classify


_UID_PREFIX_RE = re.compile(r"^B(\d+)_C(\d+)_S(\d+)")


@dataclass
class Verse:
    uid: str
    book: int
    chapter: int
    shloka: int
    sanskrit: str
    translation: str
    score: float | None = None  # set by lexical (BM25) + concept (dense)


@dataclass
class RetrievalResponse:
    query: str
    mode: str
    plan: QueryPlan
    results: list[Verse]
    total: int
    notes: list[str] = field(default_factory=list)


class Retriever:
    def __init__(
        self,
        *,
        entities: dict,
        themes: dict,
        char_index: dict,
        group_index: dict,
        theme_index: dict,
        corpus_by_uid: dict,
        translations_by_uid: dict[str, str],
        bm25: BM25Retriever,
        dense: DenseRetriever | None = None,
    ):
        self.entities = entities
        self.themes = themes
        self.char_index = char_index
        self.group_index = group_index
        self.theme_index = theme_index
        self.corpus_by_uid = corpus_by_uid
        self.translations_by_uid = translations_by_uid
        self.bm25 = bm25
        self.dense = dense
        self.gazetteer = Gazetteer(entities, themes)

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
        dense_embeddings_path: Path | None = None,
        dense_uids_path: Path | None = None,
    ) -> "Retriever":
        with open(entities_path) as f:
            entities = json.load(f)
        with open(themes_path) as f:
            themes = json.load(f)
        with open(char_index_path) as f:
            char_index = json.load(f)
        with open(group_index_path) as f:
            group_index = json.load(f)
        with open(theme_index_path) as f:
            theme_index = json.load(f)

        corpus_by_uid = {
            rec["uid"]: rec
            for rec in stream_corpus(raw_path, verbose=False)
        }

        # Precompute fluent translations once. Cheaper than parsing
        # ai_analysis on every hydrate, and gives BM25 its index input.
        translations_by_uid = {
            uid: _extract_fluent_translation(rec)
            for uid, rec in corpus_by_uid.items()
        }
        bm25_uids = list(translations_by_uid.keys())
        bm25 = BM25Retriever(
            uids=bm25_uids,
            texts=[translations_by_uid[u] for u in bm25_uids],
        )

        # Dense index is optional — load it if both paths are given
        # and the artifacts exist on disk. Lets `mbh-query` and the
        # eval harness work without dense (falling back to BM25 for
        # concept) if the user hasn't run mbh-build-embeddings yet.
        dense: DenseRetriever | None = None
        if (
            dense_embeddings_path is not None
            and dense_uids_path is not None
            and dense_embeddings_path.exists()
            and dense_uids_path.exists()
        ):
            dense = DenseRetriever.from_paths(
                embeddings_path=dense_embeddings_path,
                uids_path=dense_uids_path,
            )

        return cls(
            entities=entities,
            themes=themes,
            char_index=char_index,
            group_index=group_index,
            theme_index=theme_index,
            corpus_by_uid=corpus_by_uid,
            translations_by_uid=translations_by_uid,
            bm25=bm25,
            dense=dense,
        )

    def search(self, query: str, *, limit: int = 20) -> RetrievalResponse:
        plan = classify(query, self.gazetteer)
        notes: list[str] = []
        scores_by_uid: dict[str, float] = {}

        if plan.mode == "structural_uid":
            uids = [plan.uid] if plan.uid in self.corpus_by_uid else []
            if not uids:
                notes.append(f"UID {plan.uid} not found in corpus.")

        elif plan.mode == "structural_slice":
            uids = self._slice(plan.book, plan.chapter)
            if not uids:
                where = (
                    f"B{plan.book}_C{plan.chapter}"
                    if plan.chapter
                    else f"B{plan.book}"
                )
                notes.append(f"No verses found for slice {where}.")

        elif plan.mode == "facet":
            uids = self._facet_intersection(plan, notes)
            # NOTE: BM25 reranking of facet results was attempted and
            # reverted. It hurt frozen facet items (mean MRR 0.65 → 0.30)
            # because pure-facet queries like "Karna and friendship"
            # have no residual content tokens beyond the facet keywords,
            # so BM25 reordered by TF noise and displaced hand-curated
            # known_goods.

        elif plan.mode == "lexical":  # BM25 over the quoted text
            text = plan.lexical_text or query
            uids, scores_by_uid = self._lexical(text, limit)
            if not uids:
                notes.append(
                    f"No lexical (BM25) hits for {text!r} — no query "
                    "token overlaps the corpus translations."
                )

        else:  # concept — dense bi-encoder (BM25 fallback if no index)
            uids, scores_by_uid = self._concept(query, limit, notes)

        results = [
            self._hydrate(uid, score=scores_by_uid.get(uid))
            for uid in uids[:limit]
        ]
        return RetrievalResponse(
            query=query,
            mode=plan.mode,
            plan=plan,
            results=results,
            total=len(uids),
            notes=notes,
        )

    # --- mode implementations ---

    def _lexical(
        self, text: str, limit: int
    ) -> tuple[list[str], dict[str, float]]:
        """BM25 ranking over the quoted text — the exact-keyword lane."""
        hits = self.bm25.search(text, top_k=limit)
        return [h.uid for h in hits], {h.uid: h.score for h in hits}

    def _concept(
        self, query: str, limit: int, notes: list[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Dense bi-encoder ranking — the paraphrase lane.

        Falls back to BM25 only if the dense index wasn't loaded, so the
        retriever still answers concept queries before
        `mbh-build-embeddings` has been run (with a note).
        """
        if self.dense is not None:
            hits = self.dense.search(query, top_k=limit)
            if not hits:
                notes.append(
                    f"Dense retrieval returned no hits for {query!r}."
                )
            return [h.uid for h in hits], {h.uid: h.score for h in hits}

        notes.append(
            "Dense index not loaded — concept fell back to BM25. "
            "Run `mbh-build-embeddings` to enable dense retrieval."
        )
        hits = self.bm25.search(query, top_k=limit)
        return [h.uid for h in hits], {h.uid: h.score for h in hits}

    def _slice(self, book: int, chapter: int | None) -> list[str]:
        """Return every UID in a book or a book+chapter, in corpus order."""
        if chapter is not None:
            prefix = f"B{book}_C{chapter}_S"
        else:
            prefix = f"B{book}_C"
        matching = [uid for uid in self.corpus_by_uid if uid.startswith(prefix)]
        return sorted(matching, key=_uid_sort_key)

    def _facet_intersection(
        self, plan: QueryPlan, notes: list[str]
    ) -> list[str]:
        sets: list[tuple[str, set[str]]] = []

        for canon in plan.characters:
            verses = set(self.char_index.get(canon, {}).get("verses", []))
            sets.append((f"character:{canon}", verses))

        for canon in plan.groups:
            verses = set(self.group_index.get(canon, {}).get("verses", []))
            # Expand to member characters (Layer 1 Choice 3).
            members = (
                self.entities.get("groups", {}).get(canon, {}).get("members", [])
            )
            for m in members:
                verses |= set(self.char_index.get(m, {}).get("verses", []))
            sets.append((f"group:{canon}", verses))

        for canon in plan.themes:
            verses = set(self.theme_index.get(canon, {}).get("verses", []))
            sets.append((f"theme:{canon}", verses))

        if not sets:
            return []

        intersection = set(sets[0][1])
        for _, s in sets[1:]:
            intersection &= s

        if not intersection and len(sets) > 1:
            notes.append(
                "Intersection of all matched facets is empty. "
                "Showing the union instead — verses that match ANY "
                "of the matched facets."
            )
            union: set[str] = set()
            for _, s in sets:
                union |= s
            intersection = union

        return sorted(intersection, key=_uid_sort_key)

    def _bm25_rerank(
        self, uids: list[str], query: str
    ) -> tuple[list[str], dict[str, float]]:
        """Reorder a uid list by descending BM25 score against the query.

        UIDs with score > 0 sort to the front (descending). UIDs with
        score 0 (no content-token overlap with the query) keep their
        input order at the tail. If the query has no content tokens
        after stopword filtering — pure-facet queries like
        "Karna and friendship" reduce to ["karna", "friendship"] which
        do score, but a single-character query like just "Arjuna"
        reduces to one token — we still call BM25; if all scores
        are zero we return the uids in their input order with empty
        scores.
        """
        tokens = bm25_tokenize(query)
        if not tokens:
            return list(uids), {}

        scores_arr = self.bm25.bm25.get_scores(tokens)
        uid_to_idx = {u: i for i, u in enumerate(self.bm25.uids)}

        scored: list[tuple[float, str]] = []
        unscored: list[str] = []
        scores_by_uid: dict[str, float] = {}
        for u in uids:
            idx = uid_to_idx.get(u)
            if idx is None:
                unscored.append(u)
                continue
            s = float(scores_arr[idx])
            if s > 0.0:
                scored.append((s, u))
                scores_by_uid[u] = s
            else:
                unscored.append(u)

        scored.sort(key=lambda x: -x[0])
        return [u for _, u in scored] + unscored, scores_by_uid

    # --- hydration ---

    def _hydrate(self, uid: str, *, score: float | None = None) -> Verse:
        rec = self.corpus_by_uid.get(uid, {})
        return Verse(
            uid=uid,
            book=rec.get("book", 0),
            chapter=rec.get("chapter", 0),
            shloka=rec.get("shloka", 0),
            sanskrit=rec.get("sanskrit", ""),
            translation=self.translations_by_uid.get(uid, ""),
            score=score,
        )


def _extract_fluent_translation(rec: dict) -> str:
    """Pull the 'Fluent Translation' section out of ai_analysis.

    Falls back to an empty string if the section is missing (which
    shouldn't happen for clean records — the Layer 1 audit confirmed
    every verse has all five sections)."""
    ai_text = rec.get("ai_analysis", "")
    if not ai_text:
        return ""
    sections = parse_sections(ai_text)
    return sections.get("Fluent Translation") or ""


def _uid_sort_key(uid: str) -> tuple[int, int, int, str]:
    """Numeric-aware sort so B10 comes after B9, not after B1."""
    m = _UID_PREFIX_RE.match(uid)
    if not m:
        return (0, 0, 0, uid)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), uid)
