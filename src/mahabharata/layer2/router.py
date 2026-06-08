"""Layer 2 query router — Phase A + C entry point.

Classifies a free-form query into one of five retrieval modes:

    - structural_uid     exact verse ID like "B6_C27_S29"
    - structural_slice   book or chapter slice like "B6" or "B6_C27"
    - lexical            a double-quoted query like `"iron mace"` —
                         explicit exact-keyword intent, handled by BM25
    - facet              one or more canonical entities/themes mentioned
                         in the query — resolves to an index intersection
    - concept            free-text / paraphrase query — handled by the
                         dense bi-encoder (Phase C / Step 6)

Why lexical is quote-gated (2026-06-08)
---------------------------------------
A 7-item lexical eval and the 15-item concept eval showed that once
question-form and high-coverage-facet queries are removed, the residue
— bare short phrases like `iron mace` (lexical) vs `seeking revenge`
(concept) — is syntactically identical. The router cannot infer
exact-keyword vs concept intent from a bare phrase without a model, and
a token-count heuristic overfits a handful of examples. So lexical
intent is an *explicit* signal: wrap the query in double quotes. This
matches the architecture doc's own lexical example (`"chariot"`).
Empirically BM25 beats dense ~4x on lexical queries and dense beats
BM25 ~4x on concept queries, so we route to the single right lane
rather than fusing (RRF was tested and regressed both shapes).

The router is deliberately a hand-written rule system, not an LLM call.
See theoretical_concepts_and_architecture.md, Layer 2 Choice 1.

Design notes
------------
Rule ordering matters. Structural patterns are checked first because
they are unambiguous — anything matching `B\\d+_C\\d+_S\\d+` is a UID and
cannot legitimately be anything else. After that, two gates decide
between facet and concept:

  1. Question shape — queries ending in `?` or starting with a wh-word
     (why/how/what/when/where/who/whom/whose/which) route to concept
     regardless of any gazetteer hit. Filter-style facet queries never
     take question form; semantic questions that name a character
     always do, and they need semantic ranking, not an entity dump.
  2. Facet coverage ratio — for non-question queries with gazetteer
     hits, the matched tokens must cover at least 50% of the query's
     post-stopword content tokens to qualify as facet. Below that the
     names are incidental name-drops and the query routes to concept.

Both gates were added 2026-05-14 to fix a router-misroutes-to-facet
failure on concept eval items (12/15 concept queries were stolen by
the gazetteer before dense could rank them).

Longest-alias-first matching in the gazetteer prevents a short alias
(e.g. "Nara") from shadowing a longer one that contains it
(e.g. "Narayana"). After each successful match we also strike the
matched span from the query so the same run of text cannot double-tag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from mahabharata.layer2.bm25 import tokenize as _content_tokenize

Mode = Literal[
    "structural_uid",
    "structural_slice",
    "lexical",
    "facet",
    "concept",
]

# A double-quoted span signals explicit exact-keyword (lexical) intent.
# Straight or curly double quotes are accepted; curly quotes are
# normalized to straight before matching so Mac smart-quoting works.
_QUOTED_RE = re.compile(r'"([^"]+)"')
_CURLY_QUOTES = str.maketrans({"“": '"', "”": '"'})

# Wh-words that, when leading a query, signal semantic intent and route
# to the concept lane regardless of any incidental gazetteer hit.
_WH_WORDS = frozenset({
    "why", "how", "what", "when", "where", "who", "whom", "whose", "which",
})
_LEADING_WORD_RE = re.compile(r"\W*(\w+)")

# Fraction of content tokens that must be consumed by gazetteer matches
# for a non-question query to count as facet-style. Below this, the
# gazetteer hits are incidental name-drops and the query is routed to
# concept so dense / BM25 can do semantic ranking instead of a flat
# entity dump. 0.5 was picked against the 45-item Layer-2 eval set —
# every filter-style facet item has coverage 1.0, every concept item
# that escapes the question-shape check has coverage ≤ 0.33.
_FACET_COVERAGE_THRESHOLD = 0.5


@dataclass
class QueryPlan:
    """Structured description of what the retriever should execute."""
    mode: Mode
    # structural fields
    uid: str | None = None
    book: int | None = None
    chapter: int | None = None
    # lexical field — the unquoted text to hand to BM25
    lexical_text: str | None = None
    # facet fields — canonical names, already resolved from aliases
    characters: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    # debug / explanation — the alias strings that triggered each match
    matched_terms: list[str] = field(default_factory=list)


UID_RE = re.compile(r"^B(\d+)_C(\d+)_S(\d+)(_orphan)?$")
CHAPTER_SLICE_RE = re.compile(r"^B(\d+)_C(\d+)$")
BOOK_SLICE_RE = re.compile(r"^B(\d+)$")


class Gazetteer:
    """Alias -> canonical lookup built from entities.json + themes.json.

    Each entry is (lowercased_alias, canonical, kind) where kind is one
    of 'character', 'group', 'theme'. The canonical name itself is
    always included as an alias so queries that use canonical spelling
    still match.
    """

    def __init__(self, entities: dict, themes: dict):
        entries: list[tuple[str, str, str]] = []

        for canon, info in entities.get("characters", {}).items():
            names = set(info.get("aliases", [])) | {canon}
            for n in names:
                entries.append((n.lower(), canon, "character"))

        for canon, info in entities.get("groups", {}).items():
            names = set(info.get("aliases", [])) | {canon}
            for n in names:
                entries.append((n.lower(), canon, "group"))

        for canon, info in themes.get("themes", {}).items():
            variants = set(info.get("variants", [])) | {canon}
            for v in variants:
                entries.append((v.lower(), canon, "theme"))

        # Dedup identical (alias, canonical, kind) triples, then sort by
        # descending alias length so longer phrases win the race.
        self.entries = sorted(set(entries), key=lambda e: -len(e[0]))

    def match(
        self, query: str
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Return (characters, groups, themes, matched_terms).

        The query is normalized by lowercasing, replacing punctuation
        with spaces, collapsing whitespace, and padding with leading
        and trailing spaces. Aliases are then tested via simple
        substring containment with space-delimited boundaries, which
        gives us word-boundary matching cheaply and correctly for
        multi-word aliases.
        """
        q = " " + re.sub(r"[^\w\s]", " ", query.lower()) + " "
        q = re.sub(r"\s+", " ", q)

        chars: list[str] = []
        groups: list[str] = []
        themes: list[str] = []
        matched_terms: list[str] = []
        seen = {"character": set(), "group": set(), "theme": set()}

        for alias, canon, kind in self.entries:
            needle = f" {alias} "
            if needle not in q:
                continue
            # Strike the match to prevent double-matching on the same span.
            q = q.replace(needle, " ")
            if canon in seen[kind]:
                continue
            seen[kind].add(canon)
            matched_terms.append(alias)
            if kind == "character":
                chars.append(canon)
            elif kind == "group":
                groups.append(canon)
            else:
                themes.append(canon)

        return chars, groups, themes, matched_terms


def classify(query: str, gazetteer: Gazetteer) -> QueryPlan:
    """Route a query to a retrieval mode.

    Precedence: exact UID > chapter slice > book slice > quoted lexical >
    question-shape concept > high-coverage facet > concept fall-through.
    The quoted-lexical gate was added 2026-06-08; question-shape and
    coverage gates 2026-05-14 — see module docstring.
    """
    q = query.strip()

    if UID_RE.match(q):
        return QueryPlan(mode="structural_uid", uid=q)

    if m := CHAPTER_SLICE_RE.match(q):
        return QueryPlan(
            mode="structural_slice",
            book=int(m.group(1)),
            chapter=int(m.group(2)),
        )

    if m := BOOK_SLICE_RE.match(q):
        return QueryPlan(mode="structural_slice", book=int(m.group(1)))

    # Quoted span → explicit lexical intent (precedence over facet, so a
    # quoted `"iron mace"` reaches BM25 instead of being grabbed by the
    # "mace" theme facet). Join all quoted spans as the BM25 query text.
    quoted = _QUOTED_RE.findall(q.translate(_CURLY_QUOTES))
    if quoted:
        return QueryPlan(mode="lexical", lexical_text=" ".join(quoted).strip())

    chars, groups, themes, terms = gazetteer.match(q)
    has_gazetteer_hits = bool(chars or groups or themes)

    # Question-shape override: even with gazetteer hits, route a
    # question-form query to concept. Filter-style facet queries never
    # take this form ("Karna and friendship"); semantic questions that
    # name a character ("Why did Arjuna refuse to fight?") always do.
    # The gazetteer fields are still attached to the plan for telemetry
    # and any future RRF-style fusion that wants the entity prior.
    if _is_question_shape(q):
        return QueryPlan(
            mode="concept",
            characters=chars,
            groups=groups,
            themes=themes,
            matched_terms=terms,
        )

    if not has_gazetteer_hits:
        return QueryPlan(mode="concept")

    if _facet_coverage(q, terms) >= _FACET_COVERAGE_THRESHOLD:
        return QueryPlan(
            mode="facet",
            characters=chars,
            groups=groups,
            themes=themes,
            matched_terms=terms,
        )

    # Gazetteer hits exist but they only cover a small fraction of the
    # query — the names are incidental to a semantic intent. Drop to
    # concept and let dense / BM25 rank by meaning, keeping the matches
    # on the plan for downstream consumers.
    return QueryPlan(
        mode="concept",
        characters=chars,
        groups=groups,
        themes=themes,
        matched_terms=terms,
    )


def _is_question_shape(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    if q.endswith("?"):
        return True
    m = _LEADING_WORD_RE.match(q)
    return bool(m) and m.group(1) in _WH_WORDS


def _facet_coverage(query: str, matched_terms: list[str]) -> float:
    """Fraction of post-stopword content tokens consumed by gazetteer hits.

    Uses the same tokenizer the BM25 index uses so "content tokens" is
    a single consistent notion across the system. Coverage 1.0 = every
    content token in the query was part of a gazetteer match.
    """
    content_tokens = _content_tokenize(query)
    if not content_tokens:
        return 0.0
    matched: list[str] = []
    for term in matched_terms:
        matched.extend(_content_tokenize(term))
    return len(matched) / len(content_tokens)
