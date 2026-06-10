# MahabharataNew — Project Context

> **Companion doc:** `theoretical_concepts_and_architecture.md` — the *why*
> behind design choices, per layer. This file covers the *what*.

## Repo Layout (post 2026-04-15 reorg)

Installable Python package with a `src/` layout. Library logic lives under
`src/mahabharata/`; each build step has a pure-Python library module in
`layer1/` and a thin CLI wrapper in `cli/`. Paths are centralized in
`common/paths.py` so reorganizing data directories later is a single-file
change.

```
MahabharataNew/
├── pyproject.toml                     # editable install target
├── .venv/                              # project-local virtualenv (Python 3.12)
├── src/mahabharata/
│   ├── common/
│   │   ├── paths.py                    # canonical path constants
│   │   ├── corpus_loader.py            # stream_corpus(), UID collision patches
│   │   ├── parvas.py                   # book→parva-name map (single source)
│   │   └── sections.py                 # shared ai_analysis section parser
│   ├── layer1/
│   │   ├── entities.py                 # Step 1 library
│   │   ├── themes.py                   # Step 2 library
│   │   ├── verse_characters.py         # Step 3 library
│   │   ├── verse_themes.py             # Step 4 library
│   │   ├── indexes.py                  # Step 5 library
│   │   └── summaries.py                # Step 6 library (chapter/parva rollup)
│   ├── layer2/
│   │   ├── router.py                   # rule-based 5-mode query classifier
│   │   ├── retriever.py                # UID / slice / lexical / facet / concept
│   │   ├── bm25.py                     # Phase C Step 5 — BM25 lexical lane
│   │   ├── embed.py                    # Phase C Step 6 — offline embedding builder
│   │   ├── dense.py                    # Phase C Step 6 — dense bi-encoder lane
│   │   ├── fusion.py                   # Phase C Step 7 — Reciprocal Rank Fusion
│   │   ├── eval_harness.py             # eval-set runner + report writer
│   │   └── eval_bootstrap.py           # Ollama-driven eval-set draft generator
│   └── cli/
│       ├── build_entities.py           # `mbh-build-entities`
│       ├── build_themes.py             # `mbh-build-themes`
│       ├── build_verse_characters.py   # `mbh-build-verse-characters`
│       ├── build_verse_themes.py       # `mbh-build-verse-themes`
│       ├── build_indexes.py            # `mbh-build-indexes`
│       ├── build_summaries.py          # `mbh-build-summaries`
│       ├── build_embeddings.py         # `mbh-build-embeddings`
│       ├── query.py                    # `mbh-query`
│       ├── eval.py                     # `mbh-eval`
│       └── bootstrap_eval.py           # `mbh-bootstrap-eval`
├── scripts/
│   └── audit_ai_analysis.py            # one-off data-quality check
├── docs/
│   ├── project_context.md              # this file
│   └── theoretical_concepts_and_architecture.md
├── tests/                              # empty stub — to be populated
└── data/
    ├── raw/search_engine_db.jsonl      # single source of truth
    ├── layer1/
    │   ├── entities.json / themes.json
    │   ├── entities_overrides.json / themes_overrides.json
    │   ├── verse_characters.jsonl / verse_themes.jsonl
    │   ├── character_index.json / group_index.json / theme_index.json
    │   ├── chapter_summaries.jsonl     # Step 6 — 1,995 chapter digests
    │   ├── parva_summaries.json        # Step 6 — 18 parva aggregates
    │   └── reports/                    # all *_coverage_report.md files
    └── layer2/
        ├── dense/
        │   ├── embeddings.npy          # (73820, 384) L2-normalized bge vectors
        │   └── uids.txt                # parallel UID list, one per matrix row
        ├── eval/
        │   ├── eval_set.jsonl          # 52 hand-curated atomic items (5 shapes)
        │   ├── eval_set_draft.jsonl    # llama3 bootstrap output (provenance)
        │   ├── lexical_eval_staging.jsonl  # 7 lexical items (pre-fold provenance)
        │   └── concept_curation_review.md  # dense-hit triage for concept KGs
        └── reports/
            ├── eval_bootstrap_report.md
            ├── eval_results.md         # latest run, human-readable
            └── eval_results_{latest,previous,baseline}.json  # 3-slot rotation
```

**CLI pattern:** every build step is exposed as a `mbh-build-*` console
script (defined in `pyproject.toml`). Library functions take explicit path
args; CLI wrappers pass the defaults from `common.paths`. This means the
library halves are importable from notebooks / tests without touching the
CLI layer.

**Why the split:** Pre-reorg, each `build_*.py` was a single script mixing
path constants, library logic, argparse-free `main()`, and report writing.
The src/ layout was chosen up front (2026-04-15) for testability, notebook
importability, and because doing it later when there are many more files
would be more expensive than doing it now.

## What This Project Is

A lean, locally-running Mahabharata research tool built from scratch with better
architecture than the previous project (Mahabharat_Python_Test). The goal is a
**personal research + AI engineering learning** project — not a public product.

## Why We're Starting Over

The previous project (Mahabharat_Python_Test) had fundamental architectural issues:

- Fine-tuned Llama 3 8B on Gemini's *output format* → format imitation, not capability
- Double-translation degradation: Sanskrit → Gemini analysis → LLM re-synthesis
- Only indexed 4,359 of 73,820 verses (6% of the corpus) in the search engine
- Eval-heavy training runs (17 min per eval pass) wasted compute
- GGUF export pipeline was fragile (bitsandbytes metadata, disk space issues)
- Large model (8B) for tasks that don't need it

## The New Philosophy

The Mahabharata is a **finite, closed corpus** — ~73,820 verses across 18 books.
We can pre-build structured knowledge once (offline, using Gemini cheaply) and
make it queryable at runtime with lean models. We do NOT need a large model to
"know things" — we need smart retrieval + structured data + a small synthesis model.

```
Old:  Query → retrieve verse snippets → big model synthesises
New:  Query → route to pre-built knowledge → small model (or no model) assembles
```

## The Dataset

**Source file:** `data/raw/search_engine_db.jsonl`

- 73,820 verses, all 18 books of the Mahabharata
- Each record has:
  - `uid`: verse ID in format B{book}_C{chapter}_S{shloka} (e.g. B1_C1_S1)
  - `book`, `chapter`, `shloka`: integers
  - `sanskrit`: original Sanskrit text (Devanagari)
  - `ai_analysis`: Gemini 2.0 Flash annotation — includes fluent translation,
    poetic translation, summary, keywords, word dictionary
  - `ai_raw_json`: raw Gemini response

This is the **single source of truth**. Everything else gets derived from it.

## Target Architecture

### Layer 1 — Structured Knowledge (pre-built offline, one-time)
- **Character index**: all verses per named character, relationships, arc summary
- **Concept/theme tags**: dharma, karma, war, devotion, etc. per verse
- **Chapter/parva summaries**: aggregate summaries above verse level (~1,995 chapters)
- **Entity/event graph**: who did what to whom, in which book/chapter

### Layer 2 — Retrieval (lean, local, fast)
- BM25 keyword search over full 73,820 verses
- Semantic search via a small embedding model (e.g. paraphrase-multilingual-mpnet)
- Structural lookup: direct UID resolution for positional queries
- Cross-encoder reranker (~80MB, CPU-friendly) as a second-pass filter

### Layer 3 — Synthesis (lean model, ~1-3B params)
- Only invoked when retrieval alone can't assemble the answer
- Candidates: Phi-3 Mini (3.8B), Qwen2.5 1.5B, or a custom fine-tuned seq2seq
- Fine-tuned on *harder tasks*: character arc queries, multi-verse reasoning,
  comparative questions — NOT verse-level format imitation

### Layer 4 — Translation Model (optional, future)
- A lean fine-tuned seq2seq (mBART or NLLB) for Sanskrit→English
- Training data: the 73,820 sanskrit+translation pairs already in the dataset
- Replaces Gemini dependency for translation at inference time

## What the User Wants to Learn

This project is also an AI engineering education exercise. Concepts to cover:
- Knowledge extraction & structured data building
- Evaluation frameworks (can we measure answer quality?)
- Reranking (cross-encoders vs bi-encoders)
- Knowledge graphs / triple extraction
- Agentic routing patterns
- Fine-tuning seq2seq models for specific tasks

## Build Order (proposed)

1. ~~**Data audit**~~ ✓ **DONE** — see findings below
2. **Knowledge extraction** — parse `ai_analysis` into structured fields (characters,
   themes, keywords) across all verses
   - ✓ Step 1: canonical entity list (`mahabharata.layer1.entities` → `data/layer1/entities.json`)
   - ✓ Step 2: theme taxonomy (`mahabharata.layer1.themes` → `data/layer1/themes.json`)
   - ✓ Step 3: per-verse character tagging (`mahabharata.layer1.verse_characters` → `data/layer1/verse_characters.jsonl`)
   - ✓ Step 4: per-verse theme tagging (`mahabharata.layer1.verse_themes` → `data/layer1/verse_themes.jsonl`)
   - ✓ Step 5: inverted indexes (`mahabharata.layer1.indexes` → `data/layer1/*_index.json`) — UID collisions resolved via in-memory patch in `mahabharata.common.corpus_loader`
   - ✓ Step 6: chapter/parva summaries — naive rollup, no LLM (`mahabharata.layer1.summaries` → `data/layer1/chapter_summaries.jsonl` + `parva_summaries.json`)
   - ☐ Step 8: validation
3. **Layer 2 retrieval — Phase A (structural + facet)**
   - ✓ Router: rule-based classifier into structural_uid / structural_slice / lexical / facet / concept (`mahabharata.layer2.router`)
   - ✓ Retriever: in-memory UID dict + facet index intersection with union fallback (`mahabharata.layer2.retriever` → `mbh-query`)
   - ✓ Eval-set bootstrap (Ollama) + hand-curated final set (`mahabharata.layer2.eval_bootstrap` → `mbh-bootstrap-eval`; `data/layer2/eval/eval_set.jsonl`)
   - ✓ Eval harness: runs the eval set, reports recall@k/MRR/router accuracy, 3-slot JSON rotation (`mahabharata.layer2.eval_harness` → `mbh-eval`)
   - ✓ Frozen regression-anchor subset (10 items, `frozen: true` in `eval_set.jsonl`)
   - ✓ Router fix: `UID_RE` accepts `_orphan`-suffixed UIDs (done)
   - ✓ Router fix: question-shape + facet-coverage gates so concept queries aren't stolen by the gazetteer (2026-05-14)
4. **Layer 2 retrieval — Phase C (free-text)** — over all 73,820 verses (vs 4,359 before)
   - ✓ Step 5: BM25 lexical lane (`mahabharata.layer2.bm25`)
   - ✓ Step 6: Dense bi-encoder lane, `bge-small-en-v1.5`, offline embeddings (`mahabharata.layer2.{embed,dense}` → `mbh-build-embeddings`)
   - ✓ Step 7: RRF fusion (`mahabharata.layer2.fusion`) **built, tested, REJECTED** — equal-weight RRF regressed *both* query shapes. Replaced by **route-by-shape**: quoted queries → BM25 (lexical), paraphrase → dense (concept). `fusion.py` kept for a future "mixed" route (2026-06-08)
   - ☐ Step 8+: cross-encoder reranker (gated on eval evidence), summary-field index, event graph — all deferred
5. **Small synthesis model (Layer 3)** — **active next step** (decided
   2026-06-10). Retrieval is solid (eval 49/52); the 3 concept canaries
   were found to be reasoning/scene questions best handled by synthesis,
   not verse-level retrieval (see the 2026-06-10 section below).
6. **Translation model** — fine-tune NLLB/mBART on the Sanskrit↔English pairs

## Environment

- Mac (Apple Silicon), local development
- Python 3.12 (Homebrew: `/opt/homebrew/bin/python3.12`)
- Project-local virtualenv at `.venv/` — created 2026-04-15 during the src/
  reorg. Homebrew Python is PEP 668 externally-managed, so venv is required
- Setup: `python3.12 -m venv .venv && .venv/bin/pip install -e .`
- After install, build steps run as console scripts: `mbh-build-entities`,
  `mbh-build-themes`, etc. (via `.venv/bin/mbh-build-*`)
- Ollama available locally for model serving
- Gemini API available for one-time offline enrichment tasks (not inference)
- No RunPod needed until/unless we fine-tune a model

## Layer 1 / Step 1 — Entity Bootstrap (2026-04-12)

Library: `src/mahabharata/layer1/entities.py` (CLI: `mbh-build-entities`)
→ `data/layer1/entities.json`

### Approach

Hand-curated seed list of major Mahabharata characters (80), groups (4), and
places (24), each with their well-known epithets/aliases. Bootstrap pattern
chosen over LLM extraction because (a) the corpus is closed and the major
figures are few and famous, (b) a quick frequency cross-check against
keyword counts validates the seed without manual review.

**Override pattern is in place from day one.** Each build script reads its
matching `*_overrides.json` file (if present) and applies add/remove/modify/
merge operations on top of the seed output. Re-running the build is always
safe — the override file is never overwritten. A stub
`entities_overrides.json` with schema docs is auto-created on first run.

### Entity counts
- 80 characters (Pandavas, Kauravas, sages, deities, narrators, etc.)
- 4 groups (Pandavas, Kauravas, Brahmins, Vasus) for collective references
- 24 places (cities, kingdoms, forests, battlefield, river)

### Seed quality (from coverage report)

Top 10 characters by total mentions (canonical + aliases) in the corpus:
| Canonical | Mentions |
|---|---:|
| Arjuna | 6,356 |
| Krishna | 4,431 |
| Bhima | 3,133 |
| Yudhishthira | 3,024 |
| Karna | 2,778 |
| Indra | 2,667 |
| Bhishma | 2,285 |
| Duryodhana | 2,162 |
| Drona | 2,075 |
| Dhritarashtra | 1,703 |

Group mentions: Pandavas (3,411), Brahmins (2,806), Kauravas (636), Vasus (128).

### Bootstrap bugs found and fixed during validation

- `Drishtadyumna` → renamed to `Dhrishtadyumna` (corpus uses the latter
  spelling); added old form as alias. Was 0 hits → now 581.
- `Brahma`: removed `Brahman` alias (philosophical concept, not the deity)
  and `Pitamaha` (overlaps with Bhishma's far more common usage).
- `Balarama`: removed `Rama` alias (ambiguous — over-claimed Parashurama
  and Ramayana references).
- `Drona`: removed `Acharya` (too generic).
- `Karna`: removed `Vrisha` (ambiguous, literally "bull").
- `Nakula`/`Sahadeva`: removed shared `Madreya` alias (only 2 corpus hits).

### Override schema (for future user corrections)

```json
{
  "characters": {
    "add":    {"NewChar": {"aliases": [...], "type": "..."}},
    "remove": ["WrongChar"],
    "modify": {"Arjuna": {"add_aliases": [...], "remove_aliases": [...]}},
    "merge":  [{"into": "Bhima", "absorb": ["AltCanon"]}]
  },
  "groups": {...},
  "places": {...}
}
```

---

## Layer 1 / Step 2 — Theme Taxonomy (2026-04-13)

Library: `src/mahabharata/layer1/themes.py` (CLI: `mbh-build-themes`)
→ `data/layer1/themes.json`

### Approach

Hand-curated controlled vocabulary of themes organised into 6 families
(ethical, emotional, martial, spiritual, social, cosmological). Same
bootstrap-and-override pattern as Step 1. Families are organisational
only — queries hit themes directly, not families. See
`theoretical_concepts_and_architecture.md` for the full rationale
(faceted retrieval, controlled vocabulary over topic modeling,
residual analysis as closed-loop validation).

### Taxonomy stats (v2, post-iteration)

- **94 themes across 6 families**, 0 variant conflicts
- Residual analysis excludes both theme variants *and* all entity terms
  from `entities.json`, so surfaced candidates are genuinely uncovered

### Family totals

| Family | Total mentions | Theme count |
|---|---:|---:|
| martial | 42,727 | 17 |
| social | 26,358 | 16 |
| spiritual | 21,833 | 26 |
| cosmological | 20,009 | 13 |
| ethical | 12,605 | 12 |
| emotional | 12,312 | 10 |

Skew matches corpus reality — it's a war epic. Family totals are a
balance-check signal, not a retrieval signal.

### What iterating on the seed caught

v1 → v2 fixed three classes of seed bugs surfaced by the coverage report:

1. **Case-sensitivity bugs** — Gemini keywords include capitalised
   concept nouns. `Dharma` (419 hits), `Rakshasa`/`Rakshasas` (788
   combined), `Gandharvas` (406) were all missed by lowercase-only
   variants. Added the capitalised forms.
2. **Missing themes** — residual analysis flagged 22 legitimate theme
   candidates the seed didn't cover. Added: merit, greed, forgiveness,
   non_violence, vows, refuge, hermitage, delusion, detachment,
   purity, charity, ignorance, rebirth, humility, teachers,
   friendship, women, fame, assembly, armor, creation, plus artha
   folded into wealth.
3. **Zero-hit variants** — typo-or-wrong-spelling check dropped
   `celestials`, `honour`, `renouncer` (v1) and flagged `armour`,
   `Sudra` (v2, trivial fixes via next iteration or overrides).

### Key v1 → v2 deltas

| Theme | v1 | v2 | Reason |
|---|---:|---:|---|
| dharma | 4,001 | 4,420 | added `Dharma` cap form |
| demons | 660 | 1,448 | added `Rakshasa`/`Rakshasas` |
| divine_beings | 2,484 | 3,012 | added `Gandharvas`/`Gandharva` |
| asceticism | 1,959 | 2,360 | added `austerity` singular |

Spiritual family total went from 16,464 → 21,833 after adding ~10
new themes, rebalancing it against martial/social.

### Residual candidates left for future iteration

- **Still ambiguous** (deliberately deferred, not an error): `strength`,
  `power`, `world`, `life`, `body` — need a decision on how narrowly to
  classify generic-ish words
- **Entity candidates** (should route to `entities_overrides.json`,
  not themes): `Rama` (668), `Bharata` (645), `Panchalas` (455),
  `Nala`/`Damayanti` (321/309 — sub-story characters), `Brihaspati`
  (248), `Garuda` (219)

### Override schema (for future user corrections)

```json
{
  "themes": {
    "add":    {"new_theme": {"family": "...", "variants": [...]}},
    "remove": ["wrong_theme"],
    "modify": {"dharma": {"add_variants": [...],
                          "remove_variants": [...],
                          "set_family": "..."}},
    "merge":  [{"into": "dharma", "absorb": ["duplicate_theme"]}]
  }
}
```

---

## Layer 1 / Step 3 — Per-verse character tagging (2026-04-13)

Library: `src/mahabharata/layer1/verse_characters.py` (CLI:
`mbh-build-verse-characters`) → `data/layer1/verse_characters.jsonl`
+ `data/layer1/reports/verse_characters_coverage_report.md`

### Approach

Dictionary-based NER over the closed corpus: each verse's Keywords
section is parsed, tokens are set-intersected against alias maps built
from `entities.json` (characters and groups kept as separate maps),
and the canonical resolution is written as one JSONL record per verse.
Empty records are preserved — "no entity" is informative. Alias maps
use collision detection: any term that resolves to two different
canonicals (within or across maps) crashes the build loudly. See
`theoretical_concepts_and_architecture.md` (Step 3 in detail) for the
rationale — short version: ambiguity was resolved upstream in Step 1,
so Step 3's job is mechanical lookup + deduplication.

Groups are tagged *atomically* — "Pandavas" stays "Pandavas", not
expanded into five Pandava members. Member expansion is a query-time
decision; baking it in at Step 3 would destroy signal (a verse that
literally says "Pandavas" is different from a verse naming Yudhishthira
specifically).

### Coverage stats

- 73,820 verses streamed in a single pass
- **47.1%** of verses carry ≥1 entity tag (char or group)
- 42.2% have ≥1 character tag; 9.1% have ≥1 group tag; 4.3% have both
- 52.9% have zero entity tags — expected, heavily concentrated in the
  philosophical parvas (Shanti 16.4%, Anushasana 22.6%); narrative
  parvas sit at 50–68%
- Top by verses tagged: Arjuna 5,931 · Krishna 3,985 · Bhima 3,065 ·
  Yudhishthira 2,911 · Karna 2,716 · Indra 2,539 · Bhishma 2,248

### Validation signal

The coverage report includes a `ratio = verses_tagged / step1_mentions`
column per canonical. All characters fall in the expected 0.90–1.00
band — values below 1.00 mean a verse double-mentions a character via
multiple aliases (correct dedup behavior); ~1.00 means single-alias
characters. No outliers, so the alias maps are clean end-to-end and
Step 1 → Step 3 parity holds.

### Spot-check

- `B1_C1_S0` → `{characters: [Vishnu, Saraswati], groups: []}` — the
  opening invocation, correct.
- `B1_C1_S1` → `{characters: [Ugrashrava, Shaunaka], groups: []}` —
  the Sauti narrator framing, correct.

---

## Layer 1 / Step 4 — Per-verse theme tagging (2026-04-14)

Library: `src/mahabharata/layer1/verse_themes.py` (CLI:
`mbh-build-verse-themes`) → `data/layer1/verse_themes.jsonl`
+ `data/layer1/reports/verse_themes_coverage_report.md`

### Approach

Structural mirror of Step 3, but against `themes.json` instead of
`entities.json`. Each verse's Keywords section is set-intersected
against a single variant→canonical map (there is only one namespace
for themes — no characters/groups split). Collision detection runs at
build time: any variant claimed by two themes is a taxonomy bug and
crashes the script. Empty records are preserved.

### Coverage stats

- 73,820 verses streamed in a single pass
- **85.2%** of verses carry ≥1 theme tag (vs 47.1% for entities)
- Themes are the denser facet: most verses touch 1–3 themes, median 2
- Only 8 verses at 8+ themes — no over-matching variant

### Validation signal

- All 94 per-theme ratios (`verses_tagged / step2_mentions`) fall in
  the expected **0.86–1.00** band. No zero rows, no outliers — the
  variant map is clean, and Step 2 → Step 4 parity holds end-to-end.
- Family balance mirrors Step 2's mention totals (martial > social >
  cosmological ≈ spiritual > emotional > ethical) — no taxonomy drift
  introduced by deduplication.

### Key finding: entity vs theme facets are genuinely complementary

The by-book contrast is the payoff. Philosophical parvas jump
dramatically from entity coverage to theme coverage:

| Parva | Entity cov | Theme cov |
|---|---:|---:|
| Shanti | 16.4% | **85.6%** |
| Anushasana | 22.6% | **83.2%** |
| Drona | 67.8% | 92.2% |
| Karna | 66.9% | 91.8% |

Narrative parvas (Drona, Karna) already had strong entity coverage and
gain ~25pp from themes. But Shanti/Anushasana — where entity coverage
was almost useless — are now retrievable via the theme facet at 85%+.
This confirms the Step 2 hypothesis: philosophical verses are where
the theme facet does load-bearing work at retrieval time, and the two
facets should be indexed as orthogonal signals.

---

## Layer 1 / Step 5 — Inverted indexes (2026-04-14)

Library: `src/mahabharata/layer1/indexes.py` (CLI: `mbh-build-indexes`)
→ `data/layer1/character_index.json`, `group_index.json`,
`theme_index.json`, `reports/indexes_coverage_report.md`

### Approach

Inverts `verse_characters.jsonl` and `verse_themes.jsonl` into three
canonical→verses lookup tables — the first Layer 1 artifact consumed
directly by Layer 2 (retrieval). Schema per entry:

```json
"<canonical>": {
  "type"|"family": "...", "count": <int>, "verses": ["B1_C1_S1", ...]
}
```

Canonicals are ordered by descending count (self-sorted for top-k
browsing); verses within each entry are in corpus order.

Validation is layered: (a) no orphan canonicals — every tag in
Step 3/4 output must exist in `entities.json`/`themes.json`,
(b) no dangling UIDs — every listed UID must exist in the raw
corpus, (c) counts must match Step 3/4 coverage reports exactly.

### Coverage

All 80 characters, 5 groups (4 groups + 1 tribe fold-in), and 94
themes are present in the indexes. No zero-coverage canonicals.
Counts reproduce Step 3/4 reports exactly — inversion is faithful.

### UID collisions investigation and fix (2026-04-14)

**Root cause identified.** The raw corpus had 73,820 lines but only
73,816 unique UIDs. Investigation traced this to **orphaned shloka
fragments** — stray records where the scraper's shloka-number parser
defaulted to a bogus value (usually S1, once S5) because the source
line lacked a clean shloka marker. The 4 strays:

| Stray UID | True content | Why orphaned |
|---|---|---|
| B3_C282_S1 (line 18937) | Markandeya narration about Dyumatsena regaining sight | Mid-chapter star-passage between S21 and S22 |
| B6_C27_S5 (line 27852) | Second half of **Bhagavad Gita 5.29** (`suhṛdaṃ sarva-bhūtānāṃ...`) | The first half lives correctly at B6_C27_S29 on the immediately prior line — a full shloka that got split across two records, with the second half mis-numbered |
| B13_C33_S1 (line 64238) | "Honor the Brahmins" (Bhishma teaching) | Trailing chapter-end star-passage |
| B13_C74_S1 (line 65777) | Yudhishthira asking about rules for cows | Trailing chapter-end star-passage |

**Broader finding** (not a bug): **77% of chapters (1,531/1,995)
are interleaved** in file order, and 61% have non-monotonic shloka
sequences within them. This is a scraper pagination artifact — the
source was fetched in chunks that got concatenated without
re-sorting. Harmless for UID-keyed retrieval, just makes the raw
file ugly to browse.

### Fix: in-memory UID patching via `mahabharata.common.corpus_loader`

Rather than rewriting the raw corpus, the module
`src/mahabharata/common/corpus_loader.py` holds a 4-entry
`UID_COLLISION_PATCH` table and exposes `stream_corpus()` — a generator
that parses the raw JSONL and re-labels the 4 strays with synthetic UIDs
at read time:

- `B3_C282_S1` → `B3_C282_S1_orphan`
- `B6_C27_S5` → `B6_C27_S5_orphan`
- `B13_C33_S1` → `B13_C33_S1_orphan`
- `B13_C74_S1` → `B13_C74_S1_orphan`

Patch matching is `(uid, sanskrit_prefix)`-based rather than
line-number-based, so it remains stable under raw-file reordering.
The loader warns if any patch fails to fire or fires multiple times
(drift guard).

Library modules `layer1/verse_characters.py`, `layer1/verse_themes.py`,
and `layer1/indexes.py` import `stream_corpus` from
`mahabharata.common.corpus_loader` instead of opening the raw file
directly. `layer1/entities.py`, `layer1/themes.py`, and
`scripts/audit_ai_analysis.py` don't consume the UID field so the patch
is irrelevant to them.

### Post-fix validation

After re-running Steps 3–5:

- Both `verse_characters.jsonl` and `verse_themes.jsonl` contain
  **73,820 records with 73,820 unique UIDs** (was: 73,820 records,
  73,816 unique UIDs).
- All three indexes (`character_index.json`, `group_index.json`,
  `theme_index.json`) have **zero duplicate UIDs** in their `verses`
  lists.
- Orphan records are tagged sensibly: `B3_C282_S1_orphan` → character
  `Markandeya` (matches the `[मार्क]` speaker tag);
  `B13_C33_S1_orphan` → group `Brahmins` + themes `honor`, `devotion`
  (matches the teaching content); `B6_C27_S5_orphan` → themes `peace`,
  `friendship` (matches BG 5.29 second half about knowing Krishna as
  friend of all beings).

Layer 1 is now complete and ready to feed Layer 2 retrieval.

---

## Layer 1 / Step 6 — Chapter / parva summaries (2026-06-08)

Library: `src/mahabharata/layer1/summaries.py` (CLI: `mbh-build-summaries`)
→ `data/layer1/chapter_summaries.jsonl` + `parva_summaries.json`
+ `reports/summaries_coverage_report.md`

### Approach — naive rollup, no LLM

Per architecture-doc Layer 1 Choice 4, this is a **naive rollup**: a
chapter "summary" is its verse-level Gemini `Summary` sections
concatenated in shloka order (`[S0] … [S1] …`), with no LLM pass. YAGNI
— Layer 1's job is to materialize the data Layer 2 retrieves; if naive
concatenation grounds good retrieval, the ~1,995-call Gemini pass was
never needed. Upgrade only if eval demands it.

Each `chapter_summaries.jsonl` record also carries the chapter's
aggregated character / group / theme tallies (summed from the Step 3/4
per-verse outputs), so a chapter record is a self-contained materialized
view: prose digest + faceted metadata + verse range. `parva_summaries.json`
is a compact 18-record book-level aggregate (counts + tallies + the list
of `chapter_uids`); it deliberately does **not** re-concatenate the full
text — a consumer wanting prose reads the referenced chapter records.

The book→parva name map was centralized into `common/parvas.py`
(previously duplicated in the Step 3 and Step 4 modules, now imported by
all three).

### Coverage + validation

- **73,820 verses → 1,995 chapters → 18 parvas.** avg 37.0 verses/chapter
  (min 4 = B12_C325, max 243 = B1_C2). 0 verses with an empty Summary.
- **Load-bearing parity:** Σ chapter `verse_count` == 73,820 (build aborts
  otherwise — every verse lands in exactly one chapter).
- **Step 6 ↔ Step 5 parity (exact, 0 mismatches):** summing each chapter's
  character/theme/group tallies reproduces the Step 5 index counts exactly
  (49,631 character-tags, 131,371 theme-tags, 6,981 group-tags) — the
  rollup is a faithful aggregation.
- Per-parva top-themes match narrative reality: war parvas (Bhishma/Drona/
  Karna) → battle/archery/chariot_warfare; Shanti/Anushasana → dharma/
  kingship; Stri → grief/death; Mausala → destruction/grief.

### Verse ordering caveat

The summary text is assembled in `(shloka, uid)` order, not raw-file
order (the corpus is ~61% non-monotonic in file order — see Step 5). The
4 `_orphan` verses sit at their bogus shloka position; naive rollup
doesn't re-thread them.

### Not wired into retrieval yet (deferred)

A chapter-level retrieval lane ("what is the Drona Parva about?") is
architecture-doc Layer 2 Choice 2, explicitly deferred until eval shows
verse-level retrieval underperforms that query shape. Step 6 builds the
data; consuming it in Layer 2 is separate, later work.

---

## Layer 2 / Phase A — Router & Retriever (2026-05-03)

Library: `src/mahabharata/layer2/router.py` + `retriever.py`
(CLI: `mbh-query`)

### Approach

Phase A is deliberately scoped to **structural + facet retrieval only**
— no embeddings, no BM25. The retriever is a pure-Python in-memory
join over the Layer 1 indexes, and the router is a hand-written rule
classifier rather than an LLM call. See
`theoretical_concepts_and_architecture.md` Layer 2 Choice 1 for the
rationale (latency, debuggability, and "the corpus is closed enough
that rules are tractable").

The router classifies a free-form query into one of four modes:

| Mode | Trigger | Example |
|---|---|---|
| `structural_uid` | matches `^B\d+_C\d+_S\d+$` | `B6_C27_S29` |
| `structural_slice` | matches `^B\d+(_C\d+)?$` | `B6`, `B6_C27` |
| `facet` | gazetteer matches ≥1 alias | `Krishna and yoga` |
| `unsupported` | nothing else hit | `Why did Karna refuse?` |

Precedence: UID > chapter slice > book slice > facet > unsupported.

The **gazetteer** is built once at retriever construction from
`entities.json` + `themes.json`. Aliases are stored lowercased and
sorted by descending length so longer phrases win the race
(prevents "Nara" from shadowing "Narayana"). After each successful
match the matched span is struck from the query so the same run of
text cannot double-tag.

### Retrieval modes

- **Structural UID**: direct dict lookup against `corpus_by_uid`.
- **Structural slice**: prefix-match over UIDs, sorted in numeric
  corpus order so `B10` comes after `B9`, not after `B1`.
- **Facet**: intersect the verse-sets from `character_index`,
  `group_index`, `theme_index`. Group facets union with their member
  characters' verses (Layer 1 Choice 3 — atomic group tagging,
  expand at query time). If the intersection is empty, fall back to
  the **union** with an explanatory note attached to the response.

### Loading strategy

`Retriever.from_paths()` loads the raw corpus into a
`uid -> record` dict at construction (~73,820 entries, small).
This trades a one-time load cost for free O(1) UID lookups and
saves per-query file scans. Tests / one-shot CLI invocations pay
the cost once; interactive REPL amortizes it across many queries.

### Known router bug (not yet fixed)

`UID_RE = r"^B(\d+)_C(\d+)_S(\d+)$"` rejects the 4 `_orphan`-suffixed
UIDs that `corpus_loader.UID_COLLISION_PATCH` synthesizes
(`B3_C282_S1_orphan` etc.). Those records exist in the corpus and are
findable via slice/facet queries, but cannot be looked up by direct
UID query. Fix is one regex tweak:
`^B(\d+)_C(\d+)_S(\d+)(_orphan)?$`.

---

## Layer 2 / Phase A — Eval set (2026-05-04)

Files: `data/layer2/eval/eval_set.jsonl` (final, 30 items)
+ `data/layer2/eval/eval_set_draft.jsonl` (provenance)
+ `data/layer2/reports/eval_bootstrap_report.md`

### Approach

Two-pass: a llama3 bootstrap (`mbh-bootstrap-eval`) generated 43
candidate questions from Layer 1 canonicals, then a hand-authored
final set of 30 atomic items. The final set is **structurally
distinct** from the draft — see "Why the draft was scrapped" below.

Each `EvalItem` has:

```json
{
  "id": "facet_001",
  "query": "Karna and friendship",
  "query_shape": "structural_uid|structural_slice|facet",
  "target_facets": {"characters": [...], "groups": [...], "themes": [...]},
  "known_good_uids": ["B1_C126_S15", ...],
  "notes": "...",
  "source": "hand"
}
```

`known_good_uids` are picked by inspecting top retrieval candidates
and selecting verses whose translation directly answers the question
— **not** by dumping the retriever's full output back as ground truth
(that would make every future eval pass at 100% by construction).

### Final set composition (30 items)

| Shape | Count | Example |
|---|---:|---|
| structural_uid | 3 | `B1_C1_S1`, `B6_C27_S29`, `B12_C1_S1` |
| structural_slice | 2 | `B6_C25` (BG ch1), `B17_C1` (Mahaprasthanika) |
| facet — char + theme | 15 | `Karna and friendship`, `Krishna and yoga` |
| facet — char + char | 3 | `Drona and Drupada`, `Bhima and Hidimba` |
| facet — char + char + theme | 4 | `Krishna and Arjuna on dharma` |
| facet — group + theme | 3 | `Pandavas in hermitage`, `Kauravas in war` |

Per-item known-goods: 19 items have 2 UIDs, 6 have 1, 3 have 3,
2 (slices) have 0 — slices assert non-empty at eval time.

All 30 items pass a smoke check: every `known_good_uid` is currently
reachable by the retriever, and every slice query returns ≥1 verse.

### Why the draft was scrapped

The Ollama bootstrap (43 items) had two failure modes that made it
unusable as a Phase A eval set:

1. **Narrative questions disguised as facet queries.** llama3 wrote
   things like *"Why did Karna refuse Arjuna's chariot offer?"* — the
   gazetteer matched `Karna`, `Arjuna`, `chariot_warfare`, etc., but
   no single Layer 1 record carries all those tags simultaneously.
   13 of 37 non-`concept` items hit the union-fallback branch with
   thousands of results — useless as an eval signal.
2. **Multi-step reasoning required.** Even when the intersection was
   non-empty, "the answer" to a narrative question is a *span* of
   verses, not a single UID. Phase A retrieval can't satisfy this,
   and `known_good_uids` as a flat list is the wrong shape for the
   ground truth anyway.

The 6 `concept_*` items routed straight to `unsupported` (Phase A
has no free-text path), confirming they're Phase C eval material.

The 13 union-fallback narrative questions are also Phase C material —
they need ranking + LLM-as-judge or graded relevance to evaluate.

### Eval-set design rules (going forward)

- Atomic questions only — each item has a clear, small set of "right
  answer" UIDs.
- Target intersection size ≤ ~60 verses (so picking known-goods is
  meaningful, not arbitrary).
- Avoid union-fallback. If the gazetteer extracts facets the corpus
  doesn't co-tag, the question is structurally unanswerable in
  Phase A — write a different question.
- Pick known-goods by inspecting translations, not by copying
  retriever output. The eval should be able to detect ranking
  regressions when Phase C is added.

### Open work

- **Eval harness** — the script that actually runs `eval_set.jsonl`
  through the retriever and reports recall/precision/Phase-A pass-fail
  doesn't exist yet. Next step.
- **Frozen regression anchor** — bootstrap report recommends carving
  out ~10 items "that will never be edited again" to detect drift.
  Not yet designated; obvious candidates are the 5 structural items
  + ~5 facet items with crisp single known-goods (e.g. `facet_004`
  BG 5.1–5.2, `facet_014` Drona's broken pride, `facet_018`
  Drona–Drupada).

---

## Layer 2 / Phase A — Eval harness + frozen anchor (2026-05-14)

Library: `src/mahabharata/layer2/eval_harness.py` (CLI: `mbh-eval`)
→ `data/layer2/reports/eval_results.md` + three rotating JSON slots.

### What it does

Loads `eval_set.jsonl`, runs every item through the retriever, scores
per shape, and writes a markdown report plus a JSON dump. Pass criteria:

| Shape | Pass rule |
|---|---|
| structural_uid | `known_good_uids[0]` present in results |
| structural_slice | `total > 0` AND every known-good present |
| facet | every known-good in top-`PASS_K` (K=10) |
| concept | `recall@10 > 0` — at least one curated answer in top-10 |

The concept rule is deliberately looser than facet's all-pins-in-top-K:
paraphrastic questions have many valid answers, a curator pins only
3–8, and demanding all of them in top-10 underreports a real win.
recall@10/@20, MRR, and `found_in_full` are still surfaced.

### Slot rotation

Three named JSON files, no unbounded growth:
`eval_results_latest.json` (overwritten each run), `..._previous.json`
(rotated from latest before each write), `..._baseline.json` (manually
pinned with `cp`, never touched by the harness). Answers the only two
diff questions that matter: "vs last run?" and "vs known-good baseline?".

### Frozen subset

10 items carry `frozen: true` and roll up into a separate aggregate
block — a regression anchor that should never go red. Currently 10/10.

---

## Layer 2 / Phase C Step 5 — BM25 lexical lane (through 2026-05-14)

Library: `src/mahabharata/layer2/bm25.py`.

Classical Okapi BM25 (`rank_bm25`, pure Python — no torch) over the
fluent English translations of all 73,820 verses. Index built in memory
at retriever construction (~1–2 s). Tokenization is lowercase +
alphanumeric-run split + a short hand-curated stopword list. The
stopword list was **not** optional: the initial no-stopword build scored
mean recall@10 = 0.0 on concept items because short function words
("why/should/we/not/the") dominated scoring over content terms.

BM25 replaced the Phase A `unsupported` branch — the router's old
fourth mode is now `concept`. It is also the **measurement anchor** for
everything that follows (dense, RRF): the standard IR baseline.

**Verdict from eval:** BM25 alone still failed all 15 concept items at
recall@10 = 0.0. The failure is structural vocabulary mismatch — the
curated answers say "grieve" where the query says "mourn", "slay" where
the query says "fight". No amount of IDF/stopword tuning fixes a
shared-token retriever when the right answer shares no tokens. This is
exactly the gap dense retrieval exists to close.

---

## Layer 2 / Phase C Step 6 — Dense bi-encoder lane (through 2026-05-14)

Libraries: `src/mahabharata/layer2/embed.py` (offline builder, CLI
`mbh-build-embeddings`) + `dense.py` (runtime).
→ `data/layer2/dense/embeddings.npy` (73820 × 384, float32,
L2-normalized) + `uids.txt` (parallel UID list).

Model: `BAAI/bge-small-en-v1.5` (~130 MB, 384-dim, English-only, CPU).
Embeds the fluent translation only (architecture doc Layer 2 Choice 3).
Retrieval is brute-force: one matmul of the query vector against the
matrix, argsort top-k — no FAISS (Choice 5; at this size cosine is
single-digit ms on CPU). The SentenceTransformer model loads lazily on
first concept query so structural/facet/UID lookups never pay for it.

The retriever loads the dense index optionally: if `embeddings.npy` and
`uids.txt` exist, concept mode uses dense; otherwise it falls back to
BM25 with a note. Lets the rest of the system run before
`mbh-build-embeddings` has been run.

**Verdict from eval (dense-only concept lane):** 12/15 concept items
pass (recall@10 > 0), but mean recall@10 = 0.31 and `found_in_full` =
0.27. Dense fixes the vocabulary-mismatch floor BM25 hit, but drifts
semantically — for "Why did Arjuna refuse to fight?" it returns generic
"Arjuna in battle" verses and misses the actual Gita refusal verses.
The 3 hard fails (concept_001/003/006) are kept failing on purpose as
**dense-miss canaries** — they are the empirical case for Step 7 (RRF).

---

## Layer 2 — Router hardening (2026-05-14)

Two gates added to `router.classify` to stop the gazetteer from stealing
concept queries before dense could rank them (it was mis-routing 12/15
concept items to `facet`):

1. **Question-shape override** — a query ending in `?` or led by a
   wh-word (why/how/what/when/where/who/whom/whose/which) routes to
   `concept` regardless of gazetteer hits. The matched entities/themes
   are still attached to the plan (telemetry + future entity-prior
   fusion), just not used to force a facet dump.
2. **Facet-coverage threshold (0.5)** — for non-question queries with
   gazetteer hits, the matched tokens must cover ≥ 50% of the query's
   post-stopword content tokens to qualify as facet. Below that the
   names are incidental and the query drops to concept. Calibrated
   against the 45-item set: every filter-style facet item is at coverage
   1.0, every concept item that escapes gate 1 is ≤ 0.33.

Also done: `UID_RE` now accepts the 4 `_orphan`-suffixed UIDs
(`^B(\d+)_C(\d+)_S(\d+)(_orphan)?$`), closing the Phase A known bug.

---

## Layer 2 — Eval set v2: 45 items + concept shape (2026-05-14)

`data/layer2/eval/eval_set.jsonl` grew from 30 → **45 items** by adding
a fourth `query_shape`, **concept** (free-text/paraphrase), with 15
hand-curated items. Composition now:

| Shape | Count | Pass rule |
|---|---:|---|
| structural_uid | 3 | KG in results |
| structural_slice | 2 | non-empty + all KGs present |
| facet | 25 | all KGs in top-10 |
| concept | 15 | recall@10 > 0 |

Concept known-goods were curated by triaging the dense top-20 per query
(`data/layer2/eval/concept_curation_review.md`) and pinning only verses
whose translation genuinely answers the question — never by copying
retriever output back as ground truth (that would peg every future run
at 100%).

### Current baseline (`eval_results_latest.json`, 2026-05-14)

- **Overall: 42/45 (93.3%)** · router accuracy 1.0 · union-fallbacks 0
- structural_uid 3/3 · structural_slice 2/2 · facet 25/25 · **concept 12/15**
- facet: mean recall@10 = 1.0, mean MRR = 0.65, found-in-full 1.0
- concept (dense-only): mean recall@10 = 0.31, recall@20 = 0.41, MRR = 0.51
- frozen subset 10/10

This is the number Step 7 (RRF) has to move.

---

## Layer 2 / Phase C Step 7 — RRF tested & rejected; route-by-shape + lexical mode (2026-06-08)

New code: `src/mahabharata/layer2/fusion.py` (`reciprocal_rank_fusion`,
k=60). Router gains a fifth mode, `lexical`.

### What we tested

The plan was "RRF fusion of BM25 + dense is the next empirical test."
We built it and ran it. **It regressed.** Equal-weight RRF on the
concept set dropped mean recall@10 from 0.312 (dense-only) to 0.175.
The mechanism: BM25 scores ~0 relevance on paraphrastic queries, and
RRF fuses on rank position, so BM25's wrong top hits interleave with
dense's correct ones and displace them. Weighted RRF (dense:bm25 up to
10:1) only crawls back *toward* dense-only (0.295 < 0.312); no weighting
beats dense-only. The one extra binary pass RRF bought was an artifact
of the loose `recall@10 > 0` gate.

### Why route-by-shape instead of fusion

A 7-item **lexical eval** (`lexical_eval_staging.jsonl`, rare-anchor
keyword queries: "house of lac", "iron mace", "Panchajanya conch", …)
showed the mirror image: on lexical queries BM25 recall@10 = 0.619 vs
dense 0.167 — **BM25 ~4x dense**. So the two lanes aren't strong-vs-weak;
each is ~4x better on *its own* query shape. Fusing them dilutes
whichever lane is correct. The right move is to **route to the single
best lane per query shape**, not fuse.

### How lexical is detected: explicit quotes

Calibration against the eval queries showed that once question-form
(concept) and high-coverage (facet) queries are removed, bare short
phrases are *syntactically identical* whether the intent is lexical
("iron mace") or concept ("seeking revenge"). The router cannot infer
exact-keyword intent from a bare phrase without a model, and a
token-count heuristic overfits ~10 examples. So lexical intent is an
**explicit signal — wrap the query in double quotes** (`"iron mace"`),
matching the architecture doc's own lexical example. Router precedence:
UID > slice > **quoted lexical** > question-shape concept > high-coverage
facet > concept. Quoted-lexical sits above facet so `"iron mace"` reaches
BM25 instead of being grabbed by the "mace" theme.

### New baseline (`eval_results_latest.json`, 2026-06-08)

Eval set grew 45 → **52 items** (added the 7 quoted lexical items).

- **Overall: 49/52 (94.2%)** · router accuracy 1.0 · union-fallbacks 0
- structural_uid 3/3 · structural_slice 2/2 · **lexical 7/7** · facet 25/25 · concept 12/15
- lexical (BM25): mean recall@10 = 0.619, recall@20 = 0.857, found-in-full 1.0
- concept (dense): mean recall@10 = 0.312 — **restored** (the RRF regression is reverted)
- frozen subset 10/10

`fusion.py` stays in the tree for a future genuine "mixed" route, but is
not on the default retrieval path.

---

## Layer 2 → Layer 3 — Concept-canary investigation & pivot (2026-06-10)

Goal: rescue the 3 intentional concept canaries (`concept_001` Arjuna
refuses to fight → B6_C24; `concept_003` Draupadi disrobed →
B2_C60/62/72; `concept_006` Bhishma's vow → B1_C94). **Four retrieval-side
fixes were tested; all rejected.** This was a measurement arc, not a
build — no `src/` changes shipped.

### What was tried (and why each failed)

1. **Entity-prior fusion** ✗ — intersect/boost dense results with the
   entity facet set. Fails because the KGs often aren't entity-tagged at
   all (5/6 of concept_001's KGs are *not* in Arjuna's facet set — they're
   Arjuna's own words, which Step 3's keyword-NER doesn't tag with his
   name), and the entity sets are too broad (900–5,900 verses) to
   discriminate. Intersecting would *drop* the right answers.
2. **bge query-instruction prefix** ✗ — bge-small documents a query
   prefix for asymmetric retrieval; we weren't using it. Adding it
   slightly *hurts* here (mean concept recall@10 0.312 → 0.267).
3. **Cross-encoder reranker** ✗ (by inference) — the canary KGs sit at
   verse-dense rank **7,000–53,000**. A reranker only reorders a ~top-200
   candidate pool; it cannot recover verses that deep. The problem is
   **recall, not ranking.**
4. **Verse/chapter cosine blend** ✗ — a principled linear blend of two
   comparable bge cosines (`α·verse + (1−α)·chapter`), swept over the full
   concept set. No α beats verse-dense-only (α=1.0, r@10 0.312): the best
   blend rescues *only* concept_003 (to 0.25) while regressing every
   concept query verse-dense already handles (mean → 0.292). The same
   broad-regression-for-one-binary-pass trade that sank RRF.

### The diagnosis (why retrieval tuning is the wrong layer)

The canaries are **reasoning / scene** questions ("why did X…", "what did
X feel…"); the other concept items are **lookup** questions. Any *always-on*
combination that helps the former dilutes the latter, and the router can't
distinguish them from a free-text query (the same lexical-vs-concept
ambiguity that forced explicit quote-gating). So verse-level retrieval has
a **structural ceiling** on reasoning questions — it's not a tuning gap.

### What the arc produced

**Chapter-level dense retrieval works as a scene-localizer:** concept_003
→ chapter B2_C62 ranks **#1**, concept_001 → B6_C24 ranks **#22** (up from
verse rank ~7,000–53,000). Built and persisted (gitignored, derived):
`data/layer2/dense/chapter_embeddings.npy` (1,995×384, bge over the Step 6
`chapter_summaries.jsonl`) + `chapter_uids.txt`. concept_006's B1_C94 still
ranks ~1,016 — likely bge's 512-token truncation of a long summary
(chunking is a future refinement).

Chapter retrieval's correct home is **Layer 3 as the context-fetch step**,
not an always-on concept-lane blend: a synthesis model reads the localized
chapter and *reasons* the answer, tolerating imperfect verse-recall in a
way the strict "KG verse in top-10" metric never will. **Decision: pivot to
Layer 3 synthesis**, consuming the chapter index just built. Layer 1 & 2
remain open for later iteration (chapter-summary chunking; an explicit
user-controlled parva/chapter retrieval mode).

---

## Data Audit Findings (2026-04-12)

Script: `scripts/audit_ai_analysis.py` (standalone, run with
`.venv/bin/python scripts/audit_ai_analysis.py`)

### Coverage
- All 73,820 verses have `ai_analysis` — zero null, zero empty, zero malformed
- All 5 sections present in every record across all 18 books

### Section length (chars)
| Section            | Avg | Median | Min | Max   |
|--------------------|-----|--------|-----|-------|
| Fluent Translation | 136 | 130    | 12  | 2,613 |
| Poetic Translation | 144 | 146    | 15  | 724   |
| Summary            | 113 | 109    | 17  | 353   |
| Keywords           | 47  | 45     | 11  | 230   |
| Dictionary         | 293 | 281    | 39  | 1,772 |

### Keyword stats
- 393,233 total keyword tokens; 5.3 avg per verse; 24,441 unique tokens
- Top characters in keywords: Arjuna (5,427), Krishna (3,105), Karna (2,687),
  Yudhishthira (2,516), Bhishma (2,231), Drona (2,075), Duryodhana (2,010),
  Bhima (1,998), Dhritarashtra (1,699), Bhimasena (1,031)
- Top themes: battle (7,147), dharma (3,212), death (2,506), anger (2,132),
  sacrifice (1,838), knowledge (1,371), wisdom (1,302), truth (1,301)

### Implications for Layer 1
- **Character index**: keywords already name characters consistently — build
  character→verse list directly from keywords (no NER needed)
- **Theme tagging**: dharma, karma, sacrifice, devotion etc. appear as explicit
  keywords — a controlled vocabulary map will cover most cases
- **Chapter rollup**: Summary (avg 113 chars) is short enough to aggregate cleanly
- **Translation layer**: Dictionary section (avg 293 chars) = Sanskrit word glosses
  per verse, ready to use as seq2seq training pairs

### Parser note
The `ai_analysis` section parser must anchor `**Header:**` matches to line-start
(`re.MULTILINE`). A naive regex matching `\*\*\w` will terminate Dictionary early
because dictionary entries themselves use bold markers (`* **word**: def`).

---

## Key Lessons from Previous Project (don't repeat)

- Always cap eval dataset — full 29K eval = 17 min per pass
- HF cache must go on large disk, not root
- `q4_k_m` is not a valid `--outtype` for convert_hf_to_gguf.py — need two-step convert + quantize
- Merging a 4-bit model leaves bitsandbytes metadata — reload base in bf16 for clean merge
- RunPod workspace is at `/workspace`, not `~/workspace`
- The deduplication logic in the old search engine silently dropped most verses — always verify corpus size
