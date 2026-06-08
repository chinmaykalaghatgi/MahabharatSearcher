# MahabharataNew

A lean, locally-running Mahabharata research tool: pre-built structured
knowledge over the full 73,820-verse corpus + routed retrieval, designed
so query-time is a lookup problem rather than a reasoning problem. Also a
personal AI-engineering learning project.

**Full design + status:** [`docs/project_context.md`](docs/project_context.md)
(the *what* and current state) and
[`docs/theoretical_concepts_and_architecture.md`](docs/theoretical_concepts_and_architecture.md)
(the *why* behind each layer).

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

This registers the `mbh-*` console scripts (build steps, query, eval).

## Data policy (what's *not* in git)

Git tracks code, docs, and the small hand-authored artifacts (the
entity/theme gazetteers + overrides, the eval sets, curation reviews,
coverage reports, and the pinned eval baseline). It deliberately does
**not** track:

| Excluded | Size | How to restore |
|---|---|---|
| `data/raw/search_engine_db.jsonl` | 164 MB | **Irreplaceable source** — back up separately; everything else derives from it |
| `data/layer2/dense/` (embeddings) | 108 MB | `mbh-build-embeddings` |
| `data/layer1/*` tagging/indexes/summaries | ~25 MB | `mbh-build-verse-characters`, `-verse-themes`, `-indexes`, `-summaries` |

Both giants exceed GitHub's 100 MB file limit, and the derived data is
regenerable, so they live outside version control. The raw corpus is the
one file you must preserve yourself.

## Rebuild the derived data from raw

With `data/raw/search_engine_db.jsonl` in place, run in order:

```bash
.venv/bin/mbh-build-entities
.venv/bin/mbh-build-themes
.venv/bin/mbh-build-verse-characters
.venv/bin/mbh-build-verse-themes
.venv/bin/mbh-build-indexes
.venv/bin/mbh-build-summaries
.venv/bin/mbh-build-embeddings      # downloads BAAI/bge-small-en-v1.5
```

## Query / evaluate

```bash
.venv/bin/mbh-query "Krishna and yoga"     # facet
.venv/bin/mbh-query '"iron mace"'          # lexical (quoted -> BM25)
.venv/bin/mbh-query "Why should we not mourn the dead?"   # concept (dense)
.venv/bin/mbh-eval                          # run the eval set
```
