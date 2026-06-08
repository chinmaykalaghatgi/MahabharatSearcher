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

Git tracks code, docs, the small hand-authored artifacts (the
entity/theme gazetteers + overrides, the eval sets, curation reviews,
coverage reports, the pinned eval baseline), **and the raw corpus
itself, gzipped**: `data/raw/search_engine_db.jsonl.gz` (~33 MB vs
164 MB raw). `corpus_loader` reads the `.gz` transparently, so no manual
decompression is needed — a fresh clone can rebuild everything.

It deliberately does **not** track the large *derived* artifacts, which
are regenerable:

| Excluded | Size | How to restore |
|---|---|---|
| `data/raw/search_engine_db.jsonl` (uncompressed) | 164 MB | `gunzip -k data/raw/search_engine_db.jsonl.gz` (optional — code reads the `.gz`) |
| `data/layer2/dense/` (embeddings) | 108 MB | `mbh-build-embeddings` |
| `data/layer1/*` tagging/indexes/summaries | ~25 MB | `mbh-build-verse-characters`, `-verse-themes`, `-indexes`, `-summaries` |

The 108 MB embeddings exceed GitHub's 100 MB file limit and the rest is
cheap to regenerate, so they stay out. The corpus is small enough
gzipped to live in the repo, so the irreplaceable source travels with it.

## Rebuild the derived data from raw

The corpus ships with the repo (gzipped) and `corpus_loader` reads it
directly, so a fresh clone can regenerate everything by running, in order:

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
