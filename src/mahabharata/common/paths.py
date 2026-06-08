"""Centralized project paths.

All build scripts import constants from here instead of recomputing
`Path(__file__).parent` — keeps path layout in one place so reorganizing
the data directory later is a single-file change. Library functions
still accept explicit path args; these are just the canonical defaults
the CLI wrappers pass in.
"""

from pathlib import Path

# src/mahabharata/common/paths.py  →  parents[3] is the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PATH = RAW_DIR / "search_engine_db.jsonl"

LAYER1_DIR = DATA_DIR / "layer1"
LAYER1_REPORTS_DIR = LAYER1_DIR / "reports"

# Layer 1 artifacts
ENTITIES_PATH = LAYER1_DIR / "entities.json"
ENTITIES_OVERRIDES_PATH = LAYER1_DIR / "entities_overrides.json"
ENTITIES_REPORT_PATH = LAYER1_REPORTS_DIR / "entities_coverage_report.md"

THEMES_PATH = LAYER1_DIR / "themes.json"
THEMES_OVERRIDES_PATH = LAYER1_DIR / "themes_overrides.json"
THEMES_REPORT_PATH = LAYER1_REPORTS_DIR / "themes_coverage_report.md"

VERSE_CHARS_PATH = LAYER1_DIR / "verse_characters.jsonl"
VERSE_CHARS_REPORT_PATH = (
    LAYER1_REPORTS_DIR / "verse_characters_coverage_report.md"
)

VERSE_THEMES_PATH = LAYER1_DIR / "verse_themes.jsonl"
VERSE_THEMES_REPORT_PATH = (
    LAYER1_REPORTS_DIR / "verse_themes_coverage_report.md"
)

CHAR_INDEX_PATH = LAYER1_DIR / "character_index.json"
GROUP_INDEX_PATH = LAYER1_DIR / "group_index.json"
THEME_INDEX_PATH = LAYER1_DIR / "theme_index.json"
INDEXES_REPORT_PATH = LAYER1_REPORTS_DIR / "indexes_coverage_report.md"

# Step 6 — chapter/parva summaries (naive rollup, no LLM)
CHAPTER_SUMMARIES_PATH = LAYER1_DIR / "chapter_summaries.jsonl"
PARVA_SUMMARIES_PATH = LAYER1_DIR / "parva_summaries.json"
SUMMARIES_REPORT_PATH = LAYER1_REPORTS_DIR / "summaries_coverage_report.md"

# Layer 2 artifacts
LAYER2_DIR = DATA_DIR / "layer2"
LAYER2_REPORTS_DIR = LAYER2_DIR / "reports"
LAYER2_EVAL_DIR = LAYER2_DIR / "eval"
EVAL_SET_DRAFT_PATH = LAYER2_EVAL_DIR / "eval_set_draft.jsonl"
EVAL_SET_PATH = LAYER2_EVAL_DIR / "eval_set.jsonl"
EVAL_BOOTSTRAP_REPORT_PATH = LAYER2_REPORTS_DIR / "eval_bootstrap_report.md"

# Eval-harness outputs — three named slots + a markdown render.
# `latest` is overwritten on every run; the prior `latest` is rotated
# into `previous` first so you can diff the two most recent runs.
# `baseline` is manually pinned (cp latest baseline) when you want a
# durable anchor — never overwritten by the harness.
EVAL_RESULTS_LATEST_PATH = LAYER2_REPORTS_DIR / "eval_results_latest.json"
EVAL_RESULTS_PREVIOUS_PATH = LAYER2_REPORTS_DIR / "eval_results_previous.json"
EVAL_RESULTS_BASELINE_PATH = LAYER2_REPORTS_DIR / "eval_results_baseline.json"
EVAL_RESULTS_REPORT_PATH = LAYER2_REPORTS_DIR / "eval_results.md"

# Phase C / Step 6 — dense embedding index over fluent translations.
# Model: BAAI/bge-small-en-v1.5 (384-dim, English-only). Built offline
# by `mbh-build-embeddings`; loaded at retriever init for concept-mode
# retrieval. Stored as a single (N, 384) float32 numpy matrix + a
# parallel UID list (one per line, same order as matrix rows).
LAYER2_DENSE_DIR = LAYER2_DIR / "dense"
DENSE_EMBEDDINGS_PATH = LAYER2_DENSE_DIR / "embeddings.npy"
DENSE_UIDS_PATH = LAYER2_DENSE_DIR / "uids.txt"


def rel(path: Path) -> str:
    """Display path relative to project root (for log messages)."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
