"""Phase B Step 2 — Bootstrap the Layer 2 eval set via local Ollama.

Feeds Layer 1 artifacts (top characters, top themes) to `llama3-8k:latest`
over the Ollama HTTP API and asks for plausible user queries per shape:

    facet/char          one character, various angles
    facet/theme         one theme, various angles
    facet/char_theme    character × theme combination
    concept             paraphrase of a theme that does NOT use the theme word
    structural_uid      (hand-coded, not generated — trivial to write)
    structural_slice    (hand-coded)

Output is `eval_set_draft.jsonl` — one `EvalItem` per line. The user
then hand-edits this down to the ~30-50 final queries, filling in
`known_good_uids` by inspection. See theoretical_concepts_and_architecture.md
Layer 2 Choice 8 for the rationale.

Streaming design
----------------
Every Ollama call is streamed token-by-token to stderr so the user can
see generation progress in real time — useful because 8B inference on
CPU is slow enough that a silent wait feels broken. The same tokens are
collected into a string for parsing after `done=true`.

Prompts are deliberately strict about format ("one question per line,
no numbering, no preamble") because even with temperature=0.7 llama3
likes to add "Here are three questions:" or Markdown list markers.
Post-processing strips what we can and drops the rest.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mahabharata.common.paths import rel


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3-8k:latest"

# --- knobs ---
N_TOP_CHARACTERS = 8         # top chars by verse count
N_PER_CHARACTER = 2          # queries per character
N_TOP_THEMES = 6             # top themes by verse count
N_PER_THEME = 2              # queries per theme
N_COMBOS = 6                 # char × theme combinations
N_CONCEPT = 6                # concept paraphrases (one per theme)


@dataclass
class EvalItem:
    id: str
    query: str
    query_shape: str
    target_facets: dict
    known_good_uids: list[str] = field(default_factory=list)
    notes: str = ""
    source: str = "ollama"


# --- Ollama streaming client ---

class OllamaClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str = OLLAMA_URL,
        stream_to_stderr: bool = True,
    ):
        self.model = model
        self.url = url
        self.stream_to_stderr = stream_to_stderr

    def generate(self, prompt: str, *, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        chunks: list[str] = []
        try:
            with urllib.request.urlopen(req) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    tok = obj.get("response", "")
                    if tok:
                        chunks.append(tok)
                        if self.stream_to_stderr:
                            sys.stderr.write(tok)
                            sys.stderr.flush()
                    if obj.get("done"):
                        break
        except urllib.error.URLError as e:
            raise SystemExit(
                f"[ERROR] Could not reach Ollama at {self.url}: {e}. "
                f"Is `ollama serve` running?"
            )
        if self.stream_to_stderr:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return "".join(chunks)


# --- prompts ---

PROMPT_CHARACTER = """\
You are helping build an evaluation set for a Mahabharata question-answering system.
Generate exactly {n} natural, diverse user questions about the character named "{canon}" in the Mahabharata.
Each question should be specific enough to have a verse-level answer. Each should take a different angle — do not repeat.

Output format rules:
- Output exactly {n} questions, one per line.
- No numbering, no bullets, no preamble, no commentary.
- Plain text only.

Character: {canon}
Also known as: {aliases}
"""

PROMPT_THEME = """\
You are helping build an evaluation set for a Mahabharata question-answering system.
Generate exactly {n} natural, diverse user questions about the theme "{canon}" as it appears in the Mahabharata.
Each question should be answerable by pointing at specific verses.

Output format rules:
- Output exactly {n} questions, one per line.
- No numbering, no bullets, no preamble, no commentary.
- Plain text only.

Theme: {canon}
"""

PROMPT_COMBO = """\
You are helping build an evaluation set for a Mahabharata question-answering system.
Generate exactly 1 natural user question that ties the character "{char}" to the theme "{theme}" in the Mahabharata.
Avoid the generic phrasing "What does X say about Y" — find a more specific angle.

Output format rules:
- Output exactly 1 question, a single line.
- No numbering, no bullets, no preamble, no commentary.
"""

PROMPT_CONCEPT = """\
You are helping build an evaluation set for a Mahabharata question-answering system.
Generate exactly 1 natural user question about the theme "{canon}" as it appears in the Mahabharata.
CRITICAL: the question must NOT contain the word "{canon}" or any obvious direct synonym of it.
The point is to test whether a paraphrase-aware retrieval system can find "{canon}"-tagged verses from a question that never says the word.

Output format rules:
- Output exactly 1 question, a single line.
- No numbering, no bullets, no preamble, no commentary.
"""


# --- parsing ---

_NUMBERING_RE = re.compile(r"^\s*(?:\d+[.)]\s+|[-*]\s+)")


def parse_question_lines(text: str, expected: int) -> list[str]:
    """Extract clean question lines from an Ollama response blob.

    Strips numbering, bullets, and commentary lines (anything that
    doesn't look like a question). Keeps at most `expected` lines.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: list[str] = []
    for ln in lines:
        ln = _NUMBERING_RE.sub("", ln).strip()
        if not ln or len(ln) < 10:
            continue
        if ln.endswith(":"):  # headers like "Here are three questions:"
            continue
        # Drop obvious preambles
        low = ln.lower()
        if low.startswith(("here are", "here's", "sure", "certainly")):
            continue
        cleaned.append(ln)
        if len(cleaned) >= expected:
            break
    return cleaned


# --- shape generators ---

def top_n_by_count(index: dict, n: int) -> list[str]:
    """The indexes are already sorted by descending count, but be
    explicit so changing upstream ordering later doesn't silently
    break us."""
    rows = sorted(
        index.items(), key=lambda kv: -kv[1].get("count", 0)
    )
    return [canon for canon, _ in rows[:n]]


def generate_character_queries(
    client: OllamaClient,
    char_index: dict,
    entities: dict,
    *,
    n_chars: int = N_TOP_CHARACTERS,
    n_per: int = N_PER_CHARACTER,
) -> list[EvalItem]:
    items: list[EvalItem] = []
    chars = top_n_by_count(char_index, n_chars)
    characters_meta = entities.get("characters", {})
    for i, canon in enumerate(chars):
        aliases = characters_meta.get(canon, {}).get("aliases", [])
        alias_str = ", ".join(aliases[:8]) if aliases else "(none)"
        _log_header(f"[char {i+1}/{len(chars)}] {canon} ({n_per} queries)")
        prompt = PROMPT_CHARACTER.format(n=n_per, canon=canon, aliases=alias_str)
        text = client.generate(prompt)
        questions = parse_question_lines(text, expected=n_per)
        for j, q in enumerate(questions):
            items.append(EvalItem(
                id=f"char_{i+1:03d}_{j+1}",
                query=q,
                query_shape="facet",
                target_facets={"characters": [canon]},
                notes=f"Generated for character={canon} (verse_count={char_index[canon]['count']}). Hand-edit known_good_uids.",
                source="ollama",
            ))
    return items


def generate_theme_queries(
    client: OllamaClient,
    theme_index: dict,
    *,
    n_themes: int = N_TOP_THEMES,
    n_per: int = N_PER_THEME,
) -> list[EvalItem]:
    items: list[EvalItem] = []
    themes = top_n_by_count(theme_index, n_themes)
    for i, canon in enumerate(themes):
        _log_header(f"[theme {i+1}/{len(themes)}] {canon} ({n_per} queries)")
        prompt = PROMPT_THEME.format(n=n_per, canon=canon)
        text = client.generate(prompt)
        questions = parse_question_lines(text, expected=n_per)
        for j, q in enumerate(questions):
            items.append(EvalItem(
                id=f"theme_{i+1:03d}_{j+1}",
                query=q,
                query_shape="facet",
                target_facets={"themes": [canon]},
                notes=f"Generated for theme={canon} (verse_count={theme_index[canon]['count']}). Hand-edit known_good_uids.",
                source="ollama",
            ))
    return items


# Hand-picked char × theme combos. Picked for narrative salience: each
# pair is a combination we'd expect a Mahabharata reader to actually
# ask about, which gives a stronger eval signal than random pairing.
DEFAULT_COMBOS = [
    ("Krishna", "dharma"),
    ("Arjuna", "grief"),
    ("Yudhishthira", "truth"),
    ("Bhishma", "death"),
    ("Karna", "loyalty"),
    ("Duryodhana", "anger"),
]


def generate_combo_queries(
    client: OllamaClient,
    char_index: dict,
    theme_index: dict,
    *,
    combos: list[tuple[str, str]] = DEFAULT_COMBOS,
) -> list[EvalItem]:
    items: list[EvalItem] = []
    usable = [(c, t) for c, t in combos if c in char_index and t in theme_index]
    for i, (char, theme) in enumerate(usable):
        _log_header(f"[combo {i+1}/{len(usable)}] {char} × {theme}")
        prompt = PROMPT_COMBO.format(char=char, theme=theme)
        text = client.generate(prompt)
        questions = parse_question_lines(text, expected=1)
        for q in questions:
            items.append(EvalItem(
                id=f"combo_{i+1:03d}",
                query=q,
                query_shape="facet",
                target_facets={"characters": [char], "themes": [theme]},
                notes=f"Generated for {char} × {theme}. Hand-edit known_good_uids.",
                source="ollama",
            ))
    return items


# Themes chosen for concept paraphrases — pick themes whose canonical
# word users might not reach for naturally, so the test exercises
# paraphrase retrieval and not keyword matching.
DEFAULT_CONCEPT_THEMES = [
    "dharma",
    "fate",
    "grief",
    "devotion",
    "kingship",
    "renunciation",
]


def generate_concept_queries(
    client: OllamaClient,
    theme_index: dict,
    *,
    themes: list[str] = DEFAULT_CONCEPT_THEMES,
) -> list[EvalItem]:
    items: list[EvalItem] = []
    usable = [t for t in themes if t in theme_index]
    for i, canon in enumerate(usable):
        _log_header(f"[concept {i+1}/{len(usable)}] paraphrase of {canon}")
        prompt = PROMPT_CONCEPT.format(canon=canon)
        text = client.generate(prompt)
        questions = parse_question_lines(text, expected=1)
        for q in questions:
            items.append(EvalItem(
                id=f"concept_{i+1:03d}",
                query=q,
                query_shape="concept",
                target_facets={"themes": [canon]},
                notes=f"Paraphrase test for theme={canon}. Model was asked NOT to use the word. Hand-verify + fill known_good_uids.",
                source="ollama",
            ))
    return items


def handwritten_structural_items() -> list[EvalItem]:
    """Structural queries don't need bootstrapping — they're trivial
    to write and their "known-good" answers are mechanical. Kept
    minimal because recall@k on an exact-UID lookup is not the
    interesting metric."""
    return [
        EvalItem(
            id="struct_uid_001",
            query="B1_C1_S1",
            query_shape="structural_uid",
            target_facets={},
            known_good_uids=["B1_C1_S1"],
            notes="Opening verse of the corpus. Sanity check for exact UID lookup.",
            source="hand",
        ),
        EvalItem(
            id="struct_uid_002",
            query="B6_C27_S29",
            query_shape="structural_uid",
            target_facets={},
            known_good_uids=["B6_C27_S29"],
            notes="Exercised during Phase A smoke test — BG 5.29 first half.",
            source="hand",
        ),
        EvalItem(
            id="struct_slice_001",
            query="B1_C1",
            query_shape="structural_slice",
            target_facets={},
            known_good_uids=[],
            notes="First chapter slice. Known-good = every UID matching B1_C1_S*; leave empty and assert non-empty at eval time.",
            source="hand",
        ),
        EvalItem(
            id="struct_slice_002",
            query="B6_C27",
            query_shape="structural_slice",
            target_facets={},
            known_good_uids=[],
            notes="BG chapter containing the famous 5.29 verse.",
            source="hand",
        ),
    ]


# --- logging helpers ---

def _log_header(msg: str):
    sys.stderr.write(f"\n── {msg} ──\n")
    sys.stderr.flush()


def _log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# --- orchestrator ---

def run(
    *,
    char_index_path: Path,
    theme_index_path: Path,
    entities_path: Path,
    draft_path: Path,
    report_path: Path,
    model: str = DEFAULT_MODEL,
):
    with open(char_index_path) as f:
        char_index = json.load(f)
    with open(theme_index_path) as f:
        theme_index = json.load(f)
    with open(entities_path) as f:
        entities = json.load(f)

    _log(f"Bootstrapping eval set via Ollama model: {model}")
    _log(
        f"  knobs: {N_TOP_CHARACTERS} chars × {N_PER_CHARACTER}, "
        f"{N_TOP_THEMES} themes × {N_PER_THEME}, "
        f"{len(DEFAULT_COMBOS)} combos, "
        f"{len(DEFAULT_CONCEPT_THEMES)} concept paraphrases"
    )

    client = OllamaClient(model=model)
    t_start = time.time()

    all_items: list[EvalItem] = []
    all_items += handwritten_structural_items()
    all_items += generate_character_queries(client, char_index, entities)
    all_items += generate_theme_queries(client, theme_index)
    all_items += generate_combo_queries(client, char_index, theme_index)
    all_items += generate_concept_queries(client, theme_index)

    elapsed = time.time() - t_start

    draft_path.parent.mkdir(parents=True, exist_ok=True)
    with open(draft_path, "w") as f:
        for item in all_items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    _log(f"\nWrote {len(all_items)} draft items → {rel(draft_path)}")

    write_report(
        items=all_items,
        elapsed=elapsed,
        model=model,
        report_path=report_path,
    )


def write_report(*, items, elapsed, model, report_path: Path):
    by_shape: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for it in items:
        by_shape[it.query_shape] = by_shape.get(it.query_shape, 0) + 1
        by_source[it.source] = by_source.get(it.source, 0) + 1

    lines = []
    lines.append("# Layer 2 eval set — bootstrap report")
    lines.append("")
    lines.append(
        "Auto-generated by `mbh-bootstrap-eval`. This is a DRAFT. Review, "
        "hand-edit to ~30-50 queries, fill in `known_good_uids` per item, "
        "then save as `eval_set.jsonl`."
    )
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- model: `{model}`")
    lines.append(f"- elapsed: {elapsed:.1f}s")
    lines.append(f"- total draft items: **{len(items)}**")
    lines.append("")
    lines.append("## Breakdown by query shape")
    lines.append("")
    lines.append("| Shape | Count |")
    lines.append("|---|---:|")
    for shape, count in sorted(by_shape.items()):
        lines.append(f"| {shape} | {count} |")
    lines.append("")
    lines.append("## Breakdown by source")
    lines.append("")
    lines.append("| Source | Count |")
    lines.append("|---|---:|")
    for src, count in sorted(by_source.items()):
        lines.append(f"| {src} | {count} |")
    lines.append("")
    lines.append("## Per-item draft preview")
    lines.append("")
    for it in items:
        lines.append(f"- **{it.id}** [{it.query_shape}] `{it.query}`")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append(
        "1. Review each item. Delete duplicates and obvious junk; rewrite "
        "anything that sounds like model-speak."
    )
    lines.append(
        "2. For each kept item, run the query through `mbh-query` and "
        "eyeball the top results. Copy the UIDs that actually answer the "
        "question into `known_good_uids`."
    )
    lines.append(
        "3. Save as `eval_set.jsonl` (final, hand-curated). The draft "
        "stays in place as a provenance record."
    )
    lines.append(
        "4. Keep a frozen subset of ~10 items that will never be edited "
        "again, as a regression anchor (see theoretical_concepts_and_architecture.md, "
        "Layer 2 open questions)."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    _log(f"Wrote report → {rel(report_path)}")
