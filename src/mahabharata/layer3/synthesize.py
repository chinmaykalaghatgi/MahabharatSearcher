"""Layer 3 synthesis — grounded generation over assembled context.

Takes a ``ContextBundle`` (from ``context.py``) and a local Ollama model,
and produces a grounded answer. The contract (architecture-doc Layer 3
Choices 1 + 5):

  - Answer ONLY from the supplied context — never from parametric
    knowledge of the Mahabharata.
  - Cite the verse UIDs that support each claim, in square brackets.
  - Abstain with an explicit sentinel when the context doesn't answer
    the question, rather than fabricating.

Generation runs at low temperature for reproducibility. The Ollama call
mirrors the stdlib-urllib pattern already used in
``layer2.eval_bootstrap`` — no new dependency, talks to a local
``ollama serve`` at :11434.

This is v1: a single shot over fixed context, off-the-shelf instruct
model, no fine-tuning. Fine-tuning and an agentic multi-hop loop are
gated on eval evidence (Layer 3 open questions).
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from mahabharata.layer3.context import ContextBundle

# A bracketed verse/chapter UID anywhere in the model's answer.
_CITE_RE = re.compile(r"\[(B\d+_C\d+(?:_S\d+(?:_orphan)?)?)\]")
# Bare shloka markers inside a chapter summary, e.g. "[S11]".
_SHLOKA_MARKER_RE = re.compile(r"\[S(\d+)\]")

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3-8k:latest"

# Sentinel the model is told to emit when the context can't answer.
ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = (
    "You are a careful scholar of the Mahabharata answering questions "
    "STRICTLY from the supplied excerpts. Rules:\n"
    "1. Use ONLY the provided chapter summaries and verses. Do not use "
    "outside knowledge of the epic, even if you know it.\n"
    "2. Cite the verse UID(s) that support each claim in square brackets, "
    "e.g. [B6_C24_S11]. Cite a chapter UID (e.g. [B6_C24]) for "
    "scene-level statements drawn from a chapter summary.\n"
    "3. If the excerpts do not contain enough to answer, reply with "
    f"exactly: {ABSTAIN_TOKEN}\n"
    "4. Be concise and grounded. Do not speculate beyond the text."
)


@dataclass
class SynthesisResult:
    query: str
    answer: str
    abstained: bool
    cited_uids: list[str]
    model: str
    bundle: ContextBundle


def build_prompt(bundle: ContextBundle) -> str:
    """Render the context bundle into the user-side prompt."""
    parts: list[str] = [f"QUESTION: {bundle.query}", ""]

    if bundle.chapters:
        parts.append("CHAPTER SUMMARIES (scene context):")
        for ch in bundle.chapters:
            head = (
                f"[{ch.chapter_uid}] {ch.parva} Parva, "
                f"chapter {ch.chapter} ({ch.verse_count} verses)"
            )
            parts.append(head)
            # Rewrite the summary's bare "[S11]" verse markers into full
            # citable UIDs ("[B2_C62_S11]") so the model cites real UIDs
            # rather than chapter-local shloka numbers we can't verify.
            parts.append(
                _SHLOKA_MARKER_RE.sub(
                    rf"[{ch.chapter_uid}_S\1]", ch.summary
                )
            )
            parts.append("")

    if bundle.verses:
        parts.append("VERSES (cite these UIDs):")
        for v in bundle.verses:
            parts.append(f"[{v.uid}] {v.translation}")
        parts.append("")

    parts.append(
        "Answer the question using only the above. Remember to cite UIDs "
        f"and to reply {ABSTAIN_TOKEN} if the excerpts are insufficient."
    )
    return "\n".join(parts)


def _extract_cited_uids(answer: str, bundle: ContextBundle) -> list[str]:
    """Return the in-context UIDs the answer actually cited, in order.

    Counts a bracketed citation only if it maps to context we supplied:
    an exact supplied verse UID, a supplied chapter UID, or a verse UID
    whose chapter prefix is one of the supplied chapters (i.e. a verse
    drawn from a chapter summary). A UID the model invented outside the
    supplied context is dropped, so this doubles as a cheap groundedness
    check downstream.
    """
    supplied_verses = {v.uid for v in bundle.verses}
    supplied_chapters = {c.chapter_uid for c in bundle.chapters}
    seen: list[str] = []
    for m in _CITE_RE.finditer(answer):
        uid = m.group(1)
        if uid in seen:
            continue
        chapter_prefix = "_".join(uid.split("_")[:2])  # "B2_C62"
        grounded = (
            uid in supplied_verses
            or uid in supplied_chapters
            or chapter_prefix in supplied_chapters
        )
        if grounded:
            seen.append(uid)
    return seen


class Synthesizer:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str = OLLAMA_URL,
        temperature: float = 0.1,
        stream_to_stderr: bool = True,
    ):
        self.model = model
        self.url = url
        self.temperature = temperature
        self.stream_to_stderr = stream_to_stderr

    def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": True,
            "options": {"temperature": self.temperature},
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
                "Is `ollama serve` running?"
            )
        if self.stream_to_stderr:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return "".join(chunks)

    def answer(self, bundle: ContextBundle) -> SynthesisResult:
        if bundle.is_empty:
            return SynthesisResult(
                query=bundle.query,
                answer=ABSTAIN_TOKEN,
                abstained=True,
                cited_uids=[],
                model=self.model,
                bundle=bundle,
            )
        prompt = build_prompt(bundle)
        raw = self._generate(prompt).strip()
        abstained = ABSTAIN_TOKEN in raw
        cited = _extract_cited_uids(raw, bundle)
        return SynthesisResult(
            query=bundle.query,
            answer=raw,
            abstained=abstained,
            cited_uids=cited,
            model=self.model,
            bundle=bundle,
        )
