"""Layer 3 — Synthesis eval harness.

Evaluates the synthesis lane on the concept-shape items of the curated
eval set (the reasoning/lookup questions, each with curated
``known_good_uids``). It answers the question the Layer 2 eval cannot:
given that retrieval localized *some* context, did synthesis produce a
grounded answer — and when it couldn't, did it correctly abstain?

Why these metrics (architecture-doc Layer 3 Choice 6)
-----------------------------------------------------
Synthesis has no single gold paragraph, so we decompose quality into
components, two of them reference-based (no model judge needed, so no
circularity) and one optional model-judged:

  - context_recall   Did the assembled context (verse hits + localized
                     chapters) actually contain the curated answer verses?
                     This is the UPSTREAM gate — a low value is a Layer 2 /
                     chapter-index failure, not a synthesis failure. We
                     report it three ways: via verse hits, via localized
                     chapters (a KG verse is "available" if its chapter was
                     localized — the model can read it from the summary),
                     and combined.
  - citation_precision  Of the UIDs the answer cited, how many are
                     genuinely relevant (a curated KG verse, or a chapter
                     that contains a KG verse)? Cheap groundedness signal,
                     reference-based.
  - faithfulness     (optional, model-judged) Does every claim follow from
                     the supplied context? Off by default — the only local
                     model is the synthesizer itself, so judging is
                     circular; enable it knowingly.

Pass criterion — and why it credits correct abstention
------------------------------------------------------
The pass rule encodes the layer-attribution we care about:

  - combined_context_recall == 0  → the context can't answer the question.
    The CORRECT synthesis behavior is to abstain. So pass == abstained.
    A confident answer here ("should_have_abstained") is the real failure
    mode — faithful-to-irrelevant-context, which is how the live Arjuna
    smoke test failed.
  - combined_context_recall  > 0  → the context contains answers. Synthesis
    should use them: pass == (not abstained) AND cited at least one UID.

This cleanly separates "retrieval didn't supply the answer" (a Layer 2
problem the harness still scores fairly, by crediting abstention) from
"retrieval supplied it but synthesis fumbled" (the Layer 3 problem).

Slot rotation mirrors the Layer 2 harness: latest / previous / baseline.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from mahabharata.layer3.context import ContextBuilder, ContextBundle
from mahabharata.layer3.synthesize import (
    ABSTAIN_TOKEN,
    Synthesizer,
    build_prompt,
)

# Only the concept-shape items carry the reasoning/lookup questions with
# curated known_goods that the synthesis lane targets.
EVAL_SHAPE = "concept"

_FAITHFUL_TOKEN = "GROUNDED"
_FAITHFUL_SYSTEM = (
    "You are a strict evaluator. You are given a QUESTION, the CONTEXT "
    "excerpts a system was shown, and the ANSWER it produced. Decide "
    "whether EVERY factual claim in the ANSWER is supported by the "
    "CONTEXT (ignore outside knowledge — only the context counts). "
    f"Reply with exactly one word: {_FAITHFUL_TOKEN} if fully supported, "
    "or UNSUPPORTED if any claim is not backed by the context."
)


def _chapter_prefix(uid: str) -> str:
    """'B6_C24_S4' -> 'B6_C24'; 'B6_C24' -> 'B6_C24'."""
    return "_".join(uid.split("_")[:2])


def load_concept_items(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("query_shape") == EVAL_SHAPE:
                items.append(rec)
    return items


def _score_context_recall(
    known_good: list[str], bundle: ContextBundle
) -> dict:
    """How much of the curated answer the assembled context contained."""
    n = len(known_good)
    verse_uids = {v.uid for v in bundle.verses}
    chapter_uids = {c.chapter_uid for c in bundle.chapters}

    in_verses = {u for u in known_good if u in verse_uids}
    in_chapters = {
        u for u in known_good if _chapter_prefix(u) in chapter_uids
    }
    combined = in_verses | in_chapters

    return {
        "n_known_good": n,
        "verse_context_recall": len(in_verses) / n if n else None,
        "chapter_context_recall": len(in_chapters) / n if n else None,
        "combined_context_recall": len(combined) / n if n else None,
        "kg_in_context": sorted(combined),
        "kg_missing_from_context": sorted(set(known_good) - combined),
    }


def _score_citations(
    cited: list[str], known_good: list[str]
) -> dict:
    """Precision of cited UIDs against the curated answer set.

    A citation is 'relevant' if it is a curated KG verse, or a chapter
    UID that contains a curated KG verse (scene-level citation of the
    right chapter).
    """
    kg_set = set(known_good)
    kg_chapters = {_chapter_prefix(u) for u in known_good}

    def relevant(uid: str) -> bool:
        if uid in kg_set:
            return True
        # A bare chapter UID (no _S) citing the right chapter.
        if "_S" not in uid and uid in kg_chapters:
            return True
        return False

    rel = [u for u in cited if relevant(u)]
    return {
        "n_cited": len(cited),
        "n_cited_relevant": len(rel),
        "citation_precision": len(rel) / len(cited) if cited else None,
        "cited_relevant": rel,
    }


def evaluate_item(
    item: dict,
    builder: ContextBuilder,
    synth: Synthesizer,
    *,
    judge: Synthesizer | None = None,
    n_chapters: int = 3,
    n_verses: int = 8,
) -> dict:
    query = item["query"]
    known_good = list(item.get("known_good_uids", []))

    bundle = builder.build(query, n_chapters=n_chapters, n_verses=n_verses)
    result = synth.answer(bundle)

    rec = _score_context_recall(known_good, bundle)
    cit = _score_citations(result.cited_uids, known_good)

    combined_recall = rec["combined_context_recall"] or 0.0
    abstained = result.abstained

    if combined_recall == 0.0:
        # Context can't answer — correct behavior is to abstain.
        passed = abstained
        pass_reason = "correct_abstain" if abstained else "should_have_abstained"
    else:
        passed = (not abstained) and len(result.cited_uids) > 0
        pass_reason = (
            "grounded_answer"
            if passed
            else ("abstained_despite_context" if abstained else "no_citations")
        )

    out: dict[str, Any] = {
        "id": item["id"],
        "query": query,
        "frozen": bool(item.get("frozen", False)),
        "abstained": abstained,
        "passed": passed,
        "pass_reason": pass_reason,
        "chapters_localized": [c.chapter_uid for c in bundle.chapters],
        "verse_hits": [v.uid for v in bundle.verses],
        "cited_uids": list(result.cited_uids),
        "answer": result.answer,
        **rec,
        **cit,
    }

    if judge is not None:
        out["faithfulness"] = _judge_faithfulness(judge, bundle, result.answer)
    return out


def _judge_faithfulness(
    judge: Synthesizer, bundle: ContextBundle, answer: str
) -> bool | None:
    """LLM-as-judge groundedness check. Circular if judge == synth model."""
    if not answer or ABSTAIN_TOKEN in answer:
        return None
    context = build_prompt(bundle)
    prompt = (
        f"{context}\n\n=== ANSWER TO EVALUATE ===\n{answer}\n\n"
        f"Reply with one word: {_FAITHFUL_TOKEN} or UNSUPPORTED."
    )
    # Reuse the Synthesizer's transport but swap the system prompt by
    # temporarily generating with the judge system. Simplest: piggyback
    # on _generate with a judge-specific prompt that embeds instructions.
    verdict = judge._generate(  # noqa: SLF001 - intentional internal reuse
        f"{_FAITHFUL_SYSTEM}\n\n{prompt}"
    ).strip().upper()
    return _FAITHFUL_TOKEN in verdict


def _mean(values) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _ratio(num: int, denom: int) -> float:
    return num / denom if denom else 0.0


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    passed = sum(1 for r in results if r["passed"])
    reasons: dict[str, int] = {}
    for r in results:
        reasons[r["pass_reason"]] = reasons.get(r["pass_reason"], 0) + 1

    judged = [r for r in results if r.get("faithfulness") is not None]
    out: dict[str, Any] = {
        "total": n,
        "passed": passed,
        "pass_rate": _ratio(passed, n),
        "pass_reasons": reasons,
        "abstain_rate": _ratio(
            sum(1 for r in results if r["abstained"]), n
        ),
        "mean_combined_context_recall": _mean(
            r["combined_context_recall"] for r in results
        ),
        "mean_verse_context_recall": _mean(
            r["verse_context_recall"] for r in results
        ),
        "mean_chapter_context_recall": _mean(
            r["chapter_context_recall"] for r in results
        ),
        "mean_citation_precision": _mean(
            r["citation_precision"] for r in results
        ),
        "context_recall_nonzero": sum(
            1 for r in results if (r["combined_context_recall"] or 0) > 0
        ),
    }
    if judged:
        out["faithfulness_rate"] = _ratio(
            sum(1 for r in judged if r["faithfulness"]), len(judged)
        )
        out["n_judged"] = len(judged)

    frozen = [r for r in results if r.get("frozen")]
    if frozen:
        out["frozen_subset"] = {
            "total": len(frozen),
            "passed": sum(1 for r in frozen if r["passed"]),
            "pass_rate": _ratio(
                sum(1 for r in frozen if r["passed"]), len(frozen)
            ),
        }
    return out


def render_markdown(payload: dict) -> str:
    agg = payload["aggregates"]
    items = payload["items"]
    L: list[str] = []
    L.append("# Layer 3 — Synthesis Eval Run")
    L.append("")
    L.append(f"**Timestamp:** {payload['timestamp']}  ")
    L.append(
        f"**Eval set:** `{payload['eval_set_path']}` "
        f"(concept subset, {payload['eval_set_size']} items)  "
    )
    L.append(f"**Model:** {payload['model']}  ")
    L.append(
        f"**Context:** top-{payload['config']['n_chapters']} chapters + "
        f"top-{payload['config']['n_verses']} verses  "
    )
    if not payload["config"]["judge"]:
        L.append(
            "**Faithfulness:** not judged this run (LLM-judge off — the "
            "only local model is the synthesizer, so judging is circular).  "
        )
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(
        f"- **Overall:** {agg['passed']}/{agg['total']} pass "
        f"({agg['pass_rate']:.1%})"
    )
    L.append(f"- **Abstain rate:** {agg['abstain_rate']:.1%}")
    L.append(
        f"- **Context recall (combined, mean):** "
        f"{agg['mean_combined_context_recall']:.3f} "
        f"— via verses {agg['mean_verse_context_recall']:.3f}, "
        f"via chapters {agg['mean_chapter_context_recall']:.3f}"
    )
    L.append(
        f"- **Items with any answer-context localized:** "
        f"{agg['context_recall_nonzero']}/{agg['total']}"
    )
    L.append(
        f"- **Citation precision (mean):** "
        f"{agg['mean_citation_precision']:.3f}"
    )
    if "faithfulness_rate" in agg:
        L.append(
            f"- **Faithfulness (judged {agg['n_judged']}):** "
            f"{agg['faithfulness_rate']:.1%}"
        )
    L.append("")
    L.append("### Pass-reason breakdown")
    L.append("")
    for reason, count in sorted(agg["pass_reasons"].items()):
        L.append(f"- `{reason}`: {count}")
    L.append("")
    if "frozen_subset" in agg:
        fs = agg["frozen_subset"]
        L.append(
            f"**Frozen subset:** {fs['passed']}/{fs['total']} "
            f"({fs['pass_rate']:.1%})"
        )
        L.append("")

    L.append("## Per-item results")
    L.append("")
    L.append(
        "| ID | Query | Pass | Reason | ctx-recall | cite-prec | "
        "chapters localized |"
    )
    L.append("|---|---|---|---|---:|---:|---|")
    for r in items:
        L.append(
            f"| {r['id']} | {r['query']} | {_pass(r)} | "
            f"{r['pass_reason']} | "
            f"{_num(r['combined_context_recall'])} | "
            f"{_num(r['citation_precision'])} | "
            f"{', '.join(r['chapters_localized'])} |"
        )
    L.append("")
    return "\n".join(L).rstrip() + "\n"


def _pass(r: dict) -> str:
    return "pass" if r["passed"] else "**FAIL**"


def _num(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def rotate_slots(latest_path: Path, previous_path: Path) -> None:
    if latest_path.exists():
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(latest_path), str(previous_path))


def run(
    *,
    eval_set_path: Path,
    builder: ContextBuilder,
    synth: Synthesizer,
    latest_path: Path,
    previous_path: Path,
    markdown_path: Path,
    judge: Synthesizer | None = None,
    n_chapters: int = 3,
    n_verses: int = 8,
    eval_set_label: str | None = None,
    progress: bool = True,
) -> dict:
    items = load_concept_items(eval_set_path)
    results: list[dict] = []
    for i, item in enumerate(items, 1):
        if progress:
            print(
                f"  [{i}/{len(items)}] {item['id']}: {item['query']}",
                file=sys.stderr,
            )
        results.append(
            evaluate_item(
                item,
                builder,
                synth,
                judge=judge,
                n_chapters=n_chapters,
                n_verses=n_verses,
            )
        )
    aggregates = aggregate(results)

    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    payload = {
        "timestamp": timestamp,
        "eval_set_path": eval_set_label or str(eval_set_path),
        "eval_set_size": len(items),
        "model": synth.model,
        "config": {
            "n_chapters": n_chapters,
            "n_verses": n_verses,
            "judge": judge is not None,
        },
        "aggregates": aggregates,
        "items": results,
    }

    latest_path.parent.mkdir(parents=True, exist_ok=True)
    rotate_slots(latest_path, previous_path)
    latest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    markdown_path.write_text(render_markdown(payload))
    return payload
