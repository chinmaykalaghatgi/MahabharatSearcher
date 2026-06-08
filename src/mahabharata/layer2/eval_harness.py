"""Layer 2 Phase A — Eval harness.

Runs the curated `eval_set.jsonl` items through the Phase A retriever
and reports per-item pass/fail plus aggregate recall@k / MRR / router
accuracy. Writes a markdown report and a JSON dump on each run.

Slot rotation
-------------
Three named files in `data/layer2/reports/`:

  - eval_results_latest.json    overwritten each run
  - eval_results_previous.json  rotated from latest before the new write
  - eval_results_baseline.json  manually pinned (cp latest baseline)

Three slots, no growth. Maps to the only two diff questions that come
up in practice for a 30-item suite: "vs last run?" and "vs known-good
baseline?". A history/ directory can be added later without disrupting
the slot mechanism.

Pass criteria by query shape
----------------------------
  - structural_uid    known_good_uids[0] in retriever results
  - structural_slice  total > 0 AND every listed known_good in results
  - facet             every known_good in TOP-K (K=PASS_K, default 10)

Facet metrics
-------------
For facet items with non-empty known_good_uids:

  - recall@10, recall@20  fraction of known_goods in top-K
  - MRR                    1 / rank of first known_good (over top WIDE_K)
  - found_in_full          known_goods present anywhere in full result
                           set (Phase A diagnostic — Phase A has no
                           relevance ranking, results are corpus-ordered,
                           so the difference between recall@K and
                           found_in_full tells us how much load real
                           ranking would carry once Phase C ships)

Frozen subset
-------------
Eval items can carry an optional `frozen: true` field. If any are
present, aggregates include a separate frozen-subset block. No items
are designated frozen yet — promote them by editing `eval_set.jsonl`.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

from mahabharata.layer2.retriever import Retriever


# Top-K used for the binary pass criterion on facet items.
PASS_K = 10
# Wider K used as a recall + MRR diagnostic.
WIDE_K = 20
# Cap on retriever response size when running eval. Large enough to
# hold any realistic facet intersection so found_in_full is meaningful
# even when the known_good lives deep in corpus order.
FULL_LIMIT = 10_000

# Substring used to detect that the retriever fell back to the union of
# matched facets after an empty intersection. Must stay in sync with
# the literal note string in `mahabharata.layer2.retriever`.
_UNION_FALLBACK_MARKER = "Intersection of all matched facets is empty"


def load_eval_set(path: Path) -> list[dict]:
    """Load a JSONL eval set into a list of dicts."""
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def evaluate_item(item: dict, retriever: Retriever) -> dict:
    """Evaluate a single item against the Phase A retriever."""
    query = item["query"]
    shape = item["query_shape"]

    response = retriever.search(query, limit=FULL_LIMIT)
    result_uids = [v.uid for v in response.results]
    plan = response.plan

    union_fallback = any(
        _UNION_FALLBACK_MARKER in n for n in response.notes
    )

    base: dict[str, Any] = {
        "id": item["id"],
        "query": query,
        "query_shape": shape,
        "router_mode": plan.mode,
        "router_correct": plan.mode == shape,
        "total": response.total,
        "notes": response.notes,
        "union_fallback": union_fallback,
        "known_good_uids": list(item.get("known_good_uids", [])),
        "frozen": bool(item.get("frozen", False)),
    }

    if shape == "structural_uid":
        base.update(_score_structural_uid(item, result_uids))
    elif shape == "structural_slice":
        base.update(_score_structural_slice(item, response.total, result_uids))
    elif shape == "facet":
        base.update(_score_facet(item, plan, result_uids, response.total))
    elif shape == "concept":
        base.update(_score_concept(item, result_uids, response.total))
    elif shape == "lexical":
        base.update(_score_lexical(item, result_uids, response.total))
    else:
        base["passed"] = False
        base["missing_uids"] = list(base["known_good_uids"])

    return base


def _score_structural_uid(item: dict, result_uids: list[str]) -> dict:
    known_good = item.get("known_good_uids") or []
    target = known_good[0] if known_good else None
    passed = target is not None and target in result_uids
    return {
        "passed": passed,
        "missing_uids": [] if passed else list(known_good),
    }


def _score_structural_slice(
    item: dict, total: int, result_uids: list[str]
) -> dict:
    known_good = item.get("known_good_uids") or []
    missing = [u for u in known_good if u not in result_uids]
    return {
        "passed": total > 0 and not missing,
        "missing_uids": missing,
    }


def _score_top_k(
    known_good: list[str], result_uids: list[str], total: int
) -> dict:
    """Shared recall@K / MRR / found_in_full computation.

    Used by both facet and concept scoring. The `total` arg is the
    full result-set size (response.total from the retriever) — used
    only when the eval item has no known-goods, in which case we
    fall back to the binary "non-empty" signal.
    """
    if not known_good:
        return {
            "passed": total > 0,
            "missing_uids": [],
            "recall_at_10": None,
            "recall_at_20": None,
            "mrr": None,
            "found_in_full": None,
        }

    top10 = result_uids[:PASS_K]
    top20 = result_uids[:WIDE_K]
    top10_set = set(top10)
    top20_set = set(top20)
    full_set = set(result_uids)

    n = len(known_good)
    recall_at_10 = sum(1 for u in known_good if u in top10_set) / n
    recall_at_20 = sum(1 for u in known_good if u in top20_set) / n
    first_rank = next(
        (i + 1 for i, u in enumerate(top20) if u in known_good), 0
    )
    mrr = 1.0 / first_rank if first_rank > 0 else 0.0
    found_in_full = all(u in full_set for u in known_good)

    return {
        "passed": all(u in top10_set for u in known_good),
        "missing_uids": [u for u in known_good if u not in top10_set],
        "recall_at_10": recall_at_10,
        "recall_at_20": recall_at_20,
        "mrr": mrr,
        "found_in_full": found_in_full,
    }


def _score_facet(
    item: dict, plan, result_uids: list[str], total: int
) -> dict:
    known_good = item.get("known_good_uids") or []
    target_facets = item.get("target_facets") or {}

    facets_resolved = {
        "characters": sorted(plan.characters),
        "groups": sorted(plan.groups),
        "themes": sorted(plan.themes),
    }
    target_sorted = {
        k: sorted(target_facets.get(k, []))
        for k in ("characters", "groups", "themes")
    }
    facets_match_target = facets_resolved == target_sorted

    out: dict[str, Any] = {
        "facets_resolved": facets_resolved,
        "facets_match_target": facets_match_target,
    }
    out.update(_score_top_k(known_good, result_uids, total))
    return out


def _score_concept(
    item: dict, result_uids: list[str], total: int
) -> dict:
    """Concept items reuse the recall metrics but override the pass rule.

    Facet's `all-KGs-in-top-K` criterion (inherited from `_score_top_k`)
    is wrong for paraphrastic queries: a question like "Why should we
    not mourn the dead?" has many equally-valid verses in the corpus,
    and a curator picks 3-8 of them as known_goods. Demanding dense
    return *all* of those exact pins in top-10 underreports its actual
    win on these queries. The concept criterion is `recall@10 > 0` —
    dense found at least one curated answer in its top-10. Recall@K,
    MRR, and found_in_full are still surfaced so the strict-pin signal
    isn't lost; it's just not the pass gate.
    """
    known_good = item.get("known_good_uids") or []
    base = _score_top_k(known_good, result_uids, total)
    if known_good:
        base["passed"] = (base["recall_at_10"] or 0.0) > 0.0
    return base


def _score_lexical(
    item: dict, result_uids: list[str], total: int
) -> dict:
    """Lexical items reuse concept's metrics and pass rule.

    A quoted keyword query has a small set of canonical exact-match
    verses as known_goods; like concept, several are equally valid and
    a curator pins 2-3. Pass criterion is the same `recall@10 > 0` —
    BM25 surfaced at least one curated verse in its top-10. recall@K,
    MRR, and found_in_full are surfaced for the finer ranking signal.
    """
    known_good = item.get("known_good_uids") or []
    base = _score_top_k(known_good, result_uids, total)
    if known_good:
        base["passed"] = (base["recall_at_10"] or 0.0) > 0.0
    return base


def aggregate(item_results: list[dict]) -> dict:
    """Roll up per-item results into summary metrics."""
    by_shape: dict[str, list[dict]] = {}
    for r in item_results:
        by_shape.setdefault(r["query_shape"], []).append(r)

    out: dict[str, Any] = {
        "overall": _shape_stats(item_results),
        "by_shape": {s: _shape_stats(items) for s, items in by_shape.items()},
        "router_accuracy": _ratio(
            sum(1 for r in item_results if r["router_correct"]),
            len(item_results),
        ),
        "union_fallbacks": sum(
            1 for r in item_results if r.get("union_fallback")
        ),
    }

    facet_with_known = [
        r for r in by_shape.get("facet", []) if r["known_good_uids"]
    ]
    if facet_with_known:
        out["facet_metrics"] = {
            "n_items": len(facet_with_known),
            "mean_recall_at_10": _mean(
                r["recall_at_10"] for r in facet_with_known
            ),
            "mean_recall_at_20": _mean(
                r["recall_at_20"] for r in facet_with_known
            ),
            "mean_mrr": _mean(r["mrr"] for r in facet_with_known),
            "facets_match_rate": _ratio(
                sum(1 for r in facet_with_known if r["facets_match_target"]),
                len(facet_with_known),
            ),
            "found_in_full_rate": _ratio(
                sum(1 for r in facet_with_known if r["found_in_full"]),
                len(facet_with_known),
            ),
        }

    concept_with_known = [
        r for r in by_shape.get("concept", []) if r["known_good_uids"]
    ]
    if concept_with_known:
        out["concept_metrics"] = {
            "n_items": len(concept_with_known),
            "mean_recall_at_10": _mean(
                r["recall_at_10"] for r in concept_with_known
            ),
            "mean_recall_at_20": _mean(
                r["recall_at_20"] for r in concept_with_known
            ),
            "mean_mrr": _mean(r["mrr"] for r in concept_with_known),
            "found_in_full_rate": _ratio(
                sum(1 for r in concept_with_known if r["found_in_full"]),
                len(concept_with_known),
            ),
        }

    lexical_with_known = [
        r for r in by_shape.get("lexical", []) if r["known_good_uids"]
    ]
    if lexical_with_known:
        out["lexical_metrics"] = {
            "n_items": len(lexical_with_known),
            "mean_recall_at_10": _mean(
                r["recall_at_10"] for r in lexical_with_known
            ),
            "mean_recall_at_20": _mean(
                r["recall_at_20"] for r in lexical_with_known
            ),
            "mean_mrr": _mean(r["mrr"] for r in lexical_with_known),
            "found_in_full_rate": _ratio(
                sum(1 for r in lexical_with_known if r["found_in_full"]),
                len(lexical_with_known),
            ),
        }

    frozen = [r for r in item_results if r.get("frozen")]
    if frozen:
        out["frozen_subset"] = _shape_stats(frozen)

    return out


def _shape_stats(items: list[dict]) -> dict:
    passed = sum(1 for i in items if i["passed"])
    return {
        "total": len(items),
        "passed": passed,
        "pass_rate": _ratio(passed, len(items)),
    }


def _ratio(num: int, denom: int) -> float:
    return num / denom if denom else 0.0


def _mean(values) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def render_markdown(payload: dict) -> str:
    """Render the run payload as a human-readable markdown report."""
    agg = payload["aggregates"]
    items = payload["items"]
    overall = agg["overall"]

    lines: list[str] = []
    lines.append("# Layer 2 Phase A — Eval Run")
    lines.append("")
    lines.append(f"**Timestamp:** {payload['timestamp']}  ")
    lines.append(
        f"**Eval set:** `{payload['eval_set_path']}` "
        f"({payload['eval_set_size']} items)  "
    )
    lines.append(
        "**Retriever:** Phase A + C — route-by-shape. Facet results are "
        "corpus-ordered; lexical (quoted) queries are ranked by BM25 "
        "(Step 5); concept queries by dense bge-small (Step 6)."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- **Overall:** {overall['passed']}/{overall['total']} pass "
        f"({overall['pass_rate']:.1%})"
    )
    lines.append(
        f"- **Router accuracy:** {agg['router_accuracy']:.1%}"
    )
    lines.append(
        f"- **Union-fallback fired:** {agg['union_fallbacks']} item(s)"
    )
    lines.append("")

    lines.append("### Pass rate by shape")
    lines.append("")
    lines.append("| Shape | Pass | Total | Rate |")
    lines.append("|---|---:|---:|---:|")
    for shape in (
        "structural_uid", "structural_slice", "lexical", "facet", "concept"
    ):
        s = agg["by_shape"].get(shape)
        if not s:
            continue
        lines.append(
            f"| {shape} | {s['passed']} | {s['total']} | "
            f"{s['pass_rate']:.1%} |"
        )
    lines.append("")

    fm = agg.get("facet_metrics")
    if fm:
        lines.append(
            f"### Facet metrics ({fm['n_items']} items with known_goods)"
        )
        lines.append("")
        lines.append(f"- Mean recall@10: {fm['mean_recall_at_10']:.3f}")
        lines.append(f"- Mean recall@20: {fm['mean_recall_at_20']:.3f}")
        lines.append(f"- Mean MRR: {fm['mean_mrr']:.3f}")
        lines.append(
            f"- Facet-resolution accuracy: "
            f"{fm['facets_match_rate']:.1%}"
        )
        lines.append(
            f"- Known-goods reachable in full result set: "
            f"{fm['found_in_full_rate']:.1%}"
        )
        lines.append("")

    lm = agg.get("lexical_metrics")
    if lm:
        lines.append(
            f"### Lexical (BM25) metrics ({lm['n_items']} items "
            "with known_goods)"
        )
        lines.append("")
        lines.append(f"- Mean recall@10: {lm['mean_recall_at_10']:.3f}")
        lines.append(f"- Mean recall@20: {lm['mean_recall_at_20']:.3f}")
        lines.append(f"- Mean MRR: {lm['mean_mrr']:.3f}")
        lines.append(
            f"- Known-goods reachable in BM25 top-K: "
            f"{lm['found_in_full_rate']:.1%}"
        )
        lines.append("")

    cm = agg.get("concept_metrics")
    if cm:
        lines.append(
            f"### Concept (dense) metrics ({cm['n_items']} items "
            "with known_goods)"
        )
        lines.append("")
        lines.append(f"- Mean recall@10: {cm['mean_recall_at_10']:.3f}")
        lines.append(f"- Mean recall@20: {cm['mean_recall_at_20']:.3f}")
        lines.append(f"- Mean MRR: {cm['mean_mrr']:.3f}")
        lines.append(
            f"- Known-goods reachable in dense top-K: "
            f"{cm['found_in_full_rate']:.1%}"
        )
        lines.append("")

    if "frozen_subset" in agg:
        fs = agg["frozen_subset"]
        lines.append("### Frozen subset")
        lines.append("")
        lines.append(
            f"- {fs['passed']}/{fs['total']} pass ({fs['pass_rate']:.1%})"
        )
        lines.append("")

    lines.append("## Per-item results")
    lines.append("")
    _render_uid_table(lines, items)
    _render_slice_table(lines, items)
    _render_lexical_table(lines, items)
    _render_facet_table(lines, items)
    _render_concept_table(lines, items)

    return "\n".join(lines).rstrip() + "\n"


def _render_uid_table(lines: list[str], items: list[dict]) -> None:
    rows = [r for r in items if r["query_shape"] == "structural_uid"]
    if not rows:
        return
    lines.append("### structural_uid")
    lines.append("")
    lines.append("| ID | Query | Pass | Router | Notes |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        notes = _diagnostic_notes(r)
        lines.append(
            f"| {r['id']} | `{r['query']}` | {_pass_label(r)} | "
            f"{r['router_mode']} | {notes} |"
        )
    lines.append("")


def _render_slice_table(lines: list[str], items: list[dict]) -> None:
    rows = [r for r in items if r["query_shape"] == "structural_slice"]
    if not rows:
        return
    lines.append("### structural_slice")
    lines.append("")
    lines.append("| ID | Query | Pass | Total | Router | Notes |")
    lines.append("|---|---|---|---:|---|---|")
    for r in rows:
        notes = _diagnostic_notes(r)
        lines.append(
            f"| {r['id']} | `{r['query']}` | {_pass_label(r)} | "
            f"{r['total']} | {r['router_mode']} | {notes} |"
        )
    lines.append("")


def _render_facet_table(lines: list[str], items: list[dict]) -> None:
    rows = [r for r in items if r["query_shape"] == "facet"]
    if not rows:
        return
    lines.append("### facet")
    lines.append("")
    lines.append(
        "| ID | Query | Pass | r@10 | r@20 | MRR | Total | "
        "Resolved | Notes |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---|---|")
    for r in rows:
        notes = _diagnostic_notes(r)
        resolved = _format_resolved(r.get("facets_resolved", {}))
        lines.append(
            f"| {r['id']} | {r['query']} | {_pass_label(r)} | "
            f"{_fmt_num(r.get('recall_at_10'))} | "
            f"{_fmt_num(r.get('recall_at_20'))} | "
            f"{_fmt_num(r.get('mrr'))} | "
            f"{r['total']} | {resolved} | {notes} |"
        )
    lines.append("")


def _render_lexical_table(lines: list[str], items: list[dict]) -> None:
    rows = [r for r in items if r["query_shape"] == "lexical"]
    if not rows:
        return
    lines.append("### lexical (BM25)")
    lines.append("")
    lines.append(
        "| ID | Query | Pass | r@10 | r@20 | MRR | Total | Notes |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        notes = _diagnostic_notes(r)
        lines.append(
            f"| {r['id']} | {r['query']} | {_pass_label(r)} | "
            f"{_fmt_num(r.get('recall_at_10'))} | "
            f"{_fmt_num(r.get('recall_at_20'))} | "
            f"{_fmt_num(r.get('mrr'))} | "
            f"{r['total']} | {notes} |"
        )
    lines.append("")


def _render_concept_table(lines: list[str], items: list[dict]) -> None:
    rows = [r for r in items if r["query_shape"] == "concept"]
    if not rows:
        return
    lines.append("### concept (dense)")
    lines.append("")
    lines.append(
        "| ID | Query | Pass | r@10 | r@20 | MRR | Total | Notes |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        notes = _diagnostic_notes(r)
        lines.append(
            f"| {r['id']} | {r['query']} | {_pass_label(r)} | "
            f"{_fmt_num(r.get('recall_at_10'))} | "
            f"{_fmt_num(r.get('recall_at_20'))} | "
            f"{_fmt_num(r.get('mrr'))} | "
            f"{r['total']} | {notes} |"
        )
    lines.append("")


def _pass_label(r: dict) -> str:
    return "pass" if r["passed"] else "**FAIL**"


def _fmt_num(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def _format_resolved(facets: dict) -> str:
    parts = []
    for kind, key in (("C", "characters"), ("G", "groups"), ("T", "themes")):
        vs = facets.get(key) or []
        if vs:
            parts.append(f"{kind}:[{', '.join(vs)}]")
    return " ".join(parts) if parts else "—"


def _diagnostic_notes(r: dict) -> str:
    """Compact column for things worth flagging on a row."""
    notes: list[str] = []
    if not r["router_correct"]:
        notes.append(
            f"router→{r['router_mode']} (expected {r['query_shape']})"
        )
    if r.get("union_fallback"):
        notes.append("union-fallback")
    # facets_match_target only meaningful for facet items —
    # concept queries don't trigger gazetteer matches by design.
    if r["query_shape"] == "facet" and r.get("facets_match_target") is False:
        notes.append("facets≠target")
    missing = r.get("missing_uids") or []
    if missing:
        notes.append(f"missing={','.join(missing)}")
    if r["query_shape"] in ("facet", "concept", "lexical"):
        if r.get("found_in_full") is False:
            notes.append("not in full result set")
    return "; ".join(notes) if notes else "—"


def rotate_slots(latest_path: Path, previous_path: Path) -> None:
    """Move the existing latest dump into the previous slot.

    Run before writing the new latest. Idempotent — does nothing on
    the first run when no latest file exists yet.
    """
    if latest_path.exists():
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(latest_path), str(previous_path))


def run(
    *,
    eval_set_path: Path,
    retriever: Retriever,
    latest_path: Path,
    previous_path: Path,
    markdown_path: Path,
    eval_set_label: str | None = None,
) -> dict:
    """Top-level orchestrator. Returns the payload that was written.

    `eval_set_label` is the string written into the report's header
    and JSON metadata. Defaults to the absolute eval set path; the
    CLI passes a project-relative version for cleaner reports.
    """
    items = load_eval_set(eval_set_path)
    results = [evaluate_item(item, retriever) for item in items]
    aggregates = aggregate(results)

    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    payload = {
        "timestamp": timestamp,
        "eval_set_path": eval_set_label or str(eval_set_path),
        "eval_set_size": len(items),
        "config": {
            "pass_k": PASS_K,
            "wide_k": WIDE_K,
            "full_limit": FULL_LIMIT,
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
