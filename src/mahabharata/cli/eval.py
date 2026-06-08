"""CLI wrapper for Layer 2 Phase A eval harness (`mbh-eval`)."""

import sys

from mahabharata.common import paths
from mahabharata.layer2 import eval_harness as lib
from mahabharata.layer2.retriever import Retriever


def main():
    print("Loading Layer 1 artifacts + corpus...", file=sys.stderr)
    retriever = Retriever.from_paths(
        entities_path=paths.ENTITIES_PATH,
        themes_path=paths.THEMES_PATH,
        char_index_path=paths.CHAR_INDEX_PATH,
        group_index_path=paths.GROUP_INDEX_PATH,
        theme_index_path=paths.THEME_INDEX_PATH,
        raw_path=paths.RAW_PATH,
        dense_embeddings_path=paths.DENSE_EMBEDDINGS_PATH,
        dense_uids_path=paths.DENSE_UIDS_PATH,
    )
    print(
        f"  loaded {len(retriever.corpus_by_uid):,} verses, "
        f"{len(retriever.char_index):,} characters, "
        f"{len(retriever.theme_index):,} themes",
        file=sys.stderr,
    )

    print(
        f"Running eval set: {paths.rel(paths.EVAL_SET_PATH)}",
        file=sys.stderr,
    )
    payload = lib.run(
        eval_set_path=paths.EVAL_SET_PATH,
        eval_set_label=paths.rel(paths.EVAL_SET_PATH),
        retriever=retriever,
        latest_path=paths.EVAL_RESULTS_LATEST_PATH,
        previous_path=paths.EVAL_RESULTS_PREVIOUS_PATH,
        markdown_path=paths.EVAL_RESULTS_REPORT_PATH,
    )

    _print_summary(payload)
    print()
    print(f"  json    : {paths.rel(paths.EVAL_RESULTS_LATEST_PATH)}")
    print(f"  markdown: {paths.rel(paths.EVAL_RESULTS_REPORT_PATH)}")
    if paths.EVAL_RESULTS_PREVIOUS_PATH.exists():
        print(
            f"  previous: {paths.rel(paths.EVAL_RESULTS_PREVIOUS_PATH)}"
        )
    if paths.EVAL_RESULTS_BASELINE_PATH.exists():
        print(
            f"  baseline: {paths.rel(paths.EVAL_RESULTS_BASELINE_PATH)} "
            "(manually pinned)"
        )


def _print_summary(payload: dict) -> None:
    agg = payload["aggregates"]
    overall = agg["overall"]
    print()
    print(
        f"Overall: {overall['passed']}/{overall['total']} pass "
        f"({overall['pass_rate']:.1%})"
    )
    print(f"Router accuracy: {agg['router_accuracy']:.1%}")
    print(f"Union-fallback fired: {agg['union_fallbacks']} item(s)")
    print()
    print("By shape:")
    for shape in (
        "structural_uid", "structural_slice", "lexical", "facet", "concept"
    ):
        s = agg["by_shape"].get(shape)
        if not s:
            continue
        print(
            f"  {shape:<18} {s['passed']:>2}/{s['total']:<2} "
            f"({s['pass_rate']:.1%})"
        )
    fm = agg.get("facet_metrics")
    if fm:
        print()
        print("Facet metrics:")
        print(f"  mean recall@10  : {fm['mean_recall_at_10']:.3f}")
        print(f"  mean recall@20  : {fm['mean_recall_at_20']:.3f}")
        print(f"  mean MRR        : {fm['mean_mrr']:.3f}")
        print(f"  facets-match    : {fm['facets_match_rate']:.1%}")
        print(f"  found in full   : {fm['found_in_full_rate']:.1%}")
    lm = agg.get("lexical_metrics")
    if lm:
        print()
        print("Lexical (BM25) metrics:")
        print(f"  mean recall@10  : {lm['mean_recall_at_10']:.3f}")
        print(f"  mean recall@20  : {lm['mean_recall_at_20']:.3f}")
        print(f"  mean MRR        : {lm['mean_mrr']:.3f}")
        print(f"  found in top-K  : {lm['found_in_full_rate']:.1%}")
    cm = agg.get("concept_metrics")
    if cm:
        print()
        print("Concept (dense) metrics:")
        print(f"  mean recall@10  : {cm['mean_recall_at_10']:.3f}")
        print(f"  mean recall@20  : {cm['mean_recall_at_20']:.3f}")
        print(f"  mean MRR        : {cm['mean_mrr']:.3f}")
        print(f"  found in top-K  : {cm['found_in_full_rate']:.1%}")
    fs = agg.get("frozen_subset")
    if fs:
        print()
        print(
            f"Frozen subset: {fs['passed']}/{fs['total']} pass "
            f"({fs['pass_rate']:.1%})"
        )


if __name__ == "__main__":
    main()
