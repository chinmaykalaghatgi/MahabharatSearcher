"""CLI wrapper for the Layer 3 synthesis eval (`mbh-eval-synthesis`).

Usage:
    mbh-eval-synthesis            # context-recall + citation-precision
    mbh-eval-synthesis judge      # also run the (circular) faithfulness judge

Runs the concept-shape eval items through the full Layer 3 pipeline
(context assembly + synthesis), scoring how well retrieval localized the
curated answer and whether synthesis produced a grounded answer (or
correctly abstained). Same 3-slot JSON rotation + markdown report as the
Layer 2 harness.
"""

import sys

from mahabharata.common import paths
from mahabharata.layer3 import eval_harness as lib
from mahabharata.layer3.context import ContextBuilder
from mahabharata.layer3.synthesize import Synthesizer


def main():
    use_judge = "judge" in sys.argv[1:]

    print("Loading Layer 1/2 artifacts + corpus + indexes...", file=sys.stderr)
    builder = ContextBuilder.from_paths(
        entities_path=paths.ENTITIES_PATH,
        themes_path=paths.THEMES_PATH,
        char_index_path=paths.CHAR_INDEX_PATH,
        group_index_path=paths.GROUP_INDEX_PATH,
        theme_index_path=paths.THEME_INDEX_PATH,
        raw_path=paths.RAW_PATH,
        dense_embeddings_path=paths.DENSE_EMBEDDINGS_PATH,
        dense_uids_path=paths.DENSE_UIDS_PATH,
        chapter_embeddings_path=paths.CHAPTER_DENSE_EMBEDDINGS_PATH,
        chapter_chunks_path=paths.CHAPTER_DENSE_CHUNKS_PATH,
        chapter_summaries_path=paths.CHAPTER_SUMMARIES_PATH,
    )
    # Quiet during eval — we don't want every token streamed to stderr.
    synth = Synthesizer(stream_to_stderr=False)
    judge = Synthesizer(stream_to_stderr=False) if use_judge else None

    print(
        f"Running synthesis eval (model {synth.model}, "
        f"judge={'on' if use_judge else 'off'})...",
        file=sys.stderr,
    )
    payload = lib.run(
        eval_set_path=paths.EVAL_SET_PATH,
        eval_set_label=paths.rel(paths.EVAL_SET_PATH),
        builder=builder,
        synth=synth,
        judge=judge,
        latest_path=paths.SYNTH_EVAL_LATEST_PATH,
        previous_path=paths.SYNTH_EVAL_PREVIOUS_PATH,
        markdown_path=paths.SYNTH_EVAL_REPORT_PATH,
    )

    _print_summary(payload)
    print()
    print(f"  json    : {paths.rel(paths.SYNTH_EVAL_LATEST_PATH)}")
    print(f"  markdown: {paths.rel(paths.SYNTH_EVAL_REPORT_PATH)}")
    if paths.SYNTH_EVAL_BASELINE_PATH.exists():
        print(
            f"  baseline: {paths.rel(paths.SYNTH_EVAL_BASELINE_PATH)} "
            "(manually pinned)"
        )


def _print_summary(payload: dict) -> None:
    agg = payload["aggregates"]
    print()
    print(
        f"Overall: {agg['passed']}/{agg['total']} pass "
        f"({agg['pass_rate']:.1%})"
    )
    print(f"Abstain rate: {agg['abstain_rate']:.1%}")
    print(
        f"Context recall (combined mean): "
        f"{agg['mean_combined_context_recall']:.3f}"
    )
    print(
        f"  via verses : {agg['mean_verse_context_recall']:.3f}   "
        f"via chapters: {agg['mean_chapter_context_recall']:.3f}"
    )
    print(
        f"Answer-context localized: "
        f"{agg['context_recall_nonzero']}/{agg['total']} items"
    )
    print(f"Citation precision (mean): {agg['mean_citation_precision']:.3f}")
    if "faithfulness_rate" in agg:
        print(
            f"Faithfulness (judged {agg['n_judged']}): "
            f"{agg['faithfulness_rate']:.1%}"
        )
    print()
    print("Pass reasons:")
    for reason, count in sorted(agg["pass_reasons"].items()):
        print(f"  {reason:<24} {count}")


if __name__ == "__main__":
    main()
