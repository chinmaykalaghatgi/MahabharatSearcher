"""CLI wrapper for Layer 3 — grounded synthesis (`mbh-ask`).

Usage:
    mbh-ask "Why did Arjuna refuse to fight?"     # one-shot
    mbh-ask                                        # interactive REPL

Layer 3 is the synthesis lane: it localizes the scene (chapter-dense),
gathers quotable verses (verse-dense), and asks a local Ollama model to
answer STRICTLY from that context, citing UIDs. This is the opt-in
reasoning path (architecture-doc Layer 3 Choice 2) — facet / structural
/ lexical lookups belong to `mbh-query`, which never invokes a model.

Argparse-free by project convention; first positional arg is the
question, joined with spaces. No args drops into a REPL (the corpus +
indexes take a beat to load, so the REPL amortizes that).
"""

import sys

from mahabharata.common import paths
from mahabharata.layer3.context import ContextBuilder, ContextBundle
from mahabharata.layer3.synthesize import SynthesisResult, Synthesizer


def main():
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
    synth = Synthesizer()
    print(
        f"  ready — {len(builder.chapter_summaries):,} chapters, "
        f"model {synth.model}",
        file=sys.stderr,
    )

    argv_query = " ".join(sys.argv[1:]).strip()
    if argv_query:
        _run_once(builder, synth, argv_query)
    else:
        _repl(builder, synth)


def _run_once(builder: ContextBuilder, synth: Synthesizer, query: str):
    bundle = builder.build(query)
    _print_context(bundle)
    print("\n--- synthesizing (streaming) ---\n", file=sys.stderr)
    result = synth.answer(bundle)
    _print_result(result)


def _repl(builder: ContextBuilder, synth: Synthesizer):
    print(
        "\nInteractive mode. Enter a question, or blank / Ctrl-D to exit.",
        file=sys.stderr,
    )
    while True:
        try:
            query = input("\nask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            return
        bundle = builder.build(query)
        _print_context(bundle)
        print("\n--- synthesizing (streaming) ---\n", file=sys.stderr)
        result = synth.answer(bundle)
        _print_result(result)


def _print_context(bundle: ContextBundle):
    print()
    print(f"question : {bundle.query!r}")
    chs = ", ".join(
        f"{c.chapter_uid}({c.score:.2f})" for c in bundle.chapters
    )
    print(f"chapters : {chs or '(none)'}")
    vs = ", ".join(
        f"{v.uid}({v.score:.2f})" if v.score is not None else v.uid
        for v in bundle.verses
    )
    print(f"verses   : {vs or '(none)'}")
    for note in bundle.notes:
        print(f"note     : {note}")


def _print_result(result: SynthesisResult):
    print()
    print("=" * 70)
    if result.abstained:
        print("ANSWER: (abstained — context insufficient)")
    else:
        print("ANSWER:")
    print(result.answer)
    print("=" * 70)
    if result.cited_uids:
        print(f"grounded citations ({len(result.cited_uids)}): "
              f"{', '.join(result.cited_uids)}")
    elif not result.abstained:
        print("grounded citations: NONE — answer may be ungrounded.")


if __name__ == "__main__":
    main()
