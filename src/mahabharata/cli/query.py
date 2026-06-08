"""CLI wrapper for Layer 2 Phase A retrieval.

Usage:
    mbh-query B6_C27_S29            # exact verse
    mbh-query "B6_C27"              # chapter slice
    mbh-query "Krishna dharma"      # facet intersection
    mbh-query                       # interactive REPL

Argparse-free by project convention. The first positional arg (if any)
is treated as the query; everything is joined with spaces so quoting
is optional. Passing no args drops into a REPL, which is the ergonomic
default for interactive research since the corpus load takes a beat.
"""

import sys

from mahabharata.common import paths
from mahabharata.layer2.retriever import RetrievalResponse, Retriever


LIMIT = 10
TRANSLATION_PREVIEW = 220


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

    argv_query = " ".join(sys.argv[1:]).strip()
    if argv_query:
        _run_once(retriever, argv_query)
    else:
        _repl(retriever)


def _run_once(retriever: Retriever, query: str):
    response = retriever.search(query, limit=LIMIT)
    _print_response(response)


def _repl(retriever: Retriever):
    print(
        "\nInteractive mode. Enter a query, or blank line / Ctrl-D to exit.",
        file=sys.stderr,
    )
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            return
        response = retriever.search(query, limit=LIMIT)
        _print_response(response)


def _print_response(response: RetrievalResponse):
    print()
    print(f"query : {response.query!r}")
    print(f"mode  : {response.mode}")
    _print_plan_summary(response)
    print(f"total : {response.total:,} matching verses")
    for note in response.notes:
        print(f"note  : {note}")
    if not response.results:
        return
    shown = len(response.results)
    print(f"showing top {shown} (by corpus order):")
    print()
    for v in response.results:
        translation = v.translation.replace("\n", " ").strip()
        if len(translation) > TRANSLATION_PREVIEW:
            translation = translation[: TRANSLATION_PREVIEW - 1] + "…"
        print(f"  {v.uid}")
        print(f"    {translation}")
        print()


def _print_plan_summary(response: RetrievalResponse):
    plan = response.plan
    parts: list[str] = []
    if plan.characters:
        parts.append(f"characters={plan.characters}")
    if plan.groups:
        parts.append(f"groups={plan.groups}")
    if plan.themes:
        parts.append(f"themes={plan.themes}")
    if plan.uid:
        parts.append(f"uid={plan.uid}")
    if plan.book is not None:
        parts.append(f"book={plan.book}")
    if plan.chapter is not None:
        parts.append(f"chapter={plan.chapter}")
    if parts:
        print(f"plan  : {', '.join(parts)}")


if __name__ == "__main__":
    main()
