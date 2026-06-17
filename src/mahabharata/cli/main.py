"""Unified front door — `mbh`.

One entry point for the whole system. Classifies the query and dispatches
to the right lane: structural / facet / lexical -> direct verse lookup;
concept -> grounded synthesis. Every query is logged for dogfood capture.

Usage:
    mbh "B6_C27_S29"                       # structural lookup
    mbh "Krishna and yoga"                 # facet lookup
    mbh '"iron mace"'                      # lexical (quoted) lookup
    mbh "Why did Arjuna refuse to fight?"  # concept -> synthesis
    mbh                                    # interactive REPL
    mbh lookup "Why did Arjuna ..."        # force lookup (skip synthesis)

REPL meta-commands:
    :flag <note>   mark the last query as a bad/notable result (for the
                   eval-harvest); :bad / :good are aliases with a flag tag
    :q             quit

Argparse-free by project convention. A leading `lookup` token forces the
pure-retrieval path even for concept queries.
"""

import sys

from mahabharata.common import paths
from mahabharata.dogfood import QueryLogger
from mahabharata.orchestrator import Orchestrator, OrchestratorResponse

TRANSLATION_PREVIEW = 200


def main():
    argv = sys.argv[1:]
    force_lookup = bool(argv) and argv[0] == "lookup"
    if force_lookup:
        argv = argv[1:]

    print("Loading the full stack (corpus + indexes + model)...", file=sys.stderr)
    orch = Orchestrator.from_paths()
    logger = QueryLogger(paths.DOGFOOD_LOG_PATH)
    print(
        f"  ready — {len(orch.builder.chapter_summaries):,} chapters, "
        f"model {orch.synth.model}",
        file=sys.stderr,
    )

    query = " ".join(argv).strip()
    if query:
        _handle(orch, logger, query, synthesize=not force_lookup)
    else:
        _repl(orch, logger, default_synthesize=not force_lookup)


def _handle(orch, logger, query, *, synthesize) -> str:
    resp = orch.answer(query, synthesize=synthesize)
    _print_response(resp)
    qid = logger.log_query(
        {
            "query": resp.query,
            "mode": resp.mode,
            "kind": resp.kind,
            "abstained": resp.abstained,
            "n_verses": resp.total,
            "verse_uids": [v.uid for v in resp.verses],
            "chapters": [c.chapter_uid for c in resp.chapters],
            "cited_uids": resp.cited_uids,
            "answer": resp.answer,
        }
    )
    return qid


def _repl(orch, logger, *, default_synthesize):
    print(
        "\nInteractive. Query, or :flag/:bad/:good <note>, :q to quit.",
        file=sys.stderr,
    )
    last_id: str | None = None
    while True:
        try:
            line = input("\nmbh> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            return
        if line.startswith((":flag", ":bad", ":good")):
            _handle_feedback(logger, last_id, line)
            continue
        last_id = _handle(orch, logger, line, synthesize=default_synthesize)


def _handle_feedback(logger, last_id, line):
    cmd, _, note = line.partition(" ")
    if last_id is None:
        print("  (nothing to flag yet)")
        return
    flag = {":bad": "bad", ":good": "good"}.get(cmd, "note")
    logger.log_feedback(last_id, note.strip(), flag=flag)
    print(f"  logged feedback [{flag}] on last query.")


def _print_response(resp: OrchestratorResponse):
    print()
    print(f"query : {resp.query!r}")
    print(f"mode  : {resp.mode}   ({resp.kind})")
    for note in resp.notes:
        print(f"note  : {note}")

    if resp.kind == "lookup":
        print(f"total : {resp.total:,} matching verses")
        _print_verses(resp.verses)
        return

    # synthesis
    if resp.chapters:
        chs = ", ".join(
            f"{c.chapter_uid}({c.score:.2f})" for c in resp.chapters
        )
        print(f"scenes: {chs}")
    print()
    print("=" * 70)
    if resp.abstained:
        print("ANSWER: (abstained — context insufficient)")
    else:
        print("ANSWER:")
    print(resp.answer or "")
    print("=" * 70)
    if resp.cited_uids:
        print(
            f"grounded citations ({len(resp.cited_uids)}): "
            f"{', '.join(resp.cited_uids)}"
        )
    elif not resp.abstained:
        print("grounded citations: NONE — answer may be ungrounded.")


def _print_verses(verses):
    if not verses:
        return
    print(f"showing top {len(verses)}:")
    print()
    for v in verses:
        translation = v.translation.replace("\n", " ").strip()
        if len(translation) > TRANSLATION_PREVIEW:
            translation = translation[: TRANSLATION_PREVIEW - 1] + "…"
        score = f"  ({v.score:.2f})" if v.score is not None else ""
        print(f"  {v.uid}{score}")
        print(f"    {translation}")
        print()


if __name__ == "__main__":
    main()
