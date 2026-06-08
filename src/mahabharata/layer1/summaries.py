"""Layer 1, Step 6 — Chapter / parva summaries (naive rollup).

Aggregates the per-verse Gemini `Summary` section up to the chapter
level, and the chapter level up to the parva (book) level. Per
architecture-doc Layer 1 Choice 4, this is a **naive rollup, not an
LLM pass**: the chapter "summary" is the verse summaries concatenated
in shloka order. The point of Layer 1 is to materialize the *data*
Layer 2 retrieves; if naive concatenation grounds good retrieval, the
LLM pass was never needed. Upgrade only if eval demands it.

Outputs
-------
`chapter_summaries.jsonl` — one record per chapter (~1,995), schema:

    {
      "chapter_uid": "B6_C27", "book": 6, "chapter": 27,
      "parva": "Bhishma", "verse_count": 79,
      "first_uid": "B6_C27_S1", "last_uid": "B6_C27_S79",
      "characters": [["Krishna", 40], ...],   # (canonical, verses) desc
      "groups":     [["Pandavas", 5], ...],
      "themes":     [["yoga", 25], ...],
      "summary": "[S1] ...\n[S2] ...\n..."
    }

`parva_summaries.json` — one record per book (18). Compact aggregate:
chapter/verse counts, aggregated entity/theme tallies, and the list of
chapter_uids. Deliberately does NOT re-concatenate the full text (that
would duplicate the whole corpus) — a consumer wanting the prose reads
the referenced chapter records.

Verse order
-----------
The summary text is assembled in `(shloka, uid)` order so it reads in
verse sequence. The raw corpus is ~61% non-monotonic in file order
(scraper pagination artifact, see Step 5 notes), so file order would
produce a jumbled digest; shloka order is the readable choice. The 4
`_orphan` verses sit at their (bogus) shloka position — naive rollup
doesn't try to re-thread them.

Validation
----------
The load-bearing parity check: every corpus verse lands in exactly one
chapter, so `Σ chapter.verse_count == total corpus verses`. Per-verse
character/theme tallies are summed from the Step 3/4 outputs, so their
chapter totals reconcile to the Step 5 indexes by construction; the
report surfaces the grand totals for a cross-glance.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from mahabharata.common.corpus_loader import stream_corpus
from mahabharata.common.parvas import parva_name
from mahabharata.common.paths import rel
from mahabharata.common.sections import parse_sections

# Book + chapter prefix of a UID (ignoring shloka + any _orphan suffix).
_BC_RE = re.compile(r"^B(\d+)_C(\d+)_S")


def _bc_from_uid(uid: str) -> tuple[int, int] | None:
    m = _BC_RE.match(uid)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def aggregate_tags(jsonl_path: Path, field_keys: tuple[str, ...]):
    """Sum per-verse tag occurrences up to the chapter level.

    Returns {field_key: {(book, chapter): Counter(canonical -> verses)}}.
    A verse contributes at most 1 to each canonical it carries (the
    Step 3/4 records are already deduplicated per verse), so the counts
    are 'verses in this chapter tagged with X'.
    """
    out = {k: defaultdict(Counter) for k in field_keys}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            bc = _bc_from_uid(d["uid"])
            if bc is None:
                continue
            for k in field_keys:
                for canon in d.get(k, []):
                    out[k][bc][canon] += 1
    return out


def _sorted_pairs(counter: Counter) -> list[list]:
    """Counter -> [[canonical, count], ...] sorted by count desc, name."""
    return [
        [canon, n]
        for canon, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def build_chapter_records(raw_path, char_tags, theme_tags):
    """Stream the corpus once, building one record per (book, chapter)."""
    # (book, chapter) -> list of (shloka, uid, summary_text)
    verses = defaultdict(list)
    empty_summaries = 0
    total = 0
    for rec in stream_corpus(raw_path, verbose=False):
        total += 1
        bc = (rec["book"], rec["chapter"])
        summary = _extract_summary(rec)
        if not summary:
            empty_summaries += 1
        verses[bc].append((rec["shloka"], rec["uid"], summary))

    records = []
    for (book, chapter), items in verses.items():
        items.sort(key=lambda t: (t[0], t[1]))  # shloka, then uid
        summary_text = "\n".join(
            f"[S{shloka}] {summary}"
            for shloka, _uid, summary in items
            if summary
        )
        chars = char_tags["characters"].get((book, chapter), Counter())
        groups = char_tags["groups"].get((book, chapter), Counter())
        themes = theme_tags["themes"].get((book, chapter), Counter())
        records.append({
            "chapter_uid": f"B{book}_C{chapter}",
            "book": book,
            "chapter": chapter,
            "parva": parva_name(book),
            "verse_count": len(items),
            "first_uid": items[0][1],
            "last_uid": items[-1][1],
            "characters": _sorted_pairs(chars),
            "groups": _sorted_pairs(groups),
            "themes": _sorted_pairs(themes),
            "summary": summary_text,
        })

    records.sort(key=lambda r: (r["book"], r["chapter"]))
    return records, total, empty_summaries


def build_parva_records(chapter_records):
    """Roll chapter records up to one compact record per book."""
    by_book = defaultdict(list)
    for r in chapter_records:
        by_book[r["book"]].append(r)

    parvas = []
    for book in sorted(by_book):
        chapters = sorted(by_book[book], key=lambda r: r["chapter"])
        chars, groups, themes = Counter(), Counter(), Counter()
        for r in chapters:
            for canon, n in r["characters"]:
                chars[canon] += n
            for canon, n in r["groups"]:
                groups[canon] += n
            for canon, n in r["themes"]:
                themes[canon] += n
        parvas.append({
            "book": book,
            "parva": parva_name(book),
            "chapter_count": len(chapters),
            "verse_count": sum(r["verse_count"] for r in chapters),
            "characters": _sorted_pairs(chars),
            "groups": _sorted_pairs(groups),
            "themes": _sorted_pairs(themes),
            "chapter_uids": [r["chapter_uid"] for r in chapters],
        })
    return parvas


def _extract_summary(rec: dict) -> str:
    ai_text = rec.get("ai_analysis", "")
    if not ai_text:
        return ""
    return parse_sections(ai_text).get("Summary") or ""


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote → {rel(path)}")


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  wrote → {rel(path)}")


# --- main ---
def run(*, raw_path: Path, verse_chars_path: Path, verse_themes_path: Path,
        chapter_summaries_path: Path, parva_summaries_path: Path,
        report_path: Path):
    for p, who in ((verse_chars_path, "build_verse_characters"),
                   (verse_themes_path, "build_verse_themes")):
        if not p.exists():
            raise SystemExit(f"[ERROR] {rel(p)} missing. Run {who} first.")

    print(f"Aggregating tags from {rel(verse_chars_path)}...")
    char_tags = aggregate_tags(verse_chars_path, ("characters", "groups"))
    print(f"Aggregating tags from {rel(verse_themes_path)}...")
    theme_tags = aggregate_tags(verse_themes_path, ("themes",))

    print("Streaming corpus for verse summaries...")
    chapter_records, total_verses, empty_summaries = build_chapter_records(
        raw_path, char_tags, theme_tags
    )
    parva_records = build_parva_records(chapter_records)

    # Load-bearing parity guard: every verse lands in exactly one chapter.
    rolled = sum(r["verse_count"] for r in chapter_records)
    if rolled != total_verses:
        raise SystemExit(
            f"[ERROR] verse_count parity failed: chapters sum to {rolled:,} "
            f"but corpus streamed {total_verses:,}."
        )
    parva_rolled = sum(p["verse_count"] for p in parva_records)
    if parva_rolled != total_verses:
        raise SystemExit(
            f"[ERROR] parva verse_count parity failed: {parva_rolled:,} "
            f"!= {total_verses:,}."
        )

    chapter_summaries_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(chapter_summaries_path, chapter_records)
    write_json(parva_summaries_path, parva_records)

    write_coverage_report(
        chapter_records=chapter_records,
        parva_records=parva_records,
        total_verses=total_verses,
        empty_summaries=empty_summaries,
        chapter_summaries_path=chapter_summaries_path,
        parva_summaries_path=parva_summaries_path,
        report_path=report_path,
    )


def write_coverage_report(*, chapter_records, parva_records, total_verses,
                          empty_summaries, chapter_summaries_path,
                          parva_summaries_path, report_path):
    n_chapters = len(chapter_records)
    counts = [r["verse_count"] for r in chapter_records]
    avg = sum(counts) / n_chapters if n_chapters else 0
    biggest = max(chapter_records, key=lambda r: r["verse_count"])
    smallest = min(chapter_records, key=lambda r: r["verse_count"])

    lines = []
    lines.append("# Chapter / Parva Summaries — Coverage Report")
    lines.append("")
    lines.append(
        "Auto-generated by `build_summaries` (Layer 1 Step 6). Naive "
        "rollup — chapter summaries are verse `Summary` sections "
        "concatenated in shloka order; no LLM. The load-bearing check "
        "is verse parity: every corpus verse lands in exactly one "
        "chapter, so chapter `verse_count` sums to the corpus total "
        "(the build aborts otherwise)."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total corpus verses: **{total_verses:,}**")
    lines.append(f"- Chapters: **{n_chapters:,}** across "
                 f"{len(parva_records)} parvas")
    lines.append(f"- Verses/chapter: avg **{avg:.1f}**, "
                 f"min {smallest['verse_count']} "
                 f"({smallest['chapter_uid']}), "
                 f"max {biggest['verse_count']} "
                 f"({biggest['chapter_uid']})")
    empty_pct = (empty_summaries / total_verses * 100) if total_verses else 0
    lines.append(f"- Verses with an empty Summary section: "
                 f"**{empty_summaries:,}** ({empty_pct:.2f}%)")
    lines.append("")

    lines.append("## Per-parva rollup")
    lines.append("")
    lines.append("| Book | Parva | Chapters | Verses | Top characters | "
                 "Top themes |")
    lines.append("|---|---|---:|---:|---|---|")
    for p in parva_records:
        top_c = ", ".join(c for c, _ in p["characters"][:3]) or "—"
        top_t = ", ".join(t for t, _ in p["themes"][:3]) or "—"
        lines.append(
            f"| {p['book']} | {p['parva']} | {p['chapter_count']:,} | "
            f"{p['verse_count']:,} | {top_c} | {top_t} |"
        )
    lines.append("")

    lines.append("## Spot-check — first chapter")
    lines.append("")
    first = chapter_records[0]
    lines.append(f"`{first['chapter_uid']}` ({first['parva']} parva), "
                 f"{first['verse_count']} verses. "
                 f"Top characters: "
                 f"{', '.join(c for c, _ in first['characters'][:5]) or '—'}.")
    lines.append("")
    preview = first["summary"][:600]
    lines.append("> " + preview.replace("\n", "\n> "))
    lines.append("")

    lines.append("## Artifact sizes")
    lines.append("")
    lines.append("| File | Bytes |")
    lines.append("|---|---:|")
    for p in (chapter_summaries_path, parva_summaries_path):
        lines.append(f"| {p.name} | {p.stat().st_size:,} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote → {rel(report_path)}")
