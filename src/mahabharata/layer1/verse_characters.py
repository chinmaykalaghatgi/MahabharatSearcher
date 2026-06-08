"""Layer 1, Step 3 — Per-verse character and group tagging.

Gazetteer-match each verse's Keywords section against entities.json
to produce a per-verse list of canonical characters and groups that
appear in the verse. See `docs/theoretical_concepts_and_architecture.md`
(Step 3 in detail) for the rationale — short version: dictionary-based
NER on a closed corpus is high-precision by construction, recall is
bounded by Step 1's gazetteer, and ambiguity was handled upstream.

Re-running is safe: outputs are fully derived from entities.json and
the raw corpus. There is no per-step overrides file — entity
corrections live in entities_overrides.json and flow through.
"""

import json
from collections import Counter
from pathlib import Path

from mahabharata.common.corpus_loader import stream_corpus
from mahabharata.common.parvas import BOOK_NAMES
from mahabharata.common.paths import rel
from mahabharata.common.sections import parse_keywords_section


# --- alias map construction with collision detection ---
def build_alias_map(bucket, bucket_name):
    """Map canonical + each alias -> canonical. Raise on collision."""
    alias_to_canon = {}
    for canon, spec in bucket.items():
        terms = [canon] + list(spec.get("aliases", []))
        for t in terms:
            existing = alias_to_canon.get(t)
            if existing is not None and existing != canon:
                raise SystemExit(
                    f"[ERROR] alias collision in {bucket_name}: "
                    f"{t!r} claimed by both {existing!r} and {canon!r}. "
                    f"Fix entities.json."
                )
            alias_to_canon[t] = canon
    return alias_to_canon


def check_no_cross_map_conflicts(char_map, group_map):
    """A term must not appear in both character and group alias maps."""
    overlaps = set(char_map) & set(group_map)
    if overlaps:
        details = ", ".join(
            f"{t!r} (char={char_map[t]!r}, group={group_map[t]!r})"
            for t in sorted(overlaps)
        )
        raise SystemExit(
            f"[ERROR] terms appear in both character and group maps: "
            f"{details}"
        )


# --- per-verse tagging ---
def tag_verse(keywords, char_map, group_map):
    """Return (characters, groups), deduplicated on canonical while
    preserving first-appearance order from the keyword list."""
    seen_c = set()
    chars = []
    seen_g = set()
    groups = []
    for kw in keywords:
        c = char_map.get(kw)
        if c is not None and c not in seen_c:
            seen_c.add(c)
            chars.append(c)
        g = group_map.get(kw)
        if g is not None and g not in seen_g:
            seen_g.add(g)
            groups.append(g)
    return chars, groups


# --- main streaming pass ---
def run(*, raw_path: Path, entities_path: Path,
        verse_chars_path: Path, report_path: Path):
    if not entities_path.exists():
        raise SystemExit(
            f"[ERROR] {rel(entities_path)} not found. "
            f"Run build_entities first."
        )
    with open(entities_path) as f:
        entities = json.load(f)

    characters = entities.get("characters", {})
    groups = dict(entities.get("groups", {}))
    tribes = entities.get("tribes", {})
    if tribes:
        print(f"  folding {len(tribes)} tribes into groups for tagging")
        groups.update(tribes)

    char_map = build_alias_map(characters, "characters")
    group_map = build_alias_map(groups, "groups")
    check_no_cross_map_conflicts(char_map, group_map)

    print(f"  character alias map: {len(char_map):,} terms → "
          f"{len(characters):,} canonicals")
    print(f"  group alias map:     {len(group_map):,} terms → "
          f"{len(groups):,} canonicals")

    step1_keyword_counts = Counter()
    char_verse_counts = Counter()
    group_verse_counts = Counter()
    char_hist = Counter()
    group_hist = Counter()
    book_total = Counter()
    book_with_char = Counter()
    book_with_group = Counter()

    total_verses = 0
    verses_with_char = 0
    verses_with_group = 0
    verses_with_any = 0
    verses_only_char = 0
    verses_only_group = 0
    verses_both = 0

    verse_chars_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {rel(raw_path)}...")
    with open(verse_chars_path, "w") as f_out:
        for d in stream_corpus(raw_path):
            uid = d["uid"]
            book = d.get("book")
            keywords = parse_keywords_section(d["ai_analysis"])
            step1_keyword_counts.update(keywords)

            chars, grps = tag_verse(keywords, char_map, group_map)

            record = {"uid": uid, "characters": chars, "groups": grps}
            f_out.write(
                json.dumps(record, ensure_ascii=False,
                           separators=(",", ":")) + "\n"
            )

            total_verses += 1
            char_hist[len(chars)] += 1
            group_hist[len(grps)] += 1
            for c in chars:
                char_verse_counts[c] += 1
            for g in grps:
                group_verse_counts[g] += 1

            has_c = bool(chars)
            has_g = bool(grps)
            if has_c:
                verses_with_char += 1
            if has_g:
                verses_with_group += 1
            if has_c or has_g:
                verses_with_any += 1
            if has_c and not has_g:
                verses_only_char += 1
            if has_g and not has_c:
                verses_only_group += 1
            if has_c and has_g:
                verses_both += 1

            if book is not None:
                book_total[book] += 1
                if has_c:
                    book_with_char[book] += 1
                if has_g:
                    book_with_group[book] += 1

    print(f"  wrote {total_verses:,} records → {rel(verse_chars_path)}")

    step1_char_totals = {}
    for canon, spec in characters.items():
        terms = [canon] + list(spec.get("aliases", []))
        step1_char_totals[canon] = sum(
            step1_keyword_counts.get(t, 0) for t in terms
        )
    step1_group_totals = {}
    for canon, spec in groups.items():
        terms = [canon] + list(spec.get("aliases", []))
        step1_group_totals[canon] = sum(
            step1_keyword_counts.get(t, 0) for t in terms
        )

    write_coverage_report(
        characters=characters,
        groups=groups,
        total_verses=total_verses,
        verses_with_char=verses_with_char,
        verses_with_group=verses_with_group,
        verses_with_any=verses_with_any,
        verses_only_char=verses_only_char,
        verses_only_group=verses_only_group,
        verses_both=verses_both,
        char_hist=char_hist,
        group_hist=group_hist,
        char_verse_counts=char_verse_counts,
        group_verse_counts=group_verse_counts,
        book_total=book_total,
        book_with_char=book_with_char,
        book_with_group=book_with_group,
        step1_char_totals=step1_char_totals,
        step1_group_totals=step1_group_totals,
        report_path=report_path,
    )


def write_coverage_report(*, characters, groups, total_verses,
                          verses_with_char, verses_with_group,
                          verses_with_any, verses_only_char,
                          verses_only_group, verses_both,
                          char_hist, group_hist,
                          char_verse_counts, group_verse_counts,
                          book_total, book_with_char, book_with_group,
                          step1_char_totals, step1_group_totals,
                          report_path):
    def pct(n):
        return f"{100.0 * n / total_verses:.1f}%"

    lines = []
    lines.append("# Verse Character Tagging — Coverage Report")
    lines.append("")
    lines.append(
        "Auto-generated by `build_verse_characters`. Each verse in "
        "the corpus is tagged against the entity gazetteer (characters "
        "+ groups) via keyword set-intersection. See "
        "`docs/theoretical_concepts_and_architecture.md` (Step 3 in "
        "detail) for the design."
    )
    lines.append("")

    # ── summary ──
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total verses: **{total_verses:,}**")
    lines.append(
        f"- Verses with ≥1 character tag: **{verses_with_char:,}** "
        f"({pct(verses_with_char)})"
    )
    lines.append(
        f"- Verses with ≥1 group tag: **{verses_with_group:,}** "
        f"({pct(verses_with_group)})"
    )
    lines.append(
        f"- Verses with ≥1 entity tag (char OR group): "
        f"**{verses_with_any:,}** ({pct(verses_with_any)})"
    )
    lines.append(
        f"- Verses with zero entity tags: "
        f"**{total_verses - verses_with_any:,}** "
        f"({pct(total_verses - verses_with_any)})"
    )
    lines.append("")

    # ── tag composition ──
    lines.append("## Tag composition")
    lines.append("")
    lines.append(
        "How many verses carry only character tags, only group tags, "
        "or both. Group-only verses are candidates for group expansion "
        "at query time (e.g. 'Pandavas' without a specific Pandava named)."
    )
    lines.append("")
    lines.append("| Composition | Verses | % |")
    lines.append("|---|---:|---:|")
    lines.append(f"| only characters | {verses_only_char:,} | "
                 f"{pct(verses_only_char)} |")
    lines.append(f"| only groups | {verses_only_group:,} | "
                 f"{pct(verses_only_group)} |")
    lines.append(f"| both | {verses_both:,} | {pct(verses_both)} |")
    none_count = total_verses - verses_with_any
    lines.append(f"| neither | {none_count:,} | {pct(none_count)} |")
    lines.append("")

    # ── character tag count histogram ──
    lines.append("## Character tag count per verse")
    lines.append("")
    lines.append("| Characters per verse | Verses | % |")
    lines.append("|---:|---:|---:|")
    cutoff = 6
    max_n = max(char_hist) if char_hist else 0
    for n in range(0, min(max_n + 1, cutoff)):
        cnt = char_hist.get(n, 0)
        lines.append(f"| {n} | {cnt:,} | {pct(cnt)} |")
    six_plus = sum(c for n, c in char_hist.items() if n >= cutoff)
    if six_plus:
        lines.append(f"| {cutoff}+ | {six_plus:,} | {pct(six_plus)} |")

    # ── group tag count histogram ──
    lines.append("")
    lines.append("## Group tag count per verse")
    lines.append("")
    lines.append("| Groups per verse | Verses | % |")
    lines.append("|---:|---:|---:|")
    g_cutoff = 4
    max_g = max(group_hist) if group_hist else 0
    for n in range(0, min(max_g + 1, g_cutoff)):
        cnt = group_hist.get(n, 0)
        lines.append(f"| {n} | {cnt:,} | {pct(cnt)} |")
    g_plus = sum(c for n, c in group_hist.items() if n >= g_cutoff)
    if g_plus:
        lines.append(f"| {g_cutoff}+ | {g_plus:,} | {pct(g_plus)} |")

    # ── per-character table ──
    lines.append("")
    lines.append("## Per-character verses tagged (vs Step 1 mentions)")
    lines.append("")
    lines.append(
        "`verses tagged` = distinct verses where this canonical appears "
        "under any alias. `step1 mentions` = total keyword occurrences "
        "summed over canonical + aliases (Step 1 parity). `ratio` = "
        "verses / mentions — values below 1.0 mean a verse double-mentions "
        "the character via multiple aliases, which Step 3 correctly "
        "deduplicates. Expected range: 0.7–1.0 for famous multi-alias "
        "characters, ~1.0 for single-alias characters."
    )
    lines.append("")
    lines.append(
        "| Canonical | Type | Verses tagged | Step 1 mentions | Ratio |"
    )
    lines.append("|---|---|---:|---:|---:|")
    rows = []
    for canon in characters:
        vt = char_verse_counts.get(canon, 0)
        s1 = step1_char_totals.get(canon, 0)
        ratio = (vt / s1) if s1 else 0.0
        typ = characters[canon].get("type", "")
        rows.append((vt, canon, typ, s1, ratio))
    rows.sort(reverse=True)
    for vt, canon, typ, s1, ratio in rows:
        ratio_str = f"{ratio:.2f}" if s1 else "—"
        lines.append(
            f"| {canon} | {typ} | {vt:,} | {s1:,} | {ratio_str} |"
        )

    # ── per-group table ──
    lines.append("")
    lines.append("## Per-group verses tagged")
    lines.append("")
    lines.append(
        "| Canonical | Type | Verses tagged | Step 1 mentions |"
    )
    lines.append("|---|---|---:|---:|")
    grows = []
    for canon in groups:
        vt = group_verse_counts.get(canon, 0)
        s1 = step1_group_totals.get(canon, 0)
        typ = groups[canon].get("type", "")
        grows.append((vt, canon, typ, s1))
    grows.sort(reverse=True)
    for vt, canon, typ, s1 in grows:
        lines.append(f"| {canon} | {typ} | {vt:,} | {s1:,} |")

    # ── by-book breakdown ──
    lines.append("")
    lines.append("## By-book coverage")
    lines.append("")
    lines.append(
        "Per-book percentage of verses that received at least one "
        "character or group tag. Low-coverage books are typically "
        "philosophical (Shanti Parva, Anushasana Parva) rather than "
        "narrative — expected, not a bug, but useful to know which "
        "books will lean on the theme facet more than the entity facet "
        "at query time."
    )
    lines.append("")
    lines.append(
        "| Book | Parva | Total | With char | % | With group | % |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for b in sorted(book_total):
        bt = book_total[b]
        bc = book_with_char.get(b, 0)
        bg = book_with_group.get(b, 0)
        bc_pct = f"{100.0 * bc / bt:.1f}%"
        bg_pct = f"{100.0 * bg / bt:.1f}%"
        parva = BOOK_NAMES.get(b, "?")
        lines.append(
            f"| {b} | {parva} | {bt:,} | {bc:,} | {bc_pct} | "
            f"{bg:,} | {bg_pct} |"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote → {rel(report_path)}")
