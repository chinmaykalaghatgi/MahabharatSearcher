"""Layer 1, Step 4 — Per-verse theme tagging.

Gazetteer-match each verse's Keywords section against themes.json to
produce a per-verse list of canonical themes that the verse touches.
Mirrors build_verse_characters (Step 3) in structure and validation
strategy. See `docs/theoretical_concepts_and_architecture.md` (Step 2
in detail) for why we use a controlled vocabulary of themes + keyword
set-intersection rather than topic modeling.

Re-running is safe: outputs are fully derived from themes.json and the
raw corpus. There is no per-step overrides file — taxonomy corrections
live in themes_overrides.json and flow through Step 2.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from mahabharata.common.corpus_loader import stream_corpus
from mahabharata.common.parvas import BOOK_NAMES
from mahabharata.common.paths import rel
from mahabharata.common.sections import parse_keywords_section


# --- variant map construction with collision detection ---
def build_variant_map(themes):
    """Map each variant -> canonical theme. Raise on collision — two
    themes claiming the same variant is a taxonomy bug."""
    variant_to_canon = {}
    for canon, spec in themes.items():
        variants = list(spec.get("variants", []))
        if canon not in variants:
            variants = [canon] + variants
        for v in variants:
            existing = variant_to_canon.get(v)
            if existing is not None and existing != canon:
                raise SystemExit(
                    f"[ERROR] variant collision: {v!r} claimed by both "
                    f"{existing!r} and {canon!r}. Fix themes.json."
                )
            variant_to_canon[v] = canon
    return variant_to_canon


# --- per-verse tagging ---
def tag_verse(keywords, variant_map):
    """Return themes touched by this verse, deduplicated on canonical
    while preserving first-appearance order from the keyword list."""
    seen = set()
    out = []
    for kw in keywords:
        t = variant_map.get(kw)
        if t is not None and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --- main streaming pass ---
def run(*, raw_path: Path, themes_path: Path,
        verse_themes_path: Path, report_path: Path):
    if not themes_path.exists():
        raise SystemExit(
            f"[ERROR] {rel(themes_path)} not found. "
            f"Run build_themes first."
        )
    with open(themes_path) as f:
        payload = json.load(f)
    themes = payload.get("themes", {})
    if not themes:
        raise SystemExit(f"[ERROR] {themes_path} has no themes.")

    theme_family = {c: spec.get("family", "?") for c, spec in themes.items()}
    themes_by_family = defaultdict(list)
    for c, fam in theme_family.items():
        themes_by_family[fam].append(c)

    variant_map = build_variant_map(themes)
    print(f"  variant map: {len(variant_map):,} variants → "
          f"{len(themes):,} canonical themes across "
          f"{len(themes_by_family):,} families")

    step2_variant_counts = Counter()
    theme_verse_counts = Counter()
    theme_hist = Counter()
    family_verse_counts = Counter()
    book_total = Counter()
    book_with_theme = Counter()

    total_verses = 0
    verses_with_theme = 0

    verse_themes_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {rel(raw_path)}...")
    with open(verse_themes_path, "w") as f_out:
        for d in stream_corpus(raw_path):
            uid = d["uid"]
            book = d.get("book")
            keywords = parse_keywords_section(d["ai_analysis"])
            step2_variant_counts.update(keywords)

            tags = tag_verse(keywords, variant_map)

            record = {"uid": uid, "themes": tags}
            f_out.write(
                json.dumps(record, ensure_ascii=False,
                           separators=(",", ":")) + "\n"
            )

            total_verses += 1
            theme_hist[len(tags)] += 1
            for t in tags:
                theme_verse_counts[t] += 1
            families_here = {theme_family[t] for t in tags}
            for fam in families_here:
                family_verse_counts[fam] += 1

            if tags:
                verses_with_theme += 1

            if book is not None:
                book_total[book] += 1
                if tags:
                    book_with_theme[book] += 1

    print(f"  wrote {total_verses:,} records → {rel(verse_themes_path)}")

    step2_theme_totals = {}
    for canon, spec in themes.items():
        variants = list(spec.get("variants", []))
        if canon not in variants:
            variants = [canon] + variants
        step2_theme_totals[canon] = sum(
            step2_variant_counts.get(v, 0) for v in variants
        )

    write_coverage_report(
        themes=themes,
        theme_family=theme_family,
        themes_by_family=themes_by_family,
        total_verses=total_verses,
        verses_with_theme=verses_with_theme,
        theme_hist=theme_hist,
        theme_verse_counts=theme_verse_counts,
        family_verse_counts=family_verse_counts,
        book_total=book_total,
        book_with_theme=book_with_theme,
        step2_theme_totals=step2_theme_totals,
        report_path=report_path,
    )


def write_coverage_report(*, themes, theme_family, themes_by_family,
                          total_verses, verses_with_theme, theme_hist,
                          theme_verse_counts, family_verse_counts,
                          book_total, book_with_theme, step2_theme_totals,
                          report_path):
    def pct(n):
        return f"{100.0 * n / total_verses:.1f}%"

    lines = []
    lines.append("# Verse Theme Tagging — Coverage Report")
    lines.append("")
    lines.append(
        "Auto-generated by `build_verse_themes`. Each verse in the corpus "
        "is tagged against the theme taxonomy via keyword set-intersection. "
        "See `docs/theoretical_concepts_and_architecture.md` (Step 2 in "
        "detail) for the design."
    )
    lines.append("")

    # ── summary ──
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total verses: **{total_verses:,}**")
    lines.append(
        f"- Verses with ≥1 theme tag: **{verses_with_theme:,}** "
        f"({pct(verses_with_theme)})"
    )
    lines.append(
        f"- Verses with zero theme tags: "
        f"**{total_verses - verses_with_theme:,}** "
        f"({pct(total_verses - verses_with_theme)})"
    )
    lines.append("")

    # ── per-family coverage ──
    lines.append("## Per-family coverage")
    lines.append("")
    lines.append(
        "How many verses touch each theme family. A verse touching "
        "three `spiritual` themes counts once for spiritual. Balance "
        "here is a taxonomy-health signal — if one family is massively "
        "under-represented, it probably means missing themes in that "
        "family. The martial/social/spiritual/cosmological spread is "
        "expected given the corpus is a war epic framed by a "
        "philosophical dialogue."
    )
    lines.append("")
    lines.append("| Family | Themes | Verses | % |")
    lines.append("|---|---:|---:|---:|")
    fam_rows = []
    for fam in themes_by_family:
        cnt = family_verse_counts.get(fam, 0)
        fam_rows.append((cnt, fam, len(themes_by_family[fam])))
    fam_rows.sort(reverse=True)
    for cnt, fam, n_themes in fam_rows:
        lines.append(f"| {fam} | {n_themes} | {cnt:,} | {pct(cnt)} |")
    lines.append("")

    # ── theme tag count histogram ──
    lines.append("## Theme tag count per verse")
    lines.append("")
    lines.append(
        "Very high counts (8+) usually indicate an over-matching "
        "variant — a generic word claimed by some theme that fires "
        "too often. Watch the tail."
    )
    lines.append("")
    lines.append("| Themes per verse | Verses | % |")
    lines.append("|---:|---:|---:|")
    cutoff = 8
    max_n = max(theme_hist) if theme_hist else 0
    for n in range(0, min(max_n + 1, cutoff)):
        cnt = theme_hist.get(n, 0)
        lines.append(f"| {n} | {cnt:,} | {pct(cnt)} |")
    plus = sum(c for n, c in theme_hist.items() if n >= cutoff)
    if plus:
        lines.append(f"| {cutoff}+ | {plus:,} | {pct(plus)} |")
    lines.append("")

    # ── per-theme table grouped by family ──
    lines.append("## Per-theme verses tagged (vs Step 2 mentions)")
    lines.append("")
    lines.append(
        "`verses tagged` = distinct verses where this canonical theme "
        "appears under any variant. `step2 mentions` = total keyword "
        "occurrences summed over all variants (Step 2 parity). `ratio` "
        "= verses / mentions — values below 1.0 mean a verse "
        "double-mentions a theme via multiple variants, which Step 4 "
        "correctly deduplicates. Expected: 0.7–1.0 for multi-variant "
        "themes, ~1.0 for single-variant themes. Themes grouped by "
        "family, sorted by verse count within each family."
    )
    lines.append("")
    for fam in sorted(
        themes_by_family,
        key=lambda f: -family_verse_counts.get(f, 0),
    ):
        lines.append(f"### {fam}")
        lines.append("")
        lines.append(
            "| Theme | Verses tagged | Step 2 mentions | Ratio |"
        )
        lines.append("|---|---:|---:|---:|")
        rows = []
        for canon in themes_by_family[fam]:
            vt = theme_verse_counts.get(canon, 0)
            s2 = step2_theme_totals.get(canon, 0)
            ratio = (vt / s2) if s2 else 0.0
            rows.append((vt, canon, s2, ratio))
        rows.sort(reverse=True)
        for vt, canon, s2, ratio in rows:
            ratio_str = f"{ratio:.2f}" if s2 else "—"
            lines.append(
                f"| {canon} | {vt:,} | {s2:,} | {ratio_str} |"
            )
        lines.append("")

    # ── by-book breakdown ──
    lines.append("## By-book coverage")
    lines.append("")
    lines.append(
        "Per-book percentage of verses that received at least one "
        "theme tag. Contrasts with the entity-facet by-book breakdown "
        "in `verse_characters_coverage_report.md`: philosophical parvas "
        "(Shanti, Anushasana) should have *higher* theme coverage than "
        "entity coverage, because they discuss abstract concepts without "
        "naming characters."
    )
    lines.append("")
    lines.append("| Book | Parva | Total | With theme | % |")
    lines.append("|---:|---|---:|---:|---:|")
    for b in sorted(book_total):
        bt = book_total[b]
        bwt = book_with_theme.get(b, 0)
        bt_pct = f"{100.0 * bwt / bt:.1f}%"
        parva = BOOK_NAMES.get(b, "?")
        lines.append(
            f"| {b} | {parva} | {bt:,} | {bwt:,} | {bt_pct} |"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote → {rel(report_path)}")
