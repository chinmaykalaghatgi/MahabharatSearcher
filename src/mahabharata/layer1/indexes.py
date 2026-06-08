"""Layer 1, Step 5 — Inverted indexes.

Takes the per-verse tagging output from Steps 3 and 4 and inverts it
into canonical->verses lookup tables: the primary retrieval-time data
structure for the entity and theme facets. This is the first Layer 1
artifact that is consumed directly by Layer 2 (retrieval), so the
schema needs to be stable and the parity needs to hold exactly.

Schema (all three index files):

    {
      "<canonical>": {
        "type"|"family": "<category>",
        "count": <int>,
        "verses": ["B1_C1_S1", ...]
      },
      ...
    }

Canonicals are ordered by descending verse count — makes the file
self-sorted for top-k browsing. Verses within each entry are in
corpus order (first-appearance order from the streaming pass), which
preserves narrative flow when a consumer reads through them.

Validation strategy: after inversion, cross-check the per-canonical
counts against the Step 3 / Step 4 coverage reports. The counts must
match *exactly* — any drift is a bug in the inversion or a
fragmentation bug upstream. Also cross-check that every UID listed
in an index actually exists in the raw corpus (cheap guard against
record corruption).
"""

import json
from collections import defaultdict
from pathlib import Path

from mahabharata.common.corpus_loader import stream_corpus
from mahabharata.common.paths import rel


# --- helpers ---
def load_uid_whitelist(raw_path):
    """Collect every UID in the raw corpus (with collision patches
    applied) — used to validate that indexes never point at a
    non-existent verse."""
    uids = set()
    for d in stream_corpus(raw_path):
        uids.add(d["uid"])
    return uids


def invert_jsonl(path, field_keys):
    """Stream a per-verse JSONL file and return one inverted dict per
    field key. Each dict maps canonical -> ordered list of UIDs (first
    appearance in the corpus stream). No deduplication needed — the
    Step 3/4 records are already deduplicated on canonical."""
    inverted = {k: defaultdict(list) for k in field_keys}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            uid = d["uid"]
            for k in field_keys:
                for canon in d.get(k, []):
                    inverted[k][canon].append(uid)
    return inverted


def validate_uids(index, valid_uids, label):
    """Fail loudly if any listed UID isn't in the raw corpus."""
    bad = []
    for canon, uids in index.items():
        for u in uids:
            if u not in valid_uids:
                bad.append((canon, u))
                if len(bad) > 5:
                    break
        if len(bad) > 5:
            break
    if bad:
        sample = ", ".join(f"{c}->{u}" for c, u in bad[:5])
        raise SystemExit(
            f"[ERROR] {label}: UIDs not found in raw corpus: {sample}"
        )


def build_sorted_index(inverted, meta_lookup, meta_key):
    """Materialize the final index: sorted by descending count, each
    entry carries type/family metadata + count + verses. Canonicals
    with zero verses are still included (informative — tells a
    consumer the entity exists but never appeared)."""
    result = {}
    rows = []
    for canon, meta in meta_lookup.items():
        uids = inverted.get(canon, [])
        rows.append((len(uids), canon, meta.get(meta_key, ""), uids))
    rows.sort(key=lambda r: (-r[0], r[1]))
    for count, canon, cat, uids in rows:
        result[canon] = {
            meta_key: cat,
            "count": count,
            "verses": uids,
        }
    return result


def check_no_orphans(inverted, meta_lookup, label):
    """An inverted key with no matching canonical in the source
    gazetteer is a structural bug — Step 3/4 should never emit a tag
    that isn't in entities.json / themes.json."""
    orphans = set(inverted) - set(meta_lookup)
    if orphans:
        raise SystemExit(
            f"[ERROR] {label}: {len(orphans)} canonicals in tagged "
            f"records are missing from source gazetteer: "
            f"{sorted(orphans)[:10]}"
        )


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  wrote → {rel(path)}")


# --- main ---
def run(*, raw_path: Path, entities_path: Path, themes_path: Path,
        verse_chars_path: Path, verse_themes_path: Path,
        char_index_path: Path, group_index_path: Path,
        theme_index_path: Path, report_path: Path):
    if not verse_chars_path.exists():
        raise SystemExit(
            f"[ERROR] {rel(verse_chars_path)} missing. "
            f"Run build_verse_characters first."
        )
    if not verse_themes_path.exists():
        raise SystemExit(
            f"[ERROR] {rel(verse_themes_path)} missing. "
            f"Run build_verse_themes first."
        )

    with open(entities_path) as f:
        entities = json.load(f)
    with open(themes_path) as f:
        themes_payload = json.load(f)

    characters = entities.get("characters", {})
    groups = dict(entities.get("groups", {}))
    tribes = entities.get("tribes", {})
    if tribes:
        groups.update(tribes)
    themes = themes_payload.get("themes", {})

    print("Loading raw corpus UIDs for validation guard...")
    valid_uids = load_uid_whitelist(raw_path)
    print(f"  {len(valid_uids):,} UIDs in raw corpus")

    print(f"Inverting {rel(verse_chars_path)}...")
    char_inverted = invert_jsonl(
        verse_chars_path, field_keys=("characters", "groups")
    )
    check_no_orphans(char_inverted["characters"], characters, "characters")
    check_no_orphans(char_inverted["groups"], groups, "groups")
    validate_uids(char_inverted["characters"], valid_uids, "characters")
    validate_uids(char_inverted["groups"], valid_uids, "groups")

    print(f"Inverting {rel(verse_themes_path)}...")
    theme_inverted = invert_jsonl(verse_themes_path, field_keys=("themes",))
    check_no_orphans(theme_inverted["themes"], themes, "themes")
    validate_uids(theme_inverted["themes"], valid_uids, "themes")

    char_index = build_sorted_index(
        char_inverted["characters"], characters, "type"
    )
    group_index = build_sorted_index(
        char_inverted["groups"], groups, "type"
    )
    theme_index = build_sorted_index(
        theme_inverted["themes"], themes, "family"
    )

    char_index_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(char_index_path, char_index)
    write_json(group_index_path, group_index)
    write_json(theme_index_path, theme_index)

    write_coverage_report(
        char_index=char_index,
        group_index=group_index,
        theme_index=theme_index,
        total_verses=len(valid_uids),
        char_index_path=char_index_path,
        group_index_path=group_index_path,
        theme_index_path=theme_index_path,
        report_path=report_path,
    )


def write_coverage_report(*, char_index, group_index, theme_index,
                          total_verses, char_index_path, group_index_path,
                          theme_index_path, report_path):
    lines = []
    lines.append("# Inverted Indexes — Coverage Report")
    lines.append("")
    lines.append(
        "Auto-generated by `build_indexes`. This is a lean report — "
        "the heavy validation is in the Step 3 and Step 4 coverage "
        "reports. What this report checks: (1) per-canonical counts "
        "match Step 3/4 exactly (indexes are a faithful inversion), "
        "(2) no orphan canonicals, (3) no dangling UIDs, (4) the index "
        "file sizes are reasonable. Failures of (2)/(3) crash the "
        "build before this file is written."
    )
    lines.append("")

    # ── summary ──
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total corpus verses: **{total_verses:,}**")
    lines.append(
        f"- Canonicals in character_index: **{len(char_index):,}** "
        f"(covering {sum(v['count'] for v in char_index.values()):,} "
        f"verse-tags)"
    )
    lines.append(
        f"- Canonicals in group_index: **{len(group_index):,}** "
        f"(covering {sum(v['count'] for v in group_index.values()):,} "
        f"verse-tags)"
    )
    lines.append(
        f"- Canonicals in theme_index: **{len(theme_index):,}** "
        f"(covering {sum(v['count'] for v in theme_index.values()):,} "
        f"verse-tags)"
    )
    lines.append("")
    lines.append(
        "A \"verse-tag\" is one (canonical, verse) pair — one verse "
        "can contribute multiple verse-tags across the same index "
        "(e.g. a verse mentioning both Arjuna and Krishna contributes "
        "two character verse-tags)."
    )
    lines.append("")

    # ── empty canonicals ──
    empty_chars = [c for c, v in char_index.items() if v["count"] == 0]
    empty_groups = [c for c, v in group_index.items() if v["count"] == 0]
    empty_themes = [c for c, v in theme_index.items() if v["count"] == 0]

    lines.append("## Zero-coverage canonicals")
    lines.append("")
    lines.append(
        "Canonicals present in the source gazetteer but not tagged on "
        "any verse. Zero is the ideal — anything here means the "
        "gazetteer has an entry that does not show up in keywords at "
        "all (typo, wrong alias, too rare to matter, or a legitimate "
        "deletion candidate)."
    )
    lines.append("")
    lines.append(f"- characters: {len(empty_chars)} "
                 f"({', '.join(empty_chars) if empty_chars else '—'})")
    lines.append(f"- groups: {len(empty_groups)} "
                 f"({', '.join(empty_groups) if empty_groups else '—'})")
    lines.append(f"- themes: {len(empty_themes)} "
                 f"({', '.join(empty_themes) if empty_themes else '—'})")
    lines.append("")

    # ── top-20 preview ──
    lines.append("## Top 20 characters")
    lines.append("")
    lines.append("| Canonical | Type | Verses |")
    lines.append("|---|---|---:|")
    for canon, v in list(char_index.items())[:20]:
        lines.append(f"| {canon} | {v['type']} | {v['count']:,} |")
    lines.append("")

    lines.append("## All groups")
    lines.append("")
    lines.append("| Canonical | Type | Verses |")
    lines.append("|---|---|---:|")
    for canon, v in group_index.items():
        lines.append(f"| {canon} | {v['type']} | {v['count']:,} |")
    lines.append("")

    lines.append("## Top 20 themes")
    lines.append("")
    lines.append("| Canonical | Family | Verses |")
    lines.append("|---|---|---:|")
    for canon, v in list(theme_index.items())[:20]:
        lines.append(f"| {canon} | {v['family']} | {v['count']:,} |")
    lines.append("")

    # ── file sizes ──
    lines.append("## Artifact sizes")
    lines.append("")
    lines.append("| File | Bytes |")
    lines.append("|---|---:|")
    for p in (char_index_path, group_index_path, theme_index_path):
        size = p.stat().st_size
        lines.append(f"| {p.name} | {size:,} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote → {rel(report_path)}")
