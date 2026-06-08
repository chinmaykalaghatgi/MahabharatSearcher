"""Layer 1, Step 2 — Build theme taxonomy.

Hand-curated controlled vocabulary of themes organised into 6 families
(ethical, emotional, martial, spiritual, social, cosmological). Same
bootstrap-and-override pattern as Step 1. Re-running is always safe:
overrides are never overwritten.
"""

import json
from collections import Counter
from pathlib import Path

from mahabharata.common.paths import rel
from mahabharata.common.sections import parse_keywords_section


# --- seed data ---
# Hand-curated theme taxonomy grounded in the top uncovered keywords from
# the entity coverage report. Families are organisational buckets only —
# queries hit themes directly, not families. Variants are case-sensitive
# matches against keyword tokens (same rule as entities).
#
# Intentionally conservative on ambiguous terms (e.g. "power", "strength",
# "world", "life"): they get excluded here and surface in the residual
# report so the user can decide per-term via overrides.

THEME_SEEDS = {
    # ── Ethical / moral ────────────────────────────────────────────────────
    "ethical": {
        "dharma":         ["dharma", "Dharma", "righteousness"],
        "adharma":        ["adharma", "unrighteousness"],
        "karma":          ["karma", "action", "actions"],
        "sin":            ["sin", "sins", "evil"],
        "virtue":         ["virtue", "virtues", "virtuous"],
        "truth":          ["truth", "truthfulness"],
        "justice":        ["justice"],
        "duty":           ["duty", "duties"],
        "greed":          ["greed"],
        "forgiveness":    ["forgiveness"],
        "non_violence":   ["non-violence", "ahimsa"],
        "merit":          ["merit"],
    },

    # ── Emotional ──────────────────────────────────────────────────────────
    "emotional": {
        "anger":          ["anger", "wrath", "rage", "fury"],
        "grief":          ["grief", "sorrow", "mourning", "lamentation"],
        "fear":           ["fear", "terror", "dread"],
        "joy":            ["joy", "happiness", "delight"],
        "suffering":      ["suffering", "distress", "anguish",
                           "misery", "pain"],
        "desire":         ["desire", "lust", "craving"],
        "pride":          ["pride", "arrogance"],
        "compassion":     ["compassion", "mercy", "pity", "kindness"],
        "love":           ["love", "affection"],
        "jealousy":       ["jealousy", "envy"],
    },

    # ── Martial ────────────────────────────────────────────────────────────
    "martial": {
        "battle":         ["battle", "battles", "combat", "fight", "fighting",
                           "attack"],
        "war":            ["war", "warfare"],
        "weapons":        ["weapon", "weapons", "mace", "sword"],
        "armor":          ["armor", "armour"],
        "archery":        ["bow", "bows", "arrow", "arrows",
                           "archer", "archers"],
        "chariot_warfare": ["chariot", "chariots", "charioteer"],
        "army":           ["army", "armies", "troops", "forces", "soldiers"],
        "warrior":        ["warrior", "warriors", "hero", "heroes"],
        "elephants":      ["elephant", "elephants"],
        "cavalry":        ["horse", "horses"],
        "victory":        ["victory", "triumph", "conquest"],
        "defeat":         ["defeat"],
        "death":          ["death", "slain", "killed", "kill", "killing"],
        "destruction":    ["destruction", "ruin"],
        "valor":          ["valor", "prowess", "bravery", "courage"],
        "enemy":          ["enemy", "enemies", "foe", "foes"],
        "battlefield":    ["battlefield"],
    },

    # ── Spiritual / devotional ─────────────────────────────────────────────
    "spiritual": {
        "devotion":       ["devotion", "bhakti", "worship"],
        "meditation":     ["meditation", "contemplation"],
        "liberation":     ["liberation", "moksha", "salvation"],
        "yoga":           ["yoga"],
        "renunciation":   ["renunciation"],
        "asceticism":     ["asceticism", "ascetic", "ascetics",
                           "penance", "austerity", "austerities"],
        "sacrifice":      ["sacrifice", "sacrifices", "yajna"],
        "knowledge":      ["knowledge", "learning"],
        "wisdom":         ["wisdom", "insight"],
        "scripture":      ["Vedas", "Veda", "scripture", "scriptures"],
        "brahman_principle": ["Brahman"],  # philosophical absolute; NOT Brahma
        "self_control":   ["self-control", "restraint", "senses"],
        "soul":           ["soul", "atman"],
        "mind_intellect": ["mind", "intellect"],
        "sages":          ["sage", "sages", "rishi", "rishis"],
        "vows":           ["vow", "vows"],
        "purity":         ["purity"],
        "charity":        ["charity", "dana"],
        "detachment":     ["detachment", "vairagya"],
        "humility":       ["humility"],
        "delusion":       ["delusion", "moha"],
        "ignorance":      ["ignorance", "avidya"],
        "refuge":         ["refuge", "sharanam"],
        "hermitage":      ["hermitage", "ashrama"],
        "rebirth":        ["rebirth"],
        "teachers":       ["guru", "teacher", "teachers"],
    },

    # ── Social / political ─────────────────────────────────────────────────
    "social": {
        "kingship":       ["king", "kings", "kingdom", "kingdoms",
                           "sovereignty", "rule"],
        "caste":          ["caste", "varna", "Kshatriya", "Kshatriyas",
                           "Shudra", "Sudra", "Vaishya"],
        # "Brahmana"/"Brahmanas" deliberately omitted from caste — already
        # covered by the "Brahmins" group entity; leave the facet separation
        # clean.
        "family":         ["family", "father", "mother", "son", "sons",
                           "brother", "brothers", "daughter", "relatives"],
        "lineage":        ["lineage", "dynasty", "descendants", "ancestors"],
        "marriage":       ["marriage", "wedding", "bride", "husband", "wife"],
        "counsel":        ["counsel", "advice", "ministers"],
        "honor":          ["honor", "respect"],
        # "artha" folded into wealth — same conceptual space (material
        # prosperity, one of the four purusharthas).
        "wealth":         ["wealth", "riches", "gold", "treasure", "artha"],
        "protection":     ["protection", "protector", "protect"],
        "peace":          ["peace", "tranquility"],
        "prosperity":     ["prosperity", "welfare"],
        "hospitality":    ["hospitality", "guest", "guests"],
        "women":          ["women", "woman"],
        "fame":           ["fame", "glory"],
        "friendship":     ["friendship", "friends", "friend"],
        "assembly":       ["assembly", "sabha"],
    },

    # ── Cosmological / fated ───────────────────────────────────────────────
    "cosmological": {
        "divine_beings":  ["gods", "deities", "Gandharvas", "Gandharva"],
        "demons":         ["demons", "asuras", "rakshasas", "danavas",
                           "Rakshasa", "Rakshasas"],
        "fate":           ["fate", "destiny", "providence"],
        "curse":          ["curse", "curses", "cursed"],
        "boon":           ["boon", "boons", "blessing", "blessings"],
        "heaven":         ["heaven", "heavens", "svarga"],
        "elements":       ["fire", "water", "earth", "wind", "sky", "air"],
        "nature":         ["forest", "forests", "mountain", "mountains",
                           "ocean", "river", "rivers"],
        "celestial_bodies": ["sun", "moon", "stars"],
        "time":           ["time", "age", "yuga"],
        "auspicious":     ["auspicious", "auspiciousness"],
        "divinity":       ["divine", "divinity"],
        "creation":       ["creation"],
    },
}


def count_corpus_keywords(raw_path):
    counter = Counter()
    with open(raw_path) as f:
        for line in f:
            d = json.loads(line)
            counter.update(parse_keywords_section(d["ai_analysis"]))
    return counter


def build_from_seeds():
    """Flatten seeds into themes.json shape: theme -> {family, variants}."""
    themes = {}
    for family, items in THEME_SEEDS.items():
        for theme, variants in items.items():
            themes[theme] = {
                "family": family,
                "variants": list(variants),
            }
    return {"themes": themes}


def load_entity_terms(entities_path):
    """Flat set of every canonical + alias from entities.json, used to
    filter the theme residual analysis so entity tokens don't pollute
    the candidate list."""
    if not entities_path.exists():
        print("  [warning] entities.json not found — residual will "
              "include entity terms")
        return set()
    with open(entities_path) as f:
        ents = json.load(f)
    terms = set()
    for category in ("characters", "groups", "places", "tribes"):
        for canon, spec in ents.get(category, {}).items():
            terms.add(canon)
            terms.update(spec.get("aliases", []))
    return terms


def load_overrides(overrides_path):
    if not overrides_path.exists():
        return None
    with open(overrides_path) as f:
        data = json.load(f)
    if "themes" not in data:
        return None
    return data


def apply_overrides(data, overrides):
    if not overrides or "themes" not in overrides:
        return data
    ops = overrides["themes"]
    bucket = data["themes"]

    for name, spec in ops.get("add", {}).items():
        bucket[name] = {
            "family": spec.get("family", "uncategorized"),
            "variants": list(spec.get("variants", [])),
        }

    for name, mod in ops.get("modify", {}).items():
        if name not in bucket:
            print(f"  [override warning] modify target not found: {name}")
            continue
        entry = bucket[name]
        variants = list(entry.get("variants", []))
        for v in mod.get("add_variants", []):
            if v not in variants:
                variants.append(v)
        for v in mod.get("remove_variants", []):
            if v in variants:
                variants.remove(v)
        entry["variants"] = variants
        if "set_family" in mod:
            entry["family"] = mod["set_family"]

    for spec in ops.get("merge", []):
        into = spec["into"]
        absorb = spec.get("absorb", [])
        if into not in bucket:
            print(f"  [override warning] merge target not found: {into}")
            continue
        target = bucket[into]
        target_variants = list(target.get("variants", []))
        for src in absorb:
            if src in bucket:
                src_entry = bucket.pop(src)
                target_variants.extend(src_entry.get("variants", []))
        seen = set()
        target["variants"] = [v for v in target_variants
                              if not (v in seen or seen.add(v))]

    for name in ops.get("remove", []):
        bucket.pop(name, None)

    return data


def check_variant_conflicts(themes):
    """A variant should map to at most one theme."""
    variant_to_themes = {}
    for theme, spec in themes.items():
        for v in spec.get("variants", []):
            variant_to_themes.setdefault(v, []).append(theme)
    conflicts = {v: ts for v, ts in variant_to_themes.items() if len(ts) > 1}
    if conflicts:
        print("  [warning] variants assigned to multiple themes:")
        for v, ts in sorted(conflicts.items()):
            print(f"    {v!r} -> {ts}")


def write_coverage_report(themes_data, keyword_counts, entity_terms,
                          report_path):
    """Per-theme match counts (grouped by family), family totals, zero-hit
    variants, and a top-N residual of uncovered high-frequency keywords."""
    lines = []
    lines.append("# Themes — Coverage Report")
    lines.append("")
    lines.append(
        "Auto-generated by `build_themes`. This report is the user's "
        "tool for spotting gaps in the seed theme taxonomy. Use it to "
        "decide what to add via `themes_overrides.json`."
    )
    lines.append("")

    themes = themes_data["themes"]

    covered_variants = set()
    for theme, spec in themes.items():
        covered_variants.update(spec.get("variants", []))

    by_family = {}
    for theme, spec in themes.items():
        by_family.setdefault(spec["family"], []).append(theme)

    lines.append("## Per-theme match counts by family")
    lines.append("")

    family_totals = {}
    for family in sorted(by_family):
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| Theme | Total mentions | Top variant hits |")
        lines.append("|---|---:|---|")
        rows = []
        for theme in by_family[family]:
            variants = themes[theme]["variants"]
            hits = [(v, keyword_counts.get(v, 0)) for v in variants]
            total = sum(c for _, c in hits)
            nonzero = sorted(
                [(v, c) for v, c in hits if c > 0],
                key=lambda x: -x[1],
            )
            top_str = ", ".join(f"{v}({c})" for v, c in nonzero[:5]) or "—"
            rows.append((total, theme, top_str))
        rows.sort(reverse=True)
        for total, theme, top_str in rows:
            lines.append(f"| {theme} | {total:,} | {top_str} |")
        family_totals[family] = sum(r[0] for r in rows)
        lines.append("")

    lines.append("## Family totals")
    lines.append("")
    lines.append(
        "A rough balance check. Wildly uneven totals may mean the taxonomy "
        "is under-covering a family (add themes/variants) or that the corpus "
        "really is skewed. Use judgment."
    )
    lines.append("")
    lines.append("| Family | Total mentions | Theme count |")
    lines.append("|---|---:|---:|")
    for family, total in sorted(family_totals.items(), key=lambda x: -x[1]):
        lines.append(f"| {family} | {total:,} | {len(by_family[family])} |")

    lines.append("")
    lines.append("## Zero-hit variants")
    lines.append("")
    lines.append(
        "Variants with zero corpus matches. Usually means a typo or a "
        "concept the corpus spells differently. Fix via overrides or drop."
    )
    lines.append("")
    zero_items = []
    for theme, spec in sorted(themes.items()):
        for v in spec.get("variants", []):
            if keyword_counts.get(v, 0) == 0:
                zero_items.append((theme, v))
    if zero_items:
        for theme, v in zero_items:
            lines.append(f"- `{v}` (in `{theme}`)")
    else:
        lines.append("_(none)_")

    lines.append("")
    lines.append("## Top 150 uncovered high-frequency keywords")
    lines.append("")
    lines.append(
        "High-frequency keywords not covered by any theme variant **and** "
        "not present in `entities.json`. Candidates for extending the "
        "taxonomy — add as themes via `themes_overrides.json` if they look "
        "thematic, route to `entities_overrides.json` if they're proper "
        "nouns we missed, or leave them alone if they're noise."
    )
    lines.append("")
    lines.append("| Keyword | Count |")
    lines.append("|---|---:|")
    uncovered = []
    for kw, c in keyword_counts.most_common(2000):
        if kw in covered_variants or kw in entity_terms:
            continue
        uncovered.append((kw, c))
        if len(uncovered) >= 150:
            break
    for kw, c in uncovered:
        lines.append(f"| {kw} | {c:,} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


STUB_OVERRIDES = {
    "_doc": (
        "Human-edited corrections to the machine-bootstrapped theme taxonomy. "
        "Re-running build_themes is always safe — this file is never "
        "overwritten. Schema below; remove _doc and _example before adding "
        "real entries (or just leave them in, they're ignored)."
    ),
    "_example": {
        "themes": {
            "add": {
                "example_theme": {
                    "family": "ethical",
                    "variants": ["term1", "term2"],
                },
            },
            "remove": ["wrong_theme"],
            "modify": {
                "dharma": {
                    "add_variants": ["new_variant"],
                    "remove_variants": ["wrong_variant"],
                    "set_family": "ethical",
                },
            },
            "merge": [
                {"into": "dharma", "absorb": ["duplicate_theme"]},
            ],
        },
    },
}


def ensure_stub_overrides(overrides_path):
    if overrides_path.exists():
        return
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overrides_path, "w") as f:
        json.dump(STUB_OVERRIDES, f, indent=2, ensure_ascii=False)
    print(f"  wrote stub overrides → {rel(overrides_path)}")


def run(*, raw_path: Path, entities_path: Path, themes_path: Path,
        overrides_path: Path, report_path: Path):
    themes_path.parent.mkdir(parents=True, exist_ok=True)

    print("Counting corpus keywords...")
    keyword_counts = count_corpus_keywords(raw_path)
    print(f"  {len(keyword_counts):,} unique keyword tokens, "
          f"{sum(keyword_counts.values()):,} total")

    print("Loading entity terms (for residual exclusion)...")
    entity_terms = load_entity_terms(entities_path)
    print(f"  {len(entity_terms):,} entity terms loaded")

    print("Building themes from seeds...")
    data = build_from_seeds()
    n_families = len({t["family"] for t in data["themes"].values()})
    print(f"  seeded themes: {len(data['themes'])} across "
          f"{n_families} families")

    print("Loading overrides (if present)...")
    overrides = load_overrides(overrides_path)
    if overrides:
        print("  applying overrides")
        data = apply_overrides(data, overrides)
        print(f"  after overrides — themes: {len(data['themes'])}")
    else:
        print("  no overrides found (will create stub)")
        ensure_stub_overrides(overrides_path)

    check_variant_conflicts(data["themes"])

    with open(themes_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  wrote → {rel(themes_path)}")

    print("Writing coverage report...")
    write_coverage_report(data, keyword_counts, entity_terms, report_path)
    print(f"  wrote → {rel(report_path)}")
