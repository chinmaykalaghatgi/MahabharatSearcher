"""Layer 1, Step 1 — Build canonical entity list.

Bootstraps a curated list of Mahabharata characters, groups, and places,
each with their well-known epithets/aliases. The seed data is hand-curated
knowledge. The `run()` function:

  1. Loads the hardcoded seed entities
  2. Counts how often each alias appears in the corpus's keyword field
     (sanity check — if a seeded entity has zero matches, the seed name
     likely has a typo or the alias isn't used in the Gemini annotations)
  3. Loads entities_overrides.json if it exists, applies it
  4. Writes entities.json (the merged final, used by downstream)
  5. Writes entities_coverage_report.md showing which seeds matched the
     corpus and what high-frequency keywords are NOT yet covered
     (candidates for the user to add via overrides)
  6. Creates a stub entities_overrides.json with schema docs if one
     does not already exist

Re-running is always safe: overrides are never overwritten.
"""

import json
from collections import Counter
from pathlib import Path

from mahabharata.common.paths import rel
from mahabharata.common.sections import parse_keywords_section


# --- seed data ---
# Hand-curated canonical characters with their well-known epithets.
# This list focuses on the major figures of the Mahabharata. Minor figures
# can be added later via the overrides file. Aliases are case-sensitive
# matches against keyword tokens — Gemini's annotations use English/Latin
# transliteration consistently (e.g. "Arjuna", not "अर्जुन").

CHARACTER_SEEDS = {
    # ── Pandavas & immediate family ────────────────────────────────────────
    "Yudhishthira": {
        "aliases": ["Dharmaraja", "Ajatashatru", "Kanka", "Dharmaputra",
                    "Ajatasatru", "Yudhisthira"],
        "type": "Pandava",
    },
    "Bhima": {
        "aliases": ["Bhimasena", "Vrikodara", "Pavanaputra", "Pavanasuta",
                    "Vayuputra", "Vayusuta"],
        "type": "Pandava",
    },
    "Arjuna": {
        "aliases": ["Partha", "Dhananjaya", "Phalguna", "Savyasachi",
                    "Kiritin", "Jishnu", "Gudakesha", "Bibhatsu",
                    "Vijaya", "Kapidhwaja", "Kaunteya", "Parantapa"],
        "type": "Pandava",
    },
    "Nakula": {
        # "Madreya" omitted — shared with Sahadeva and only 2 corpus hits.
        "aliases": [],
        "type": "Pandava",
    },
    "Sahadeva": {
        "aliases": [],
        "type": "Pandava",
    },
    "Kunti": {
        "aliases": ["Pritha", "Prtha"],
        "type": "Pandava-mother",
    },
    "Madri": {
        "aliases": [],
        "type": "Pandava-mother",
    },
    "Pandu": {
        "aliases": [],
        "type": "Pandava-father",
    },
    "Draupadi": {
        "aliases": ["Krishnaa", "Panchali", "Yajnaseni", "Sairandhri",
                    "Drupadi"],
        "type": "Pandava-wife",
    },
    "Subhadra": {
        "aliases": [],
        "type": "Pandava-wife",
    },
    "Hidimba": {
        "aliases": ["Hidimbi"],
        "type": "Pandava-wife",
    },
    "Abhimanyu": {
        "aliases": ["Saubhadra"],
        "type": "Pandava-son",
    },
    "Ghatotkacha": {
        "aliases": [],
        "type": "Pandava-son",
    },
    "Parikshit": {
        "aliases": [],
        "type": "Pandava-descendant",
    },

    # ── Kauravas & their side ──────────────────────────────────────────────
    "Dhritarashtra": {
        "aliases": ["Dhrtarashtra"],
        "type": "Kaurava-father",
    },
    "Gandhari": {
        "aliases": [],
        "type": "Kaurava-mother",
    },
    "Duryodhana": {
        "aliases": ["Suyodhana", "Kuruvardhana"],
        "type": "Kaurava",
    },
    "Dushasana": {
        "aliases": ["Duhshasana", "Dussasana"],
        "type": "Kaurava",
    },
    "Vikarna": {
        "aliases": [],
        "type": "Kaurava",
    },
    "Yuyutsu": {
        "aliases": [],
        "type": "Kaurava-defector",
    },
    "Vidura": {
        "aliases": ["Kshatta", "Kshattri"],
        "type": "Kaurava-uncle",
    },

    # ── Elders / preceptors ────────────────────────────────────────────────
    "Bhishma": {
        "aliases": ["Devavrata", "Gangaputra", "Pitamaha", "Shantanava",
                    "Bhisma", "Gangeya"],
        "type": "elder",
    },
    "Shantanu": {
        "aliases": ["Santanu"],
        "type": "ancestor",
    },
    "Satyavati": {
        "aliases": [],
        "type": "ancestor",
    },
    "Drona": {
        "aliases": ["Dronacharya", "Bharadwaja-putra"],
        "type": "preceptor",
    },
    "Kripa": {
        "aliases": ["Kripacharya", "Sharadvata", "Saradvata"],
        "type": "preceptor",
    },
    "Ashwatthama": {
        "aliases": ["Asvatthama", "Ashvatthama", "Drauni"],
        "type": "Kaurava-ally",
    },

    # ── Karna and Kaurava allies ───────────────────────────────────────────
    "Karna": {
        # "Vrisha" omitted — literally "bull", used for many figures.
        "aliases": ["Vasusena", "Radheya", "Suryaputra"],
        "type": "Kaurava-ally",
    },
    "Shakuni": {
        "aliases": ["Saubala", "Gandhararaja"],
        "type": "Kaurava-ally",
    },
    "Jayadratha": {
        "aliases": ["Saindhava", "Sindhuraja"],
        "type": "Kaurava-ally",
    },
    "Bhagadatta": {
        "aliases": [],
        "type": "Kaurava-ally",
    },
    "Bhurishravas": {
        "aliases": ["Bhurisravas"],
        "type": "Kaurava-ally",
    },
    "Shalya": {
        "aliases": ["Salya", "Madra-raja", "Madraraja"],
        "type": "Kaurava-ally",
    },
    "Kritavarma": {
        "aliases": ["Krtavarma", "Kritavarman"],
        "type": "Kaurava-ally",
    },

    # ── Krishna's circle ───────────────────────────────────────────────────
    "Krishna": {
        "aliases": ["Vasudeva", "Govinda", "Janardana", "Madhava",
                    "Keshava", "Hrishikesha", "Madhusudana", "Achyuta",
                    "Damodara", "Yadava", "Devakiputra", "Krsna",
                    "Varshneya", "Mukunda"],
        "type": "deity-incarnate",
    },
    "Balarama": {
        # "Rama" deliberately omitted — too ambiguous (Parashurama,
        # Ramayana's Rama, generic), would over-claim mentions.
        "aliases": ["Baladeva", "Sankarshana", "Halayudha", "Rauhineya"],
        "type": "Yadava",
    },
    "Satyaki": {
        "aliases": ["Yuyudhana", "Saineya"],
        "type": "Yadava",
    },
    "Pradyumna": {
        "aliases": [],
        "type": "Yadava",
    },
    "Aniruddha": {
        "aliases": [],
        "type": "Yadava",
    },

    # ── Drupada's family (Panchala) ────────────────────────────────────────
    "Drupada": {
        "aliases": ["Yajnasena", "Panchala-raja"],
        "type": "Panchala",
    },
    "Dhrishtadyumna": {
        "aliases": ["Drishtadyumna", "Drstadyumna", "Dhrstadyumna"],
        "type": "Panchala",
    },
    "Shikhandi": {
        "aliases": ["Sikhandin", "Shikhandin"],
        "type": "Panchala",
    },

    # ── Virata's family ────────────────────────────────────────────────────
    "Virata": {
        "aliases": ["Matsya-raja", "Matsyaraja"],
        "type": "Matsya",
    },
    "Uttara": {
        "aliases": [],
        "type": "Matsya",
    },

    # ── Sages & seers ──────────────────────────────────────────────────────
    "Vyasa": {
        "aliases": ["Vedavyasa", "Krishna-Dvaipayana", "Dvaipayana",
                    "Parasharya"],
        "type": "sage",
    },
    "Vasishtha": {
        "aliases": ["Vasistha"],
        "type": "sage",
    },
    "Vishvamitra": {
        "aliases": ["Visvamitra", "Kaushika"],
        "type": "sage",
    },
    "Narada": {
        "aliases": [],
        "type": "sage",
    },
    "Markandeya": {
        "aliases": [],
        "type": "sage",
    },
    "Bharadwaja": {
        "aliases": ["Bharadvaja"],
        "type": "sage",
    },
    "Agastya": {
        "aliases": [],
        "type": "sage",
    },
    "Atri": {
        "aliases": [],
        "type": "sage",
    },
    "Bhrigu": {
        "aliases": ["Bhrgu"],
        "type": "sage",
    },
    "Parashurama": {
        "aliases": ["Parasurama", "Bhargava", "Jamadagnya", "Rama-Jamadagnya"],
        "type": "sage",
    },
    "Durvasa": {
        "aliases": ["Durvasas"],
        "type": "sage",
    },
    "Galava": {
        "aliases": [],
        "type": "sage",
    },
    "Uttanka": {
        "aliases": [],
        "type": "sage",
    },
    "Shaunaka": {
        "aliases": ["Saunaka"],
        "type": "sage",
    },
    "Ugrashrava": {
        "aliases": ["Sauti", "Lomaharshana-putra"],
        "type": "narrator",
    },
    "Sanjaya": {
        "aliases": [],
        "type": "narrator",
    },

    # ── Other notable figures ──────────────────────────────────────────────
    "Ekalavya": {
        "aliases": [],
        "type": "warrior",
    },
    "Janamejaya": {
        "aliases": [],
        "type": "Pandava-descendant",
    },
    "Babhruvahana": {
        "aliases": [],
        "type": "Pandava-son",
    },
    "Iravan": {
        "aliases": ["Iravat"],
        "type": "Pandava-son",
    },

    # ── Deities ────────────────────────────────────────────────────────────
    # Note: Krishna and Vishnu/Narayana are kept distinct from each other —
    # the text frequently treats them as one, but for retrieval we want to
    # separate verses about Krishna-as-character from cosmological references
    # to Vishnu/Narayana as the supreme deity.
    "Indra": {
        "aliases": ["Sachipati", "Devendra", "Shakra", "Sakra",
                    "Purandara", "Vasava", "Maghavan", "Maghavat"],
        "type": "deity",
    },
    "Agni": {
        "aliases": ["Pavaka", "Hutashana", "Vahni", "Anala", "Jvalanа"],
        "type": "deity",
    },
    "Yama": {
        "aliases": ["Antaka", "Kala", "Mrityu", "Vaivasvata"],
        "type": "deity",
    },
    "Vayu": {
        "aliases": ["Pavana", "Anila", "Maruta", "Marut"],
        "type": "deity",
    },
    "Varuna": {
        "aliases": [],
        "type": "deity",
    },
    "Surya": {
        "aliases": ["Aditya", "Bhaskara", "Ravi", "Bhanu", "Tapana",
                    "Vivasvan", "Vivasvat"],
        "type": "deity",
    },
    "Chandra": {
        "aliases": ["Soma", "Indu", "Shashin"],
        "type": "deity",
    },
    "Brahma": {
        # "Brahman" deliberately omitted — refers to the philosophical
        # supreme principle, not the deity Brahma.
        # "Pitamaha" omitted — overlaps with Bhishma's far more common
        # use of the same epithet.
        "aliases": ["Svayambhu", "Dhata", "Lokesha", "Vidhi"],
        "type": "deity",
    },
    "Vishnu": {
        "aliases": ["Narayana", "Hari"],
        "type": "deity",
    },
    "Shiva": {
        "aliases": ["Mahadeva", "Rudra", "Hara", "Shankara", "Sankara",
                    "Ishana", "Pashupati", "Mahesha", "Bhava", "Sthanu",
                    "Tryambaka", "Girisha", "Nilakantha"],
        "type": "deity",
    },
    "Parvati": {
        "aliases": ["Uma", "Gauri", "Devi", "Haimavati"],
        "type": "deity",
    },
    "Lakshmi": {
        "aliases": ["Sri", "Shri"],
        "type": "deity",
    },
    "Saraswati": {
        "aliases": ["Sarasvati"],
        "type": "deity",
    },
    "Ganga": {
        "aliases": ["Bhagirathi", "Jahnavi"],
        "type": "deity",
    },
    "Kubera": {
        "aliases": ["Vaishravana", "Vaisravana", "Dhanada"],
        "type": "deity",
    },
    "Kartikeya": {
        "aliases": ["Skanda", "Kumara", "Subrahmanya", "Guha"],
        "type": "deity",
    },
}


GROUP_SEEDS = {
    # Collective references that the corpus uses heavily but which aren't
    # individuals. Each group lists known canonical members where applicable;
    # downstream tagging can expand a group to its members at query time.
    "Pandavas": {
        "aliases": ["Pandava"],
        "type": "dynasty",
        "members": ["Yudhishthira", "Bhima", "Arjuna",
                    "Nakula", "Sahadeva"],
    },
    "Kauravas": {
        "aliases": ["Kaurava"],
        "type": "dynasty",
        # The 100 sons of Dhritarashtra; canonical members are the few
        # named ones — the rest stay implicit.
        "members": ["Duryodhana", "Dushasana", "Vikarna", "Yuyutsu"],
    },
    "Brahmins": {
        "aliases": ["Brahmin", "Brahmana", "Brahmanas"],
        "type": "varna",
        "members": [],
    },
    "Vasus": {
        # The eight Vasus, attendant deities. Bhishma is the incarnated
        # eighth Vasu (Dyaus).
        "aliases": [],
        "type": "deity-group",
        "members": [],
    },
}


PLACE_SEEDS = {
    "Hastinapura": {
        "aliases": ["Hastinapur", "Nagasahvaya", "Gajasahvaya"],
        "type": "city",
    },
    "Indraprastha": {
        "aliases": ["Khandavaprastha"],
        "type": "city",
    },
    "Kurukshetra": {
        "aliases": ["Samantapanchaka", "Dharmakshetra"],
        "type": "battlefield",
    },
    "Dvaraka": {
        "aliases": ["Dwaraka", "Dwarka", "Dwarika"],
        "type": "city",
    },
    "Mathura": {
        "aliases": [],
        "type": "city",
    },
    "Naimisha": {
        "aliases": ["Naimisharanya"],
        "type": "forest",
    },
    "Kamyaka": {
        "aliases": [],
        "type": "forest",
    },
    "Dvaitavana": {
        "aliases": ["Dwaita-vana"],
        "type": "forest",
    },
    "Khandava": {
        "aliases": ["Khandava-vana"],
        "type": "forest",
    },
    "Panchala": {
        "aliases": ["Pancala"],
        "type": "kingdom",
    },
    "Magadha": {
        "aliases": [],
        "type": "kingdom",
    },
    "Anga": {
        "aliases": [],
        "type": "kingdom",
    },
    "Kalinga": {
        "aliases": [],
        "type": "kingdom",
    },
    "Sindhu": {
        "aliases": [],
        "type": "kingdom",
    },
    "Kamboja": {
        "aliases": [],
        "type": "kingdom",
    },
    "Gandhara": {
        "aliases": [],
        "type": "kingdom",
    },
    "Madra": {
        "aliases": [],
        "type": "kingdom",
    },
    "Kashi": {
        "aliases": ["Varanasi"],
        "type": "kingdom",
    },
    "Kosala": {
        "aliases": [],
        "type": "kingdom",
    },
    "Videha": {
        "aliases": [],
        "type": "kingdom",
    },
    "Matsya": {
        "aliases": [],
        "type": "kingdom",
    },
    "Saurashtra": {
        "aliases": [],
        "type": "kingdom",
    },
    "Kuru": {
        "aliases": ["Kurus"],
        "type": "kingdom",
    },
    "Yamuna": {
        "aliases": [],
        "type": "river",
    },
}


# --- corpus keyword counting ---
def count_corpus_keywords(raw_path):
    """One-pass scan of the corpus, returning Counter of keyword tokens."""
    counter = Counter()
    with open(raw_path) as f:
        for line in f:
            d = json.loads(line)
            counter.update(parse_keywords_section(d["ai_analysis"]))
    return counter


# --- seed -> entities builder ---
def build_from_seeds():
    """Convert the hardcoded seed dicts into the entities.json shape."""
    return {
        "characters": {
            name: dict(spec) for name, spec in CHARACTER_SEEDS.items()
        },
        "groups": {
            name: dict(spec) for name, spec in GROUP_SEEDS.items()
        },
        "places": {
            name: dict(spec) for name, spec in PLACE_SEEDS.items()
        },
    }


# --- override application ---
def load_overrides(overrides_path):
    if not overrides_path.exists():
        return None
    with open(overrides_path) as f:
        data = json.load(f)
    valid_keys = ("characters", "groups", "places", "tribes")
    if not any(k in data for k in valid_keys):
        return None
    return data


def apply_overrides(entities, overrides):
    """Apply add/remove/modify/merge ops from an overrides dict in place."""
    if not overrides:
        return entities

    for category in ("characters", "groups", "places", "tribes"):
        if category not in overrides:
            continue
        ops = overrides[category]
        bucket = entities.setdefault(category, {})

        for name, spec in ops.get("add", {}).items():
            bucket[name] = dict(spec)

        for name, mod in ops.get("modify", {}).items():
            if name not in bucket:
                print(f"  [override warning] modify target not found: {name}")
                continue
            entry = bucket[name]
            aliases = list(entry.get("aliases", []))
            for a in mod.get("add_aliases", []):
                if a not in aliases:
                    aliases.append(a)
            for a in mod.get("remove_aliases", []):
                if a in aliases:
                    aliases.remove(a)
            entry["aliases"] = aliases
            if "set_type" in mod:
                entry["type"] = mod["set_type"]

        # merge: absorb a list of canonical names into a target canonical
        for spec in ops.get("merge", []):
            into = spec["into"]
            absorb = spec.get("absorb", [])
            if into not in bucket:
                print(f"  [override warning] merge target not found: {into}")
                continue
            target = bucket[into]
            target_aliases = list(target.get("aliases", []))
            for src in absorb:
                if src in bucket:
                    src_entry = bucket.pop(src)
                    target_aliases.append(src)
                    target_aliases.extend(src_entry.get("aliases", []))
            seen = set()
            target["aliases"] = [a for a in target_aliases
                                 if not (a in seen or seen.add(a))]

        for name in ops.get("remove", []):
            bucket.pop(name, None)

    return entities


# --- coverage report ---
def write_coverage_report(entities, keyword_counts, report_path):
    """Show which seeds matched the corpus and what high-frequency keywords
    are NOT yet covered by any seed (candidates for the user to add)."""
    lines = []
    lines.append("# Entities — Coverage Report")
    lines.append("")
    lines.append(
        "Auto-generated by `build_entities`. This report is the user's "
        "tool for spotting gaps in the seed entity list. Use it to decide "
        "what to add via `entities_overrides.json`."
    )
    lines.append("")

    covered_terms = set()
    for category in ("characters", "groups", "places", "tribes"):
        for canon, spec in entities.get(category, {}).items():
            covered_terms.add(canon)
            for alias in spec.get("aliases", []):
                covered_terms.add(alias)

    # ── per-character match counts ─────────────────────────────────────────
    lines.append("## Character match counts (canonical + aliases)")
    lines.append("")
    lines.append("| Canonical | Type | Total mentions | Top alias hits |")
    lines.append("|---|---|---:|---|")

    char_rows = []
    for canon, spec in entities["characters"].items():
        terms = [canon] + spec.get("aliases", [])
        hits = [(t, keyword_counts.get(t, 0)) for t in terms]
        total = sum(c for _, c in hits)
        nonzero = sorted(
            [(t, c) for t, c in hits if c > 0],
            key=lambda x: -x[1],
        )
        top_str = ", ".join(f"{t}({c})" for t, c in nonzero[:5]) or "—"
        char_rows.append((total, canon, spec["type"], top_str))

    char_rows.sort(reverse=True)
    for total, canon, typ, top_str in char_rows:
        lines.append(f"| {canon} | {typ} | {total:,} | {top_str} |")

    zero_chars = [r for r in char_rows if r[0] == 0]
    if zero_chars:
        lines.append("")
        lines.append("### ⚠️  Seeded characters with ZERO matches")
        lines.append(
            "These were in the seed list but no alias was found in the "
            "corpus keywords. Likely a spelling/transliteration mismatch — "
            "either fix the alias or add the corpus's preferred spelling "
            "as an alias via overrides."
        )
        lines.append("")
        for _, canon, typ, _ in zero_chars:
            lines.append(f"- `{canon}` ({typ})")

    # ── group match counts ────────────────────────────────────────────────
    lines.append("")
    lines.append("## Group match counts")
    lines.append("")
    lines.append("| Canonical | Type | Total mentions | Top alias hits |")
    lines.append("|---|---|---:|---|")

    group_rows = []
    for canon, spec in entities.get("groups", {}).items():
        terms = [canon] + spec.get("aliases", [])
        hits = [(t, keyword_counts.get(t, 0)) for t in terms]
        total = sum(c for _, c in hits)
        nonzero = sorted(
            [(t, c) for t, c in hits if c > 0],
            key=lambda x: -x[1],
        )
        top_str = ", ".join(f"{t}({c})" for t, c in nonzero[:5]) or "—"
        group_rows.append((total, canon, spec["type"], top_str))

    group_rows.sort(reverse=True)
    for total, canon, typ, top_str in group_rows:
        lines.append(f"| {canon} | {typ} | {total:,} | {top_str} |")

    # ── place match counts ────────────────────────────────────────────────
    lines.append("")
    lines.append("## Place match counts")
    lines.append("")
    lines.append("| Canonical | Type | Total mentions | Top alias hits |")
    lines.append("|---|---|---:|---|")

    place_rows = []
    for canon, spec in entities["places"].items():
        terms = [canon] + spec.get("aliases", [])
        hits = [(t, keyword_counts.get(t, 0)) for t in terms]
        total = sum(c for _, c in hits)
        nonzero = sorted(
            [(t, c) for t, c in hits if c > 0],
            key=lambda x: -x[1],
        )
        top_str = ", ".join(f"{t}({c})" for t, c in nonzero[:5]) or "—"
        place_rows.append((total, canon, spec["type"], top_str))

    place_rows.sort(reverse=True)
    for total, canon, typ, top_str in place_rows:
        lines.append(f"| {canon} | {typ} | {total:,} | {top_str} |")

    # ── high-frequency keywords NOT covered ────────────────────────────────
    lines.append("")
    lines.append("## Top 100 uncovered high-frequency keywords")
    lines.append(
        "These appeared frequently in the corpus but are not covered by "
        "any seed entity. Many will be **themes/concepts** (battle, dharma, "
        "fear) which is correct — themes belong to the theme taxonomy, not "
        "the entity list. But scan for **proper nouns** that look like "
        "characters or places we missed, and add them via overrides."
    )
    lines.append("")
    lines.append("| Keyword | Count |")
    lines.append("|---|---:|")
    uncovered = [
        (kw, c) for kw, c in keyword_counts.most_common(500)
        if kw not in covered_terms
    ][:100]
    for kw, c in uncovered:
        lines.append(f"| {kw} | {c:,} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


# --- stub overrides file ---
STUB_OVERRIDES = {
    "_doc": (
        "Human-edited corrections to the machine-bootstrapped entity list. "
        "Re-running build_entities is always safe — this file is never "
        "overwritten. Schema below; remove _doc and the _example block "
        "before adding real entries (or just leave them in, they're ignored)."
    ),
    "_example": {
        "characters": {
            "add": {
                "ExampleCharacter": {
                    "aliases": ["AltName1", "AltName2"],
                    "type": "category-string",
                },
            },
            "remove": ["NotARealCharacter"],
            "modify": {
                "Arjuna": {
                    "add_aliases": ["NewEpithet"],
                    "remove_aliases": ["WrongAlias"],
                    "set_type": "Pandava",
                },
            },
            "merge": [
                {"into": "Bhima", "absorb": ["Bhimasena"]},
            ],
        },
        "places": {
            "add": {},
            "remove": [],
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


# --- orchestration entry point ---
def run(*, raw_path: Path, entities_path: Path,
        overrides_path: Path, report_path: Path):
    entities_path.parent.mkdir(parents=True, exist_ok=True)

    print("Counting corpus keywords...")
    keyword_counts = count_corpus_keywords(raw_path)
    print(f"  {len(keyword_counts):,} unique keyword tokens, "
          f"{sum(keyword_counts.values()):,} total")

    print("Building entities from seeds...")
    entities = build_from_seeds()
    print(f"  seeded characters: {len(entities['characters'])}")
    print(f"  seeded places:     {len(entities['places'])}")

    print("Loading overrides (if present)...")
    overrides = load_overrides(overrides_path)
    if overrides:
        print("  applying overrides")
        entities = apply_overrides(entities, overrides)
        print(f"  after overrides — characters: {len(entities['characters'])},"
              f" places: {len(entities['places'])}")
    else:
        print("  no overrides found (will create stub)")
        ensure_stub_overrides(overrides_path)

    with open(entities_path, "w") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)
    print(f"  wrote → {rel(entities_path)}")

    print("Writing coverage report...")
    write_coverage_report(entities, keyword_counts, report_path)
    print(f"  wrote → {rel(report_path)}")
