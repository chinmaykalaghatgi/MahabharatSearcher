"""
Data audit for ai_analysis field across all 73,820 verses.

Checks:
- Section presence and completeness
- Empty / null / missing records
- Section length distribution
- Keyword extraction: token counts, top terms
- Character name candidates (proper nouns in keywords)
- Distribution across books
- Malformed / anomalous records
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path("data/raw/search_engine_db.jsonl")

SECTIONS = [
    "Fluent Translation",
    "Poetic Translation",
    "Summary",
    "Keywords",
    "Dictionary",
]

# Section headers always appear as "**Section Name:**" at the start of a line.
# We anchor to line-start so bold markers *inside* sections
# (e.g. dictionary entries like "* **word**: def") don't get mistaken
# for the next section header.
SECTION_HEADER_RE = re.compile(
    r"^\*\*(" + "|".join(re.escape(s) for s in SECTIONS) + r"):\*\*\s*",
    re.MULTILINE,
)


def parse_sections(text):
    """Extract each section from the markdown blob.

    Returns dict of section_name -> content string (or None if absent).
    """
    all_matches = list(SECTION_HEADER_RE.finditer(text))
    result = {s: None for s in SECTIONS}
    for i, m in enumerate(all_matches):
        name = m.group(1)
        start = m.end()
        end = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(text)
        result[name] = text[start:end].strip()
    return result


def extract_keywords(keywords_text):
    """Split comma-separated keyword string into a list."""
    if not keywords_text:
        return []
    return [k.strip() for k in keywords_text.split(",") if k.strip()]


# ── counters ──────────────────────────────────────────────────────────────────
total = 0
null_analysis = 0
empty_analysis = 0
section_present = Counter()
section_empty = Counter()

keyword_counter = Counter()
keyword_counts_per_verse = []  # how many keywords per verse

section_lengths = defaultdict(list)  # section -> [char_length]

book_counts = Counter()
book_missing_analysis = Counter()

malformed = []  # uids where ai_analysis exists but no sections parsed


# ── main pass ─────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    for line in f:
        d = json.loads(line)
        total += 1
        uid = d["uid"]
        book = d["book"]
        book_counts[book] += 1

        ai = d.get("ai_analysis")

        if ai is None:
            null_analysis += 1
            book_missing_analysis[book] += 1
            continue
        if not ai.strip():
            empty_analysis += 1
            book_missing_analysis[book] += 1
            continue

        sections = parse_sections(ai)
        found_any = False

        for s in SECTIONS:
            val = sections[s]
            if val:
                section_present[s] += 1
                section_lengths[s].append(len(val))
                found_any = True
            else:
                section_empty[s] += 1

        if not found_any:
            malformed.append(uid)

        # keywords
        kw_text = sections.get("Keywords")
        kws = extract_keywords(kw_text)
        keyword_counts_per_verse.append(len(kws))
        keyword_counter.update(kws)


# ── report ────────────────────────────────────────────────────────────────────
def pct(n, d):
    return f"{n:,} ({100*n/d:.1f}%)"


print("=" * 60)
print("DATA AUDIT — ai_analysis field")
print("=" * 60)

print(f"\nTotal verses:          {total:,}")
print(f"ai_analysis null:      {pct(null_analysis, total)}")
print(f"ai_analysis empty:     {pct(empty_analysis, total)}")
print(f"No sections parsed:    {pct(len(malformed), total)}")
usable = total - null_analysis - empty_analysis - len(malformed)
print(f"Usable records:        {pct(usable, total)}")

print("\n-- Section presence (out of usable) -----------------------")
for s in SECTIONS:
    present = section_present[s]
    missing = section_empty[s]
    print(f"  {s:<22} present={present:,}  missing={missing:,}")

print("\n-- Section length stats (chars) ---------------------------")
for s in SECTIONS:
    lengths = section_lengths[s]
    if lengths:
        avg = sum(lengths) / len(lengths)
        mn = min(lengths)
        mx = max(lengths)
        sl = sorted(lengths)
        med = sl[len(sl) // 2]
        print(
            f"  {s:<22} avg={avg:,.0f}  med={med:,}"
            f"  min={mn}  max={mx:,}"
        )

print("\n-- Keyword stats ------------------------------------------")
total_kw = sum(keyword_counts_per_verse)
n_verses = len(keyword_counts_per_verse)
avg_kw = total_kw / n_verses if n_verses else 0
zero_kw = sum(1 for x in keyword_counts_per_verse if x == 0)
print(f"  Total keyword tokens:  {total_kw:,}")
print(f"  Avg per verse:         {avg_kw:.1f}")
print(f"  Verses with 0 kw:      {pct(zero_kw, total)}")
print(f"  Unique keyword tokens: {len(keyword_counter):,}")

print("\n  Top 50 keywords:")
for kw, cnt in keyword_counter.most_common(50):
    print(f"    {cnt:5,}  {kw}")

print("\n-- Book-level coverage ------------------------------------")
print(f"  {'Book':>4}  {'Verses':>7}  {'Missing analysis':>16}")
for book in sorted(book_counts):
    missing = book_missing_analysis.get(book, 0)
    print(f"  {book:>4}  {book_counts[book]:>7,}  {missing:>16,}")

if malformed:
    print("\n-- Malformed records (no sections parsed) -----------------")
    print(f"  Count: {len(malformed)}")
    print(f"  First 20: {malformed[:20]}")

print("\n" + "=" * 60)
print("Audit complete.")
print("=" * 60)
