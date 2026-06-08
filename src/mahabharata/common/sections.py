"""Shared parser for the Gemini `ai_analysis` markdown blob.

Every Layer 1 build script needs to pull the Keywords section out of
each verse's ai_analysis field. Before the src/ refactor each script
had its own copy of SECTIONS + SECTION_HEADER_RE + parse_keywords_section.
This module is the one source of truth — identical behavior, importable.
"""

import re

SECTIONS = [
    "Fluent Translation",
    "Poetic Translation",
    "Summary",
    "Keywords",
    "Dictionary",
]

# Section headers always appear as "**Section Name:**" at the start of a
# line. Anchoring to line-start is important so bold markers *inside* a
# section (e.g. dictionary entries like "* **word**: def") don't get
# mistaken for the next section header.
SECTION_HEADER_RE = re.compile(
    r"^\*\*(" + "|".join(re.escape(s) for s in SECTIONS) + r"):\*\*\s*",
    re.MULTILINE,
)


def parse_sections(text):
    """Extract every section from an ai_analysis blob.

    Returns dict of section_name -> content string (or None if absent).
    """
    all_matches = list(SECTION_HEADER_RE.finditer(text))
    result = {s: None for s in SECTIONS}
    for i, m in enumerate(all_matches):
        name = m.group(1)
        start = m.end()
        end = (all_matches[i + 1].start()
               if i + 1 < len(all_matches) else len(text))
        result[name] = text[start:end].strip()
    return result


def parse_keywords_section(ai_text):
    """Return the comma-separated keywords list from an ai_analysis blob."""
    matches = list(SECTION_HEADER_RE.finditer(ai_text))
    for i, m in enumerate(matches):
        if m.group(1) == "Keywords":
            start = m.end()
            end = (matches[i + 1].start()
                   if i + 1 < len(matches) else len(ai_text))
            kw_text = ai_text[start:end].strip()
            return [k.strip() for k in kw_text.split(",") if k.strip()]
    return []
