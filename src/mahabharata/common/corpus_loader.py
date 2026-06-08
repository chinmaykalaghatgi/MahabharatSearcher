"""Raw corpus loader with in-memory UID collision patching.

Exists because `data/raw/search_engine_db.jsonl` has 4 distinct verses
that share UIDs with earlier verses (see docs/project_context.md "UID
collisions" section for the full investigation). Until the raw corpus
is properly rebuilt, this module applies a minimal, stable patch to
re-label the 4 stray records with synthetic `_orphan` UIDs so
downstream consumers see a unique-UID view of the corpus.

Why not rewrite the raw file: the raw file is single source of
truth; patching at read time keeps the fix reversible and out of the
way, and lets us iterate on the fix without re-running expensive
Gemini annotations. Layer 1 build scripts that care about UIDs call
`stream_corpus()` instead of opening the raw file directly.

The patch is keyed on `(uid, sanskrit_prefix)` rather than line
number so it remains stable under raw-file reordering.
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path


# Each entry is one stray record in the raw corpus.
# match_uid + match_sanskrit_prefix uniquely identifies the record;
# new_uid is the replacement canonical.
UID_COLLISION_PATCH = [
    {
        "match_uid": "B3_C282_S1",
        "match_sanskrit_prefix": (
            "[मार्क] एतस्मिन्न एव काले तु दयुमत्सेनॊ"
        ),
        "new_uid": "B3_C282_S1_orphan",
        "note": (
            "Stray record wedged between S21 and S22 of ch 282. "
            "Likely a mid-chapter star-passage whose shloka number "
            "defaulted to 1 when the scraper couldn't parse its "
            "marker. Narrative content is the Savitri story "
            "conclusion (Dyumatsena regaining his sight)."
        ),
    },
    {
        "match_uid": "B6_C27_S5",
        "match_sanskrit_prefix": (
            "सुहृदं सर्वभूतानां जञात्वा मां शान्तिम"
        ),
        "new_uid": "B6_C27_S5_orphan",
        "note": (
            "Confirmed split: this is the literal second half of "
            "Bhagavad Gita 5.29. The first half lives at the "
            "correctly-numbered B6_C27_S29 (immediately prior "
            "line in the raw file). The second half was "
            "mis-numbered as S5, colliding with the real S5."
        ),
    },
    {
        "match_uid": "B13_C33_S1",
        "match_sanskrit_prefix": (
            "[भ] बराह्मणान एव सततं भृशं संप्रतिपूजयेत"
        ),
        "new_uid": "B13_C33_S1_orphan",
        "note": (
            "Trailing stray after S25 of ch 33. Likely a "
            "chapter-end star-passage defaulted to S1. Content "
            "is a Bhishma teaching about honoring Brahmins."
        ),
    },
    {
        "match_uid": "B13_C74_S1",
        "match_sanskrit_prefix": (
            "[य] विधिं गवां परम अहं शरॊतुम इच्छामि"
        ),
        "new_uid": "B13_C74_S1_orphan",
        "note": (
            "Trailing stray after S39 of ch 74. Likely a "
            "chapter-end star-passage defaulted to S1. Content "
            "is Yudhishthira asking about the rules regarding cows."
        ),
    },
]


# Precomputed lookups for the hot path (73,820 records per stream).
_PATCH_UIDS = {p["match_uid"] for p in UID_COLLISION_PATCH}
_PATCH_BY_UID = {}
for _p in UID_COLLISION_PATCH:
    _PATCH_BY_UID.setdefault(_p["match_uid"], []).append(_p)


def _maybe_patch(d):
    """If this record matches a collision patch, mutate its uid in
    place and return the patch entry that fired. Otherwise return
    None."""
    uid = d.get("uid")
    if uid not in _PATCH_UIDS:
        return None
    sanskrit = d.get("sanskrit", "")
    for p in _PATCH_BY_UID[uid]:
        if sanskrit.startswith(p["match_sanskrit_prefix"]):
            d["uid"] = p["new_uid"]
            return p
    return None


def _open_corpus(path):
    """Open the corpus for text reading, transparently handling gzip.

    Resolution: use `path` as given if it exists; otherwise fall back to
    a sibling `<path>.gz` if present. A `.gz` path (given or resolved) is
    read through gzip. The repo ships the corpus gzipped
    (`search_engine_db.jsonl.gz`, ~33MB vs 164MB raw) since the plain
    file exceeds GitHub's size limit; a local uncompressed `.jsonl`, if
    present, takes precedence because it streams a bit faster.
    """
    p = Path(path)
    if not p.exists():
        gz = Path(str(p) + ".gz")
        if gz.exists():
            p = gz
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, encoding="utf-8")


def stream_corpus(path, *, verbose=True):
    """Yield parsed records from the raw corpus JSONL with collision
    patches applied in-memory. After exhaustion, validates that every
    patch fired exactly once and prints a warning otherwise.

    `path` may point at the uncompressed `.jsonl` or the gzipped
    `.jsonl.gz`; see `_open_corpus`."""
    applied = Counter()
    try:
        with _open_corpus(path) as f:
            for line in f:
                d = json.loads(line)
                p = _maybe_patch(d)
                if p is not None:
                    applied[p["new_uid"]] += 1
                yield d
    finally:
        expected = {p["new_uid"] for p in UID_COLLISION_PATCH}
        missing = sorted(expected - set(applied))
        duplicated = sorted(u for u, n in applied.items() if n > 1)
        if missing:
            print(
                f"[WARN] corpus_loader: patches did not fire: "
                f"{missing}. Raw corpus content may have changed.",
                file=sys.stderr,
            )
        if duplicated:
            print(
                f"[WARN] corpus_loader: patches matched multiple "
                f"records: {duplicated}. Prefix is no longer unique.",
                file=sys.stderr,
            )
        if verbose and applied and not missing and not duplicated:
            print(
                f"  [corpus_loader] applied {sum(applied.values())} "
                f"UID collision patches: "
                f"{', '.join(sorted(applied))}"
            )
