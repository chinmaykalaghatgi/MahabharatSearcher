"""CLI wrapper for Layer 1 Step 3 — build_verse_characters."""

from mahabharata.common import paths
from mahabharata.layer1 import verse_characters as lib


def main():
    lib.run(
        raw_path=paths.RAW_PATH,
        entities_path=paths.ENTITIES_PATH,
        verse_chars_path=paths.VERSE_CHARS_PATH,
        report_path=paths.VERSE_CHARS_REPORT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
