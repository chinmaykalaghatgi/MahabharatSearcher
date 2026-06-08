"""CLI wrapper for Layer 1 Step 4 — build_verse_themes."""

from mahabharata.common import paths
from mahabharata.layer1 import verse_themes as lib


def main():
    lib.run(
        raw_path=paths.RAW_PATH,
        themes_path=paths.THEMES_PATH,
        verse_themes_path=paths.VERSE_THEMES_PATH,
        report_path=paths.VERSE_THEMES_REPORT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
