"""CLI wrapper for Layer 1 Step 6 — build_summaries."""

from mahabharata.common import paths
from mahabharata.layer1 import summaries as lib


def main():
    lib.run(
        raw_path=paths.RAW_PATH,
        verse_chars_path=paths.VERSE_CHARS_PATH,
        verse_themes_path=paths.VERSE_THEMES_PATH,
        chapter_summaries_path=paths.CHAPTER_SUMMARIES_PATH,
        parva_summaries_path=paths.PARVA_SUMMARIES_PATH,
        report_path=paths.SUMMARIES_REPORT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
