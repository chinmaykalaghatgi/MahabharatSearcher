"""CLI wrapper for Layer 1 Step 5 — build_indexes."""

from mahabharata.common import paths
from mahabharata.layer1 import indexes as lib


def main():
    lib.run(
        raw_path=paths.RAW_PATH,
        entities_path=paths.ENTITIES_PATH,
        themes_path=paths.THEMES_PATH,
        verse_chars_path=paths.VERSE_CHARS_PATH,
        verse_themes_path=paths.VERSE_THEMES_PATH,
        char_index_path=paths.CHAR_INDEX_PATH,
        group_index_path=paths.GROUP_INDEX_PATH,
        theme_index_path=paths.THEME_INDEX_PATH,
        report_path=paths.INDEXES_REPORT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
