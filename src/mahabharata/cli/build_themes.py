"""CLI wrapper for Layer 1 Step 2 — build_themes."""

from mahabharata.common import paths
from mahabharata.layer1 import themes as themes_lib


def main():
    themes_lib.run(
        raw_path=paths.RAW_PATH,
        entities_path=paths.ENTITIES_PATH,
        themes_path=paths.THEMES_PATH,
        overrides_path=paths.THEMES_OVERRIDES_PATH,
        report_path=paths.THEMES_REPORT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
