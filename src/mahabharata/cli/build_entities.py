"""CLI wrapper for Layer 1 Step 1 — build_entities."""

from mahabharata.common import paths
from mahabharata.layer1 import entities as entities_lib


def main():
    entities_lib.run(
        raw_path=paths.RAW_PATH,
        entities_path=paths.ENTITIES_PATH,
        overrides_path=paths.ENTITIES_OVERRIDES_PATH,
        report_path=paths.ENTITIES_REPORT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
