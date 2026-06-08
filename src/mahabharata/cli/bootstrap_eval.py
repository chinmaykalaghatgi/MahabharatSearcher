"""CLI wrapper for Layer 2 Phase B Step 2 — bootstrap_eval."""

from mahabharata.common import paths
from mahabharata.layer2 import eval_bootstrap as lib


def main():
    lib.run(
        char_index_path=paths.CHAR_INDEX_PATH,
        theme_index_path=paths.THEME_INDEX_PATH,
        entities_path=paths.ENTITIES_PATH,
        draft_path=paths.EVAL_SET_DRAFT_PATH,
        report_path=paths.EVAL_BOOTSTRAP_REPORT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
