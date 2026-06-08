"""CLI wrapper for Layer 2 Phase C / Step 6 — dense index build.

Usage:
    .venv/bin/mbh-build-embeddings
"""

from mahabharata.common import paths
from mahabharata.layer2 import embed as lib


def main():
    stats = lib.build(
        raw_path=paths.RAW_PATH,
        out_embeddings=paths.DENSE_EMBEDDINGS_PATH,
        out_uids=paths.DENSE_UIDS_PATH,
    )
    print()
    print("Done.")
    print(f"  records:           {stats['n_records']:,}")
    print(f"  empty translations:{stats['n_empty_translations']:>4}")
    print(f"  model:             {stats['model']}")
    print(f"  dim:               {stats['dim']}")


if __name__ == "__main__":
    main()
