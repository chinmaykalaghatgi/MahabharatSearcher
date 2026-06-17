"""CLI wrapper — build the chapter-level dense index.

Coarse scene-localizer for Layer 3 synthesis. Embeds the Step 6 chapter
summaries (`mbh-build-summaries` must have run first).

Usage:
    .venv/bin/mbh-build-chapter-embeddings
"""

from mahabharata.common import paths
from mahabharata.layer2 import embed as lib


def main():
    stats = lib.build_chapter_index(
        chapter_summaries_path=paths.CHAPTER_SUMMARIES_PATH,
        out_embeddings=paths.CHAPTER_DENSE_EMBEDDINGS_PATH,
        out_uids=paths.CHAPTER_DENSE_UIDS_PATH,
        out_chunks=paths.CHAPTER_DENSE_CHUNKS_PATH,
    )
    print()
    print("Done.")
    print(f"  chapters:        {stats['n_chapters']:,}")
    print(f"  chunks:          {stats['n_chunks']:,}")
    print(f"  chunks/chapter:  {stats['mean_chunks_per_chapter']:.2f}")
    print(f"  empty summaries: {stats['n_empty_summaries']:>4}")
    print(f"  model:           {stats['model']}")
    print(f"  dim:             {stats['dim']}")


if __name__ == "__main__":
    main()
