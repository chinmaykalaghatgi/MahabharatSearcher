"""Phase C / Step 6 — Build the dense embedding index.

Streams the corpus, extracts each verse's fluent translation, encodes
the lot with `BAAI/bge-small-en-v1.5`, and writes:

  - `embeddings.npy` — shape (N, 384), float32, L2-normalized
  - `uids.txt`       — N lines, parallel order

This is an offline step, run once (rerun if the corpus changes or the
model is upgraded). At 73,820 verses, the encode pass takes minutes
on CPU and seconds on Apple-Silicon MPS — sentence-transformers
auto-detects the best available device.

L2-normalization is done at write time (`normalize_embeddings=True`)
so the runtime retriever can compute cosine similarity as a single
matmul without re-normalizing on each query.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mahabharata.common.corpus_loader import stream_corpus
from mahabharata.common.sections import parse_sections


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH = 64


def build(
    *,
    raw_path: Path,
    out_embeddings: Path,
    out_uids: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
) -> dict:
    """Build the dense index. Returns a small stats dict for the CLI."""
    print(f"Loading corpus: {raw_path}")
    records = list(stream_corpus(raw_path, verbose=False))
    print(f"  {len(records):,} records")

    print("Extracting fluent translations...")
    uids: list[str] = []
    texts: list[str] = []
    n_empty = 0
    for rec in records:
        ai_text = rec.get("ai_analysis", "")
        translation = ""
        if ai_text:
            sections = parse_sections(ai_text)
            translation = sections.get("Fluent Translation") or ""
        if not translation:
            n_empty += 1
        uids.append(rec["uid"])
        texts.append(translation)
    print(f"  {len(texts) - n_empty:,} non-empty, {n_empty} empty")

    print(f"Loading model: {model_name}")
    # Lazy import so paths-only ops (e.g. mbh-build-indexes) don't pay
    # the sentence-transformers import cost.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # `get_embedding_dimension` is the new name; `get_sentence_embedding_dimension`
    # is the backward-compatible alias that emits a FutureWarning in
    # sentence-transformers >= 5.x. Prefer the new name when available.
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    print(f"  embedding dim: {dim}")

    print(f"Encoding {len(texts):,} verses (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_embeddings, embeddings)
    out_uids.write_text("\n".join(uids) + "\n")
    print(f"  wrote: {out_embeddings}")
    print(f"  wrote: {out_uids}")

    return {
        "n_records": len(records),
        "n_empty_translations": n_empty,
        "model": model_name,
        "dim": int(dim),
        "embeddings_path": str(out_embeddings),
        "uids_path": str(out_uids),
    }
