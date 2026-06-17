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

import json
import re
from pathlib import Path

import numpy as np

from mahabharata.common.corpus_loader import stream_corpus
from mahabharata.common.sections import parse_sections


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH = 64

# Max characters per chapter-summary chunk. Kept well under bge's
# 512-token window (~2000 chars) so no chunk is silently truncated.
# ~1200 chars ≈ 10 verse-summary segments — fine scene granularity.
CHAPTER_CHUNK_MAX_CHARS = 1200

# Splits a chapter summary on its "[S12]" verse markers, keeping each
# marker attached to the text that follows it.
_SHLOKA_SPLIT_RE = re.compile(r"(\[S\d+\])")


def chunk_chapter_summary(
    summary: str, *, max_chars: int = CHAPTER_CHUNK_MAX_CHARS
) -> list[str]:
    """Pack a chapter summary's verse segments into <= max_chars chunks.

    Splits on ``[Sx]`` markers so chunk boundaries fall between verses,
    then greedily fills chunks. A short summary returns a single chunk
    (identical to the un-chunked behavior); a long one is split so a
    mid-chapter scene gets a vector that isn't diluted or truncated.
    """
    parts = _SHLOKA_SPLIT_RE.split(summary)
    # parts == ['', '[S0]', ' text ', '[S1]', ' text', ...]; recombine
    # each marker with the text that follows it into one segment.
    segments: list[str] = []
    i = 1
    while i < len(parts):
        marker = parts[i]
        text = parts[i + 1] if i + 1 < len(parts) else ""
        segments.append(marker + text)
        i += 2
    if not segments:  # no markers at all — treat whole summary as one
        seg = summary.strip()
        return [seg] if seg else []

    chunks: list[str] = []
    cur = ""
    for seg in segments:
        if cur and len(cur) + len(seg) > max_chars:
            chunks.append(cur.strip())
            cur = seg
        else:
            cur += seg
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


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


def build_chapter_index(
    *,
    chapter_summaries_path: Path,
    out_embeddings: Path,
    out_uids: Path,
    out_chunks: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
    max_chars: int = CHAPTER_CHUNK_MAX_CHARS,
) -> dict:
    """Build the chunked chapter-level dense index from Step 6 summaries.

    The coarse scene-localizer for Layer 3 (architecture doc Layer 2
    Choice 2 outcome). Each chapter's naive-rollup ``summary`` is split
    into <= ``max_chars`` chunks on verse boundaries and each chunk is
    embedded separately, so a mid-chapter scene gets its own vector
    instead of being truncated at bge's 512-token window (the failure the
    L3 eval surfaced: chapter context recall ~0.07). Ranking max-pools
    chunks back to the chapter (see ``layer2.chapter_dense``).

    Reads ``chapter_summaries.jsonl`` (one chapter per line); writes:
      - ``out_embeddings`` — (N_chunks, D) float32, L2-normalized
      - ``out_uids``       — N_chunks lines, the chapter_uid per row
      - ``out_chunks``     — N_chunks JSON lines {chapter_uid, chunk_idx,
                             text}, the chunk text Layer 3 shows the model
    All three are row-parallel.
    """
    print(f"Loading chapter summaries: {chapter_summaries_path}")
    uids: list[str] = []
    texts: list[str] = []
    chunk_meta: list[dict] = []
    n_chapters = 0
    n_empty = 0
    with open(chapter_summaries_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_chapters += 1
            chapter_uid = rec["chapter_uid"]
            summary = rec.get("summary") or ""
            chunks = chunk_chapter_summary(summary, max_chars=max_chars)
            if not chunks:
                n_empty += 1
                continue
            for idx, chunk in enumerate(chunks):
                uids.append(chapter_uid)
                texts.append(chunk)
                chunk_meta.append(
                    {
                        "chapter_uid": chapter_uid,
                        "chunk_idx": idx,
                        "text": chunk,
                    }
                )
    print(
        f"  {n_chapters:,} chapters -> {len(texts):,} chunks "
        f"({n_empty} empty summaries)"
    )

    print(f"Loading model: {model_name}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    print(f"  embedding dim: {dim}")

    print(f"Encoding {len(texts):,} chunks (batch_size={batch_size})...")
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
    with open(out_chunks, "w") as f:
        for m in chunk_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"  wrote: {out_embeddings}")
    print(f"  wrote: {out_uids}")
    print(f"  wrote: {out_chunks}")

    return {
        "n_chapters": n_chapters,
        "n_chunks": len(uids),
        "n_empty_summaries": n_empty,
        "mean_chunks_per_chapter": len(uids) / n_chapters if n_chapters else 0,
        "model": model_name,
        "dim": int(dim),
        "embeddings_path": str(out_embeddings),
        "uids_path": str(out_uids),
        "chunks_path": str(out_chunks),
    }
