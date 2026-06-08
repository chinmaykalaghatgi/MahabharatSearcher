"""Layer 2 Phase C / Step 7 — Reciprocal Rank Fusion.

Combines two or more ranked UID lists (BM25 lexical + dense semantic)
into a single ranking. RRF (Cormack, Clarke & Buettcher 2009):

    score(d) = Σᵢ 1 / (k + rankᵢ(d))

where rankᵢ(d) is the 1-based rank of document d in ranking i (a
document absent from ranking i contributes nothing for that i).

Why RRF over score combination
------------------------------
BM25 scores and cosine similarities live on incomparable scales —
BM25 is unbounded TF·IDF mass, cosine is [-1, 1]. Normalizing them to
combine linearly needs a learned or hand-tuned weight, and we have no
labelled query/relevance pairs to fit one. RRF sidesteps the score
distributions entirely and fuses on *rank position*, which is why it is
the standard parameter-free hybrid baseline. See architecture doc
Layer 2 Choice 6.

The constant k=60 is the literature default. It is a smoothing term:
larger k flattens the contribution curve (rank 1 vs rank 10 matter
less), smaller k sharpens it (top ranks dominate). 60 is robust across
corpora and we have no eval signal yet to justify tuning it.
"""

from __future__ import annotations

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked UID lists into one ranking by RRF score.

    Args:
        rankings: each element is a list of UIDs in rank order, best
            first. Lists may have different lengths and may overlap.
            Empty lists (a lane that returned nothing) are harmless.
        k: RRF smoothing constant.

    Returns:
        (uid, score) pairs sorted by descending fused score. Ties are
        broken by UID for deterministic output.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, uid in enumerate(ranking, start=1):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
