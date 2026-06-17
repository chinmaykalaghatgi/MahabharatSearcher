"""Layer 3 — Synthesis.

Grounded RAG over retrieved context. Layer 2 localizes; Layer 3 reads
the localized context and assembles an answer, citing verse UIDs and
abstaining when the context doesn't support one.

- `context.py`     — coarse chapter localization + fine verse gathering
- `synthesize.py`  — prompt construction + local (Ollama) generation

See `docs/theoretical_concepts_and_architecture.md` (Layer 3) for the
design rationale: RAG over parametric recall, conditional invocation,
chapter-localize-then-verse-gather, small-model-first, mandatory
citation + abstention.
"""
