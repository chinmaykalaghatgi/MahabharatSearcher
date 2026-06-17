# Theoretical Concepts & Architecture

Companion to `project_context.md`. Where `project_context.md` describes *what*
the project is and its current state, this file documents the *why* — the
theoretical reasoning and design choices made per layer. It is meant as a
learning artifact: reading it should teach the concepts behind the build, not
just describe the build.

Each layer section captures:
- **Theoretical basis** — the principle or prior art the layer rests on
- **Design choices** — concrete decisions and the alternatives considered
- **Tradeoffs** — what we gave up, and under what conditions we'd reconsider
- **Open questions** — things we punted on, to revisit later

---

## Layer 1 — Structured Knowledge

### Theoretical basis

**Closed corpus, offline pre-computation.**
The Mahabharata is a finite, immutable text: 73,820 verses, 18 books, fixed.
This is radically different from open-domain problems where the "world" grows
continuously. For a closed corpus, the cost of pre-building structured
knowledge is paid *once*, and amortized over every future query. This inverts
the usual assumption behind LLM-first systems, which optimize for flexibility
at the expense of per-query cost.

The theoretical frame here is **materialized views** (from database theory):
rather than re-deriving facts from raw text on every query, we compute them
once, store them, and let retrieval become a lookup problem instead of a
reasoning problem. When the underlying data never changes, materialization is
a near-free win.

**Why not just rely on an LLM's parametric knowledge?**
The previous project (Mahabharat_Python_Test) tried to fine-tune an 8B model
to "know" the Mahabharata. This conflates *knowledge storage* with *reasoning*.
Storage is cheap (disk) and exact; reasoning is expensive (compute) and
approximate. Pushing storage into model weights is strictly worse than a
retrieval layer for a closed corpus — you pay compute to re-derive facts you
already have on disk, and you get hallucinations as a bonus.

**Why structured, not just chunks + embeddings?**
Pure vector retrieval works well for "find me similar text" but poorly for
queries with structural constraints: *"every verse where Arjuna speaks in
Book 5"*, *"chapter summaries for the Drona Parva"*, *"all events where
Krishna intervenes"*. These are **structural queries**, answerable in O(1)
lookup if the data is pre-indexed, but requiring brittle approximation if the
data is only embedded as vectors. Structured knowledge is the right tool for
structural queries; embeddings are the right tool for paraphrase/concept
queries. Layer 1 and Layer 2 are complementary, not competing.

### Design choices

#### Choice 1 — Keyword-derived entity extraction vs named entity recognition (NER)

**What we're doing:** Building the character/entity list from the `keywords`
field that Gemini already produced during offline annotation.

**Alternative considered:** Training or using a Sanskrit/English NER model to
extract entities from translations.

**Why the keyword path wins here:**
- The work is already done. Gemini extracted keywords per verse during a
  prior offline pass, and the audit shows the top keywords *are* the main
  characters (Arjuna, Krishna, Karna, etc.) appearing with consistent
  spelling thousands of times each.
- NER models trained on modern English news corpora mis-handle Indic proper
  nouns and often fail on Sanskritic morphology. Using one here would
  introduce a new failure mode for no benefit.
- Cost: zero additional compute. The audit found 24,441 unique keyword
  tokens across the corpus — we only need to dedupe and alias-merge them.

**What we give up:** Characters who appear in a verse without being named in
the keywords will be missed. In practice, the Gemini keyword extraction is
high-recall for named figures, so the loss is small.

#### Choice 2 — Alias normalization is mandatory, not optional

**The problem:** Sanskrit (and the Mahabharata in particular) uses dozens of
epithets for major characters. Arjuna is also called Partha, Dhananjaya,
Phalguna, Savyasachi, Kiritin, Jishnu, Gudakesha, and several more. A naive
count that treats these as separate entities will fragment the character
index and produce bad retrieval.

**What we're doing:** Manually curating an alias map for the top ~50
characters and rolling all variants into a single canonical form. For the
long tail (minor figures), we accept some fragmentation.

**Theoretical note:** This is **entity resolution** — a well-studied problem
in record linkage and knowledge-base construction. The serious version uses
string similarity, embedding similarity, and supervised classifiers. For a
closed, well-known corpus where the top entities are few and famous, manual
curation is both cheaper and more accurate than a trained model.

#### Choice 3 — Controlled theme vocabulary vs unsupervised topic modeling

**What we're doing:** Defining a fixed taxonomy of ~30-50 themes (dharma,
karma, war, devotion, kingship, grief, fate, etc.) and mapping raw keywords
to this taxonomy.

**Alternatives considered:**
- **LDA / BERTopic** (unsupervised topic modeling): Lets themes emerge from
  the data. More "principled" but produces noisy, hard-to-name clusters and
  requires tuning the number of topics. For a well-understood corpus, this
  feels like a strictly worse version of what a human can do in an hour.
- **Per-verse LLM theme classification**: High quality but slow and costs
  money. Overkill when keywords already carry the signal.

**The controlled-vocabulary approach** is the standard technique in library
science and document indexing (e.g. MeSH for medical literature, the Library
of Congress Subject Headings). It's not fashionable in ML circles because it
requires domain knowledge, but for a closed corpus with domain structure
(the Mahabharata *has* well-known themes), it's the right tool.

**Tradeoff:** The taxonomy becomes part of the project's ontology — we are
making editorial choices about what "counts" as a theme. This is both a
feature (we control semantics) and a risk (our taxonomy may miss
subthemes). We'll validate by spot-checking.

#### Choice 4 — Naive rollup for chapter summaries, not LLM-generated

**What we're doing, to start:** Chapter summaries built by concatenating
verse-level summaries (~plus minimal structure). No LLM call.

**Alternative considered:** Gemini-generated chapter summaries — one call per
chapter (~1,995 calls total). More readable, but introduces a second
dependency on external inference and a new failure mode (summary drift from
source).

**Why start naive:** The point of Layer 1 is to build the *data* that Layer 2
retrieves. If naive concatenation is enough to ground good retrieval, the
LLM pass was unnecessary. If it isn't, we can upgrade later. This is the
**YAGNI principle** applied to pipeline steps: don't pay for capability
before you've demonstrated that you need it.

> **Outcome (2026-06-08): shipped naive, and it paid off twice.** Step 6 built
> 1,995 chapter digests + 18 parva aggregates by pure concatenation, no LLM
> (`chapter_summaries.jsonl` + `parva_summaries.json`). Step 6↔Step 5 tally
> parity is exact. The naive text then turned out to be *good enough to embed*:
> the chapter-dense index built on it (2026-06-10) localizes scenes well enough
> to serve as the Layer 3 context-fetch substrate (see Layer 2 Choice 2 outcome
> and the Layer 3 section). The ~1,995-call Gemini pass was never needed.

#### Choice 5 — Defer the event graph (Step 7)

**The ambitious vision:** Extract `(subject, predicate, object, verse_uid)`
triples from every verse summary, creating a queryable knowledge graph of who
did what to whom. This is a real research project on its own — it requires
either a rule-based extractor (brittle, incomplete) or an LLM pass (expensive,
noisy) — and the output is only useful once we're doing comparative /
reasoning queries, which is a Layer 3 concern.

**Decision:** Defer Step 7 to *after* Layers 2-3 are working end-to-end. We
want a queryable Mahabharata tool as fast as possible; the event graph is an
enhancement, not a prerequisite. Once retrieval works and we see which query
types it *can't* answer, we'll know what shape of event graph we actually
need — and we'll build it for those queries specifically instead of building
it speculatively.

This is a **critical path** discipline: ship the shortest path to a working
system, then extend. The alternative (build everything, then integrate)
produces expensive, over-engineered artifacts that may not fit the actual
usage patterns.

### Build steps (dependency order)

```
Phase A — Foundation
  Step 1: Canonical entity list   (entities.json)
  Step 2: Theme taxonomy          (themes.json)

Phase B — Per-verse tagging (depends on A)
  Step 3: Character tagging pass  (verse_characters.jsonl)
  Step 4: Theme tagging pass      (verse_themes.jsonl)

Phase C — Aggregations (depends on B)
  Step 5: Character index         (character_index.json)
  Step 6: Chapter/parva summaries (chapter_summaries.jsonl)
  Step 8: Validation / spot-check

Deferred
  Step 7: Event/triple graph      (events.jsonl) — revisit after Layer 3
```

### Step 2 in detail — Theme taxonomy

#### Goal

Produce `data/layer1/themes.json`, a controlled vocabulary that maps each
canonical theme to its keyword variants, grouped into a small set of theme
families. Downstream, Step 4 uses it to tag every verse with the themes it
touches, in the same way Step 3 uses `entities.json` to tag characters.

#### Theoretical framing: themes as a retrieval facet

From information retrieval theory, **faceted search** organizes results along
multiple orthogonal axes — e.g. entity, theme, book, chapter — each with its
own controlled vocabulary, independently indexed. For a narrative corpus the
two most useful facets are the *who* (entities) and the *what it's about*
(themes). Queries then decompose cleanly:

- *"What does the Mahabharata say about dharma?"* → pure theme query
- *"What does Krishna say about dharma?"* → theme × entity intersection
- *"All grief verses in the Drona Parva"* → theme × book slice

Each of those reduces to a set operation over pre-indexed lists — O(1) lookup
instead of approximate reasoning. Step 1 built the entity facet; Step 2 builds
the theme facet. Together they unlock structural queries that pure vector
retrieval handles poorly.

This framing also explains why we do Step 2 *before* any embedding work:
facets are the skeleton the rest of retrieval hangs off. Embeddings fill in
the gaps (paraphrase, concept-drift), but the facets carry the load on
queries that have structure.

#### Why hand-curate rather than let themes emerge

Covered in Choice 3 above, recapped briefly: for a closed, well-understood
corpus with known structure, any clustering algorithm will rediscover buckets
a domain-aware human can write in an afternoon, and then require a post-hoc
naming pass. The controlled-vocabulary tradition from library science (MeSH,
LCSH) is the right ancestor here, not unsupervised topic modeling. We are
making editorial choices about ontology on purpose.

#### Theme families (organizational, not retrieval)

We cluster the ~30-50 themes into ~6 families:

- **Ethical / moral** — dharma, adharma, karma, sin, virtue, truth,
  righteousness
- **Emotional** — anger, grief, fear, love, jealousy, compassion, pride
- **Martial** — battle, weapon, death, victory, defeat, valor, strategy
- **Spiritual / devotional** — devotion, meditation, liberation, yoga,
  renunciation, sacrifice
- **Social / political** — kingship, caste, family, marriage, lineage,
  counsel
- **Cosmological / fated** — fate, curse, boon, prophecy, divine intervention

**Families are diagnostic, not queried.** Their job is to make coverage gaps
visible: if the emotional family has 4 themes with a total of 2K corpus hits
while the martial family has 8 themes with 15K hits, the taxonomy is
lopsided and we should either add emotional themes or accept the asymmetry
deliberately. Queries still hit themes directly, never families.

A two-level hierarchy (family → theme) is enough. Deeper trees add
organizational cost without clear retrieval benefit at this corpus size.

#### Methodology (mirrors Step 1)

1. **Seed taxonomy.** Hand-written dict
   `{family → {canonical_theme: [variants]}}`. Variants include
   morphological forms (*dharma, dharmas, dharmic*), common synonyms
   (*battle, war, combat*), and the actual spellings Gemini used in the
   keyword field — anchored against the audit's top-theme frequencies so we
   start from real corpus signal, not vibes.
2. **Corpus grounding.** Count each variant in the keyword field.
   Zero-hit variants flag typos or concepts Gemini uses under a different
   word. Same sanity pass as `entities_coverage_report.md`.
3. **Residual analysis.** Compute the top-N highest-frequency keyword
   tokens *not* covered by any theme variant *and* not present in
   `entities.json`. These are the honest candidates we missed. Write them
   to the coverage report as "uncovered candidates" so a human can decide
   per-token whether to extend the taxonomy via overrides. This is a
   closed-loop, corpus-driven validation: the data itself tells us where
   the taxonomy is thin.
4. **Override pattern.** `themes_overrides.json` is never overwritten,
   supports `add / remove / modify / merge` against the seed output.
   Re-running `build_themes.py` is always safe.
5. **Outputs.**
   - `data/layer1/themes.json` — final merged taxonomy
   - `data/layer1/themes_coverage_report.md` — per-theme counts, family
     totals, uncovered-keyword tail
   - `data/layer1/themes_overrides.json` — stub on first run

#### Decisions deferred

- **Multi-label per keyword.** A keyword like "sacrifice" plausibly belongs
  to both ritual and devotion. Start single-label for simplicity; revisit
  if downstream tagging (Step 4) produces visibly wrong attributions.
- **Multi-family membership for a theme.** Same story — single-family first,
  lift the restriction only if forced choices are distorting the
  coverage report.
- **Stemming vs explicit variants.** We list morphological variants by hand
  rather than running a stemmer. Cheaper to audit and avoids English
  stemmer assumptions leaking into Sanskritic terms.

#### Failure modes to watch for

- **Over-aliasing.** Rolling "truth" into "dharma" would fragment a
  legitimate distinct theme. Default to more themes, not fewer — the
  taxonomy can always be merged later, but splitting after Step 4 has
  tagged verses is expensive.
- **Surface-word vs concept mismatch.** A verse *about* dharma may not
  contain the word "dharma" in its keywords; a verse containing the word
  may be about something adjacent. This is the same limitation Step 1
  accepts for entities — inherent to keyword-derived tagging — and is
  flagged as an open question for Step 8 spot-checking rather than
  something Step 2 can solve on its own.
- **Entity bleed.** If entity tokens (Arjuna, Krishna) show up in the
  uncovered-keyword tail, something is wrong with the entity alias map,
  not the theme seed. The residual analysis catches this for free by
  subtracting `entities.json` before reporting uncovered candidates.

### Step 3 in detail — Per-verse character tagging

#### Goal

Produce `data/layer1/verse_characters.jsonl`, one record per verse,
listing which canonical characters (and groups) are mentioned in that
verse. This is the per-verse materialisation of the entity facet: the
data Step 5 will invert into a character → verses index, and the data
Layer 2 retrieval will hit for any entity-constrained query.

Record shape (one JSON object per line):
```json
{"uid": "B1_C1_S1", "characters": ["Vyasa"], "groups": []}
```

#### Theoretical framing: dictionary-based NER on a closed corpus

Named Entity Recognition (NER) in the open-domain setting is a
supervised learning problem — you train a sequence model to tag spans
because the entity vocabulary is unbounded, new names appear every
day, and the context around a name is what tells you it's an entity
rather than a common noun.

A closed corpus with a famous, finite cast inverts this. We already
know every significant character's name and epithets (Step 1). The
tagging problem collapses from "recognise entities in text" to
"look up known strings in text" — a **gazetteer-matching** or
**dictionary-based NER** approach. This is the oldest form of NER and
the right tool whenever the entity list is enumerable and stable:

- **Precision is high by construction.** If the string "Yudhishthira"
  appears, there is no real-world ambiguity about who it refers to in
  this corpus.
- **Recall is bounded by the gazetteer.** Any character we didn't put
  in `entities.json` will simply not be tagged. Step 1's alias
  curation sets the recall ceiling; Step 3 cannot exceed it.
- **Ambiguity is handled upstream, not at tag time.** When Step 1
  dropped `Rama` as an alias of Balarama (over-claimed Parashurama
  and Ramayana references), it decided the correct behaviour was
  **not to tag** rather than to guess. Step 3 inherits those
  decisions for free.

This is also the reason we do not reach for a Sanskrit/English NER
model here. A trained model would add one new failure mode (model
errors on unfamiliar Sanskritic morphology) for zero benefit — the
gazetteer already covers the entities we care about, and the ones
it doesn't cover are minor enough that mis-tagging them would be
worse than not tagging them.

#### Why keywords are the right input field

Each verse's `ai_analysis` has five sections. Step 3 tags against
the **Keywords** section only, not the full translation or summary.
Three reasons:

1. **The keywords field is where the entity list came from.** Step 1
   built `entities.json` by grounding its seed against the corpus
   keyword counter. Tagging against the same field means Step 3's
   recall matches Step 1's coverage report exactly — no drift.
2. **Keywords are already tokenised.** The field is a flat
   comma-separated list of canonical tokens Gemini chose per verse.
   No sentence splitting, no stemming, no compound handling. The
   match is a set membership test.
3. **Keywords bias toward "salient" entities.** Gemini's keyword
   extraction tends to surface the characters a verse is *about*,
   not every name transiently mentioned. A full-text match would
   inflate tag lists with throwaway references (e.g. a verse that
   mentions "Arjuna said" as a speech marker, but is really about
   something Krishna is explaining). Keywords give us a better
   notion of "present in this verse" than raw text would.

**What we give up:** some verses mention a character only in the
translation, not the keywords. Those will be missed. This is the
same recall gap we accepted in Choice 1 of Layer 1, and the right
place to revisit it is Step 8 (spot-check validation) — if the
gap is large, we can add a full-text fallback pass; if not, we
don't need to.

#### Groups vs members: tag atomically, expand at query time

When a verse says "Pandavas", what should the tag list contain?

Two options:

- **Atomic group tagging**: write `{"groups": ["Pandavas"]}` only
- **Group expansion at tag time**: write `{"characters":
  ["Yudhishthira", "Bhima", "Arjuna", "Nakula", "Sahadeva"]}`

We choose atomic tagging. Rationale:

- **Tag data should say what the corpus said.** Expanding "Pandavas"
  to its five members fabricates a claim — the verse might be about
  the Pandavas collectively in a way where no individual member is
  meaningfully "in" the verse.
- **Expansion is cheap at query time.** The `members` field on each
  group in `entities.json` lets Layer 2 expand on demand: a query
  for "Arjuna" can pull both his own tagged verses *and* any
  Pandavas-tagged verses, with a flag telling the caller which
  matches were direct and which were via group expansion.
- **It is reversible.** Atomic group tags can always be expanded
  later; expanded tags cannot be un-expanded without re-running
  the pass.

This is the same principle as storing normalised data in a database
and joining at query time, rather than denormalising upfront. The
storage is cleaner, the semantics are truthful, and the flexibility
lives on the query side where it can be applied selectively.

#### Methodology

1. **Load `entities.json`.** Build an alias → canonical map for
   characters and, separately, for groups. (Places are not tagged
   in Step 3 — they are a separate facet, deferred until we know
   we need them.) Fail loudly on alias collisions: if two canonicals
   claim the same alias, Step 1 has a bug that needs fixing before
   Step 3 runs.
2. **Stream the corpus** line-by-line (73,820 records). For each
   verse, parse the Keywords section using the same `re.MULTILINE`
   parser Step 1/Step 2 used.
3. **Set-intersect** keywords against the alias map. Collect the
   set of canonical characters and groups matched. Preserve order
   by corpus appearance (useful for debugging, cheap to do).
4. **Write one JSONL record per verse** — always, even empty. A
   verse with no character or group matches gets
   `{"uid": ..., "characters": [], "groups": []}`. Empty records
   are informative: they say "this verse is about something other
   than a named entity" rather than silently absent.
5. **Coverage report**. Summary stats the user will want:
   - total verses tagged vs verses with ≥1 character
   - distribution: % of verses with 0/1/2/3+ characters
   - per-character count of tagged verses (cross-check against
     Step 1's keyword-count column — they should match within the
     small slop introduced by case-sensitivity and alias dedupe)
   - top 20 most-tagged characters, for eyeballing

#### Decisions deferred

- **Speaker vs subject attribution.** The same open question from
  Layer 1 — a verse where Krishna speaks to Arjuna tags both, but
  the query semantics differ. For v1, treat "mentioned" as the
  atomic relation. Revisit in Step 8 if spot-checks reveal this
  matters for queries.
- **Full-text fallback.** If Step 8 shows the keyword-only recall
  is insufficient, add a second pass that greps the fluent
  translation against the alias map and merges with the keyword
  tag list. Don't build it pre-emptively.
- **Place tagging.** Places (Hastinapura, Kurukshetra, etc.) are a
  separate facet and intentionally not in Step 3. When a query
  type surfaces that needs location, build `verse_places.jsonl`
  as a parallel step — same shape, same pattern.
- **Confidence scores.** Gazetteer matches are binary. We are not
  going to fabricate a confidence score. If probabilistic tagging
  becomes necessary (e.g. for full-text fallback where partial
  matches and morphological variants matter), revisit then.

#### Failure modes to watch for

- **Alias collisions across canonicals.** Step 1 should have caught
  these, but Step 3 re-checks and fails loudly if the alias map
  isn't injective. A silent collision would mean one canonical's
  tags leak into another's.
- **Cross-check drift from Step 1 coverage.** If Step 3's
  per-character verse counts don't roughly match Step 1's keyword
  counts (within a small slop for dedup / single-verse-multi-alias
  cases), something is wrong with either the Step 1 counts or the
  Step 3 matcher — they're using the same signal and should agree.
- **Unexpected group dominance.** If "Pandavas" as a group tags
  thousands of verses that would be better attributed to a specific
  Pandava, it's a corpus style signal, not a bug — but worth
  noting so query-time group expansion is well-calibrated.

### Open questions for Layer 1

- **How do we handle speaker attribution?** Many verses are spoken *by* a
  character (e.g. Krishna speaking to Arjuna). The keyword field may not
  distinguish *subject of verse* from *speaker of verse*. This matters for
  queries like "what did Krishna say about dharma?" — revisit if spot-checks
  reveal problems.
- **Do we need a place/location index?** Locations like Kurukshetra, Hastinapura,
  Indraprastha appear in keywords. Building a parallel place index is cheap
  but not clearly necessary yet. Defer until a query needs it.
- **Are the Gemini keyword fields themselves reliable?** The audit only
  checked presence and coverage, not factual correctness. Before we over-trust
  them, we should spot-check ~50 random verses against a canonical translation
  (Ganguli, van Buitenen, or Bibek Debroy).

---

## Layer 2 — Retrieval

### Theoretical basis

**Retrieval is the ceiling, not the polish.**
Layer 2's job is to produce a small, high-precision candidate verse set for
Layer 3 to reason over — not to answer queries itself. The division of labour
matters: if retrieval quietly filters out the right verse, no amount of
synthesis cleverness recovers it. Layer 2 sets the ceiling on what Layer 3
can do; the rest is polish on top of that ceiling.

**Routed, multi-strategy retrieval — not one-size-fits-all RAG.**
The standard RAG pattern (embed everything, dense-retrieve, rerank) was
designed for open-domain corpora with no prior structure. We are in the
opposite situation: **Layer 1 already built exact, materialized facets**
(character_index, group_index, theme_index, UID → verse). Treating those as
a first-class retrieval lane rather than an afterthought to embeddings is
the central theoretical move of this layer.

Concretely, Layer 2 is a **router over four retrieval modes**, each matched
to a query shape:

| Query shape | Example | Retrieval mode | Cost |
|---|---|---|---|
| Structural | "B6_C27_S29" | Direct UID lookup | free |
| Entity/theme-constrained | "verses where Krishna speaks on dharma" | Layer 1 index intersection | free |
| Lexical | "verses containing 'chariot'" | BM25 over translations | cheap |
| Concept / paraphrase | "verses about the futility of grief" | Dense bi-encoder | medium |
| Mixed | most real queries | Hybrid fusion (RRF) | medium |

The theoretical frame is **faceted search + RAG**: use the cheapest mode
that can answer a given query, and only fall through to dense retrieval
when the query genuinely needs paraphrase handling. Layer 1 made the
structural lane free, so we should use it aggressively.

**Two-stage retrieval is the standard shape for the non-free modes.**
From Nogueira & Cho (2019), "Passage Re-ranking with BERT": stage 1 is a
recall-focused wide net (bi-encoder / BM25 / facet lookup, top 50-100),
stage 2 is a precision-focused cross-encoder that jointly encodes
(query, candidate) and scores the shortlist. Bi-encoders scale to the
corpus (precomputed, cheap per query); cross-encoders scale to the
shortlist (expensive per pair, but only k pairs). You get both efficiency
and quality. We adopt the pattern but treat stage 2 as optional and gated
on eval evidence.

**Hybrid fusion via Reciprocal Rank Fusion (RRF).**
When combining BM25 and dense rankings, the clean move is RRF
(Cormack, Clarke, Buettcher 2009):

    score(d) = Σᵢ 1 / (k + rankᵢ(d))

RRF is parameter-free (`k=60` is the literature default), robust across
wildly different score distributions, and well-studied. Learned fusion
(weighted combiners) would need labelled training data we don't have.
RRF is the YAGNI-correct choice until eval says otherwise.

### Design choices

#### Choice 1 — Route by query type, not one retriever for all

**What we're doing:** A top-level router inspects the query and dispatches
to whichever retrieval mode matches its shape. Structural queries never
touch embeddings. Entity-constrained queries hit Layer 1 indexes directly.
Only free-text queries reach BM25 / dense retrieval.

**Alternative considered:** A single hybrid retriever that always runs
all strategies and fuses — simpler code, but does unnecessary work on the
80% of queries that a Layer 1 lookup answers exactly.

**Why the router wins:** Cost asymmetry. Layer 1 lookups are O(1) and
exact; dense retrieval is O(corpus) with ranking noise. Running both
when the first is sufficient is strictly worse — more compute, more
chances to mis-rank the right answer. The router encodes the principle
"use the cheapest tool that works."

**Router implementation:** Hand-written rule system, not an LLM classifier.
Rules like: "query matches `B\d+_C\d+_S\d+` → structural"; "query contains
a known canonical from `entities.json` → facet lookup or hybrid with
entity constraint"; "else → free-text hybrid." An LLM router is a
tempting upgrade but introduces a model dependency at the top of the
pipeline, and rules are more than enough for the query shapes we have.
Revisit only if the rule set grows unwieldy.

#### Choice 2 — The verse is the atomic retrieval unit

**What we're doing:** Every retrieval mode indexes and returns at the
verse level, using the existing UIDs (e.g. `B6_C27_S29`).

**Why:** The corpus is already chunked for us. The source data is
verse-aligned, UIDs are stable, and Layer 1 tagged everything at the
verse level. Adopting a different retrieval granularity would fight the
corpus structure for no gain.

**Decision deferred:** a chapter-level secondary index. Useful for
queries like "what is the Drona Parva about?" — but buildable later as
a parallel index once eval shows verse-level retrieval underperforms on
that query shape. Don't build speculatively.

> **Outcome (2026-06-10): the deferral condition fired — chapter index now
> built.** The concept canaries gave the empirical proof this choice was
> waiting for: verse-level retrieval has a structural ceiling on reasoning/scene
> questions (the answer is a *scene*, not a verse). A chapter-dense index
> (`chapter_embeddings.npy`, 1,995×384 bge over the Step 6 summaries +
> `chapter_uids.txt`) localizes those scenes far better — `concept_003` → its
> chapter at rank #1, `concept_001` → rank #22 (vs verse rank ~7,000–53,000).
> Its home is **not** an always-on concept-lane blend (that trade was rejected,
> see Choice 6 outcome) but **Layer 3's context-fetch step** — coarse chapter
> localization feeding fine verse-gathering. See Layer 3 Choice 3.

#### Choice 3 — What text we embed per verse

**What we're doing:** The dense index embeds **the fluent English
translation only**. Queries are English-only for v1, so the embed target
is the same language as the query — no multilingual alignment needed.

**Why not also embed the Gemini summary or keywords?**
- The summary is a paraphrase of the translation; concatenating them
  double-counts the same semantic content and can *dilute* the signal
  rather than amplify it.
- The keywords field is already exploited by the structural lane via
  Layer 1 facets. Embedding it a second time is redundant work.
- Keeping the dense index to a single, well-understood field makes
  eval interpretable: if recall is bad, we know it's the translation's
  fault, not a concat artifact.

**User-facing vs retrieval text is a separate decision.** The corpus
also contains a Sanskrit-word-to-English gloss per verse (the literal
word-by-word translation). This is a **display option** for the user —
shown alongside results when they want to see the original phrasing —
but it is not part of the retrieval embedding input. Query in, English
fluent translation embedded, both translation and gloss returned for
display.

**What we give up:** Some concept-level queries may be better answered
by embedding the Gemini summary (which often states the verse's *theme*
more directly than the translation does). If eval exposes this gap,
we add a parallel summary index and fuse with RRF. Deferred, not
rejected.

#### Choice 4 — Embedding model: small, English, local

**What we're doing:** Start with `BAAI/bge-small-en-v1.5` (~130 MB, strong
English retrieval benchmark scores) or `intfloat/e5-small-v2` as the
bi-encoder. Both run on CPU in reasonable time and produce 384-dim
vectors — our corpus is 73,820 verses, so the full matrix is ~110 MB
of float32 (or half that at fp16), trivially held in RAM.

**Why not multilingual?**
All of our retrieval targets are English (the fluent translation), and
queries are English. A multilingual model would spend capacity
representing languages we don't query in, for no benefit. If we later
want Sanskrit-side queries ("find verses containing *yoga*"), we add a
multilingual index then — a hedge that pays nothing today.

**Why not a larger model?**
`bge-large` or `e5-large` (~1.3 GB) score a few points higher on
benchmarks but triple the RAM and inference cost. At our corpus size
and query volume, the small variants are the right point on the
quality/cost curve. Upgrade only if eval shows recall is the
bottleneck.

#### Choice 5 — Dense index: numpy brute force, not FAISS

**What we're doing:** Store verse embeddings as a single `(N, D)` numpy
matrix on disk, load to RAM at query time, compute cosine similarity
against the query vector with a single matmul, argsort for top-k.

**Alternative considered:** FAISS with IVF or HNSW indexing.

**Why numpy wins at our scale:**
- 73,820 verses × 384 dims × 4 bytes = ~110 MB. Fits in RAM with room
  to spare.
- Brute-force cosine is one matmul. Measured latency on CPU at this
  size is well under 100 ms per query, which is imperceptible for a
  research tool.
- FAISS is an ANN library — it trades exactness for speed. At our
  scale, we don't need the speed and we don't want to pay the
  exactness cost (or the dependency weight).
- Numpy is already a transitive dep of sentence-transformers, so it
  costs us nothing additional.

**When we'd reconsider:** If the corpus grows ~10× (which it won't —
it's a closed text) or if we index chunks much finer than verses
(which we won't unless eval forces it).

#### Choice 6 — Hybrid fusion via RRF (if needed)

**What we're doing:** For free-text queries, run BM25 and dense
retrieval in parallel, fuse via Reciprocal Rank Fusion with `k=60`.

**Why not a weighted combiner?** It needs training data — labelled
query/relevance pairs — which we don't have. RRF is parameter-free
and the literature shows it's competitive with tuned combiners when
both input rankers are reasonable. Classic case of a simpler method
being the right one until proven otherwise.

**When we'd reconsider:** If eval shows one ranker dominates and the
fusion is just adding noise, we drop the weaker one. If both contribute
but RRF under-weights the stronger signal, we could try a learned
combiner once the eval set is large enough.

> **Outcome (2026-06-08): RRF built, tested, REJECTED — the prediction was
> wrong.** This choice argued RRF was the YAGNI-correct default; eval overruled
> it. Equal-weight RRF dropped concept mean recall@10 from 0.312 (dense-only) to
> 0.175 — BM25 scores ~0 on paraphrastic queries, and fusing on *rank position*
> interleaves BM25's wrong top hits with dense's correct ones. No weighting
> (dense:bm25 up to 10:1) beat dense-only; weighting only crawled back *toward*
> it. The "reconsider" clause fired: a 7-item lexical eval showed the mirror
> image (BM25 r@10 0.619 vs dense 0.167 — BM25 ~4×). The two lanes aren't
> strong-vs-weak; each is ~4× better on *its own* shape, so fusing dilutes
> whichever is right. **The fix is route-by-shape, not fuse:** the router gained
> a fifth mode, `lexical`, gated by **explicit double-quotes** (`"iron mace"`) —
> bare phrases are syntactically identical whether lexical or concept intent, so
> intent is made an explicit signal rather than guessed. Precedence: UID > slice
> > quoted-lexical > question-shape concept > high-coverage facet > concept.
> `fusion.py` stays in the tree for a future genuine "mixed" route but is off the
> default path. New baseline: **49/52** (lexical 7/7, facet 25/25, concept
> 12/15). This is the canonical example of test-set-first overruling a
> literature-default prior.

#### Choice 7 — Cross-encoder reranker is gated on evidence, not built upfront

**What we're doing:** The hybrid retriever (choices 1-6) is the v1
shipping target. A cross-encoder reranker (~80 MB, e.g.
`cross-encoder/ms-marco-MiniLM-L-6-v2`) is on the roadmap *only* as a
second phase, and *only* if the eval set shows stage 1 recall is
adequate but top-k precision is not.

**Why gate it:** A reranker is the most expensive component per query
(runs a transformer forward pass per candidate) and the least likely to
help if the stage-1 retriever is already returning good candidates. The
right decision procedure is: measure stage 1 first, then decide.
Building it speculatively contradicts the critical-path discipline we
used in Layer 1.

> **Outcome (2026-06-10): not built — ruled out by inference for the canaries.**
> The gate stayed shut. A reranker only reorders a ~top-200 candidate pool, but
> the canary answer-verses sit at verse-dense rank **7,000–53,000**. A reranker
> cannot recover what stage 1 never surfaced — the canary problem is **recall,
> not ranking**. So a cross-encoder is the wrong tool *for this failure*; it
> remains a live option if eval later shows a recall-adequate-but-precision-poor
> query shape. The canaries' recall ceiling is what motivated the chapter index
> + Layer 3 instead (see Choice 2 outcome).

#### Choice 8 — Build the eval set before (not after) the retriever

**What we're doing:** Hand-crafting a small eval set of ~30-50 queries,
each tagged with its query shape (structural / facet / lexical /
concept / mixed) and with a known-good set of verse UIDs. Measure
`recall@k` and `MRR@k` on every retriever variant.

**Bootstrap strategy:** Use the local Ollama model (`llama3-8k:latest`)
to draft a first-pass query set from Layer 1 artifacts — e.g. feed it
character/theme combinations and ask for plausible queries — then
hand-edit for correctness and known-good UIDs. This is faster than
writing from scratch and more honest than trusting the model's output
verbatim. Local Ollama keeps it free and reproducible.

**Why eval before building:** Without a measurement contract, we can't
honestly compare BM25 vs dense vs hybrid, can't decide whether to add
the reranker, and can't tell regressions from improvements. This
mirrors the Layer 1 discipline of "validate by cross-checking counts
before shipping the pass."

**Theoretical note:** This is the **test-set-first** discipline from
classical IR evaluation (TREC tradition). The point isn't that the
eval set is perfect — it's that having *any* quantitative contract
forces honest comparisons and prevents "it feels better" reasoning.

### Build steps (dependency order)

```
Phase A — Structural retrieval (zero new deps)
  Step 1: Query router scaffold + structural lookup + facet lookup
          (uses only Layer 1 indexes — shippable immediately)

Phase B — Eval infrastructure
  Step 2: Bootstrap eval set via local Ollama (llama3-8k:latest)
  Step 3: Hand-edit to ~30-50 queries × known-good UIDs
  Step 4: Retrieval eval harness (recall@k, MRR@k)

Phase C — Free-text retrieval
  Step 5: BM25 over English translations (rank_bm25, pure Python)
  Step 6: Dense retrieval — embed translations with bge-small-en-v1.5,
          store as numpy (N, 384) matrix
  Step 7: Hybrid fusion via RRF

Gated / deferred
  Step 8: Cross-encoder reranker — only if Step 7 eval demands it
  Step 9: Summary-field parallel dense index — only if translation
          embeddings miss concept queries
  Step 10: Chapter-level secondary index — only if eval surfaces the need
  Step 11: Multilingual/Sanskrit-side retrieval — only when a Sanskrit
           query use case appears
```

Steps 1-4 give a measurable retriever with zero ML dependencies.
Steps 5-7 add ML only after we can measure its payoff. Steps 8-11 are
strictly opportunistic — we build them only when the eval set tells us
they're needed.

### Tradeoffs

- **Toolchain growth.** Dense retrieval introduces a
  `sentence-transformers` + `torch` dependency the project currently
  doesn't have. Still local, still CPU, still in the `.venv`, but the
  dep graph roughly doubles in weight. Worth it because concept-paraphrase
  queries are a genuine weakness of the pure-facet lane.
- **Router as a new failure mode.** If the router mis-classifies a
  query, the right retrieval mode never runs and no amount of good
  ranking recovers it. Mitigation: the router is itself testable
  against the eval set — we can measure routing accuracy as a first-class
  metric and catch regressions before they hit retrieval recall.
- **RRF is a blunt hybrid.** We give up potential quality from a tuned
  combiner, but save ourselves the training-data problem. Revisit only
  when eval says the leakage matters.
- **Eval set is small.** 30-50 queries is enough to catch gross
  regressions and order methods, not enough for fine-grained claims.
  That's the right size for a first pass; we grow it over time as
  real usage exposes query shapes we didn't anticipate.

### Open questions for Layer 2

- **Speaker-filtered queries** ("what did Krishna *say*") are not
  cleanly answerable with the current facets — verse tagging is
  "mentioned," not "spoken by." Deferred from Layer 1; if eval
  exposes this as a common query shape, we revisit whether to build
  a speaker index.
- **Multi-hop queries** ("who killed Bhishma and why") need retrieval
  *plus* reasoning — partly a Layer 3 concern, but Layer 2 needs to
  return enough candidates to support the reasoning step. Open
  question: what's the right k for multi-hop, and should the router
  detect them?
- **Query rewriting / HyDE.** Hypothetical Document Embeddings
  (Gao et al. 2022) improve dense retrieval by asking an LLM to
  generate a fake answer and embedding *that* instead of the query.
  Strong in the literature but adds an LLM dependency at query time.
  Deferred unless eval shows dense retrieval is underperforming
  specifically because query/doc distribution mismatch.
- **Eval set drift.** As we change the retriever, the eval set starts
  to co-evolve with the thing it's meant to measure. Standard IR
  trap. Mitigation: keep a frozen subset (~10 queries) that never
  gets edited once written, as a regression anchor.

---

## Layer 3 — Synthesis

> Status note (2026-06-17): this section is written *before* the build, in the
> project's test-set-first discipline — the design contract comes first, code
> second. Choices here are the plan; outcome callouts will be appended as eval
> evidence arrives, the same way Layer 2's RRF prediction was later overturned.

### Theoretical basis

**Synthesis is reading comprehension over retrieved context — not parametric
recall.**
This is the same materialized-views thesis that governs Layer 1, applied one
level up. Layer 1 said: don't make a model *store* facts you can keep on disk.
Layer 3 says: don't make a model *know* the Mahabharata — make it *read* the
verses Layer 2 just handed it and assemble an answer. The model's competence we
pay for is reading comprehension and faithful composition, not knowledge. This
is the exact inversion of the previous project (Mahabharat_Python_Test), which
fine-tuned an 8B model to "know" the epic and got format imitation plus
hallucination for its trouble. Knowledge lives in the corpus; reasoning is
rented per query and discarded.

**Layer 2 is the ceiling; Layer 3 is what you build on it.**
Restating the Layer 2 basis: "if retrieval quietly filters out the right verse,
no amount of synthesis cleverness recovers it." Layer 3 cannot exceed the
candidate set it is given. This is not a caveat — it is the central design
constraint. Every Layer 3 decision is downstream of "what context did retrieval
localize," which is why the synthesis layer is co-designed with the chapter-dense
index, not bolted onto verse retrieval.

**The empirical motivation: the canaries proved a structural ceiling.**
The 2026-06-10 arc established that the three reasoning/scene canaries
(`concept_001` why Arjuna refuses to fight, `concept_003` Draupadi's disrobing,
`concept_006` Bhishma's vow) are *not* fixable at the retrieval layer. Four
fixes were built and rejected; the diagnosis was that "why did X…" / "what did X
feel…" questions have answers that are *spans reasoned across a scene*, not a
single verse a top-10 metric can pin. The conclusion was explicit: these belong
to synthesis. Layer 3 exists precisely to convert that retrieval ceiling into a
reasoning problem a small model can solve over a localized chapter.

**RAG, strictly — and grounded RAG specifically.**
Layer 3 is retrieval-augmented generation in its strict form: the model sees
only the retrieved context and must answer from it, cite the verse UIDs it used,
and abstain when the answer is not present. The anti-hallucination contract is
not a nicety; it is the whole reason this architecture beats parametric storage.
A grounded answer with citations is *verifiable* — the user (or an eval) can
open the cited UID and check. A parametric answer is not.

### Design choices

#### Choice 1 — RAG over parametric: the model reads, never recalls

**What we're doing:** The synthesis model is prompted with retrieved context and
instructed to answer *only* from it, cite the verse UIDs it draws on, and say
"not found in the retrieved verses" rather than fill the gap from training data.

**Alternative considered:** Fine-tune a model on the corpus so it can answer
from parameters. This is exactly what the previous project did and exactly why
this one exists. Parametric storage is lossy, unverifiable, and hallucination-
prone for a corpus we already hold exactly on disk.

**Why grounded RAG wins:** It makes the cardinal failure (hallucination) into a
*checkable* event — a cited UID either supports the claim or it doesn't. It
keeps knowledge in the corpus where it is exact and editable. And it lets a
small model punch above its weight, because reading 5–15 supplied verses is a
far easier task than recalling the right ones from 73,820 memorized ones.

#### Choice 2 — Conditional invocation: most queries never reach Layer 3

**What we're doing:** Synthesis is the most expensive lane (an LLM forward pass,
nondeterministic, hundreds of ms to seconds) and fires *last and least*.
Structural (UID/slice), facet, and lexical queries are already answered
exactly by Layer 1/2 lookups — they return verses directly, no synthesis. Layer
3 is invoked only for the query shapes that genuinely need an *assembled* answer:
the reasoning/scene questions the canaries exemplify.

**Why:** This is the same cost-asymmetry discipline as the Layer 2 router —
"use the cheapest tool that works." A facet query like `Krishna and yoga` wants
a verse list, not a paragraph; running a model over it adds latency, nondeterm-
inism, and a hallucination surface for zero benefit.

**The hard part — and it is genuinely hard:** the router *cannot reliably tell*
a synthesis-needing reasoning question from a plain concept lookup. This is the
same lexical-vs-concept ambiguity that forced explicit quote-gating in Layer 2:
"Why did Arjuna refuse to fight?" and "verses about refusing battle" are
syntactically near-identical but want different machinery. v1 resolves this the
honest way (see Open Questions): synthesis is **opt-in** behind an explicit flag
/ mode rather than auto-detected, until eval shows a heuristic that beats the
ambiguity. We do not repeat the mistake of an always-on combiner that helps the
reasoning case and dilutes everything else — that is exactly the trade that sank
RRF and the verse/chapter blend.

#### Choice 3 — Context assembly: chapter-localize, then verse-gather

**What we're doing:** Two-stage context fetch, reusing artifacts that already
exist. Stage 1 (coarse): the **chapter-dense index**
(`chapter_embeddings.npy`, built 2026-06-10) ranks the 1,995 chapter summaries
against the query and localizes the scene — this is where it earns its keep
(`concept_003` → chapter B2_C62 at rank #1). Stage 2 (fine): pull the localized
chapter's `chapter_summaries.jsonl` digest plus the top verse-level dense hits
*within or near* that chapter, with fluent translations (and Sanskrit gloss
available for display). That bundle is the model's context window.

**Why coarse-to-fine:** It is the standard hierarchical-retrieval shape, and it
is exactly what the canary arc validated — chapter retrieval recovers scenes
that verse retrieval buries at rank 7,000–53,000. The chapter localizes "where
in the epic this happens"; the verses supply the quotable ground truth the model
must cite. Neither alone is sufficient: chapter-only loses verse-level citation,
verse-only loses the scene.

**What we give up:** `concept_006`'s chapter (B1_C94) still ranks ~1,016, likely
because bge truncates the long summary at 512 tokens. Chapter-summary chunking
is the known fix and is deferred to Open Questions — v1 tolerates it because
synthesis degrades gracefully (a near-miss chapter still narrows context vs the
whole corpus) where the strict retrieval metric did not.

> **Outcome (2026-06-17): chunking implemented — chapter recall doubled.** Each
> chapter summary is now split into ≤1,200-char chunks on verse boundaries (1,995
> chapters → 8,711 chunks, 4.37 avg), each embedded separately; a chapter ranks
> by its best chunk (max-pool, `layer2.chapter_dense.ChapterRetriever`), and the
> *matching chunk* — not a head-truncated summary — is what the model is shown.
> On the L3 eval this doubled mean chapter context recall (0.071 → 0.148) and
> lifted combined recall 0.301 → 0.355 and citation precision 0.249 → 0.285
> (verse recall unchanged at 0.247, confirming the gain is the chapter lane).
> Standout: `concept_003` chapter recall 0.38 → 0.88. **But `concept_006` was
> *not* rescued** — B1_C94 (Bhishma's vow) still doesn't localize even chunked,
> so chunking is necessary-not-sufficient; some scenes need a better localizer
> signal than the naive-rollup summary provides.

#### Choice 4 — Small instruct model, prompted before fine-tuned

**What we're doing:** Start with an off-the-shelf small instruction-tuned model
(1–3B class — Qwen2.5-1.5B/3B-Instruct, Phi-3 Mini, Llama-3.2-3B are the
candidates), served locally via **Ollama** (already in the stack, already used
for the eval bootstrap), prompted with the assembled context. **Zero training in
v1.**

**Alternative considered (and deferred, not rejected):** Fine-tuning on hard
tasks — character-arc queries, multi-verse reasoning, comparative questions.
This is the roadmap's eventual ambition, but the previous project's scar is
precisely *premature fine-tuning*. The YAGNI/critical-path discipline says:
prove a prompted base model can't assemble grounded answers *before* paying for
training data, compute, and a serving pipeline. Fine-tuning is gated on eval
evidence, exactly like the cross-encoder in Layer 2.

**Why small is the right starting point:** Synthesis-over-supplied-context is a
fundamentally easier task than open-domain QA — the model is not asked to recall,
only to read 5–15 verses and compose faithfully. Task difficulty, not corpus
size, sets the model-size floor, and reading comprehension over a short context
is squarely within small-model range. Upgrade only if eval shows the small model
mangles multi-verse reasoning.

#### Choice 5 — Grounding via mandatory UID citation + abstention

**What we're doing:** The output contract requires every substantive claim to
cite the verse UID(s) supporting it, and requires explicit abstention ("the
retrieved verses don't answer this") over fabrication. Generation runs at low
temperature for reproducibility.

**Why:** Citation turns the answer into something checkable and converts
hallucination from an invisible failure into a detectable one (a cited UID whose
translation doesn't support the claim). This is the single most important design
lever against the previous project's failure mode, and it is what makes the
synthesis layer trustworthy enough to build on.

#### Choice 6 — Evaluation without gold answers: faithfulness + citation precision + context recall

**The problem:** Layers 1–2 could be evaluated against known-good UIDs because
retrieval has a set-membership ground truth. Synthesis produces free text — there
is no single gold paragraph, and writing one per question both doesn't scale and
co-evolves with the system it measures.

**What we're doing:** Decompose answer quality into measurable, mostly
reference-free components, in the RAGAS tradition:
- **Faithfulness / groundedness** — does every claim follow from the cited
  context? Scored by an LLM-as-judge (a *different* model than the synthesizer,
  to limit the obvious circularity), and partially mechanizable by checking that
  cited UIDs exist and were in the supplied context.
- **Citation precision** — are the cited UIDs actually relevant? This is
  checkable against the curated `known_good_uids` we already maintain. The three
  canaries become the **first synthesis eval items**: they already have curated
  answer-verse spans (B6_C24; B2_C60/62/72; B1_C94).
- **Context recall** — did retrieval supply the verses needed to answer? This is
  just the existing Layer 2 `recall@k` re-used as an upstream gate; it cleanly
  attributes failures to the right layer (bad retrieval vs bad synthesis).

**Why this shape:** It keeps the test-set-first discipline alive into a layer
that resists gold labels, and it preserves layer attribution — the single most
useful property of the existing eval harness. An answer can fail because
retrieval missed (context recall low) or because the model hallucinated despite
good context (faithfulness low); these demand different fixes and the metrics
keep them separate.

### Tradeoffs

- **First nondeterministic runtime path.** Every prior lane (UID, slice, facet,
  BM25, dense) is deterministic and instantly debuggable. An LLM in the hot path
  introduces latency, run-to-run variance, and a hallucination surface.
  Mitigations: conditional invocation (most queries never pay it), low
  temperature, mandatory citation, and abstention.
- **LLM-as-judge is circular and noisy.** Using a model to grade a model is a
  known weak measurement. Mitigation: judge with a different/larger model,
  cross-check citation precision against the curated UIDs (which is *not*
  model-graded), and treat judge scores as trend signal, not ground truth.
- **A local 1–3B model may simply not reason well enough.** The fallback
  (fine-tuning, or a larger local model) is more expensive on every axis. We
  accept the risk because the alternative — assuming we need it — is the exact
  premature-scaling error the project was founded to avoid.
- **Toolchain growth, again.** Layer 3 adds a generation dependency (Ollama
  serving + a prompt/orchestration module) on top of the retrieval stack. Still
  local, still CPU/Metal, still free per query.

### Open questions for Layer 3

- **When to invoke synthesis automatically.** v1 is opt-in by flag. Can a cheap
  signal (question-form + low facet-coverage + the dense score *distribution*,
  e.g. a flat top-k implying "no single verse answers this") detect
  reasoning-shaped queries reliably enough to auto-route? This is the live
  research question; it is the same ambiguity that quote-gating dodged rather
  than solved.
- **Fine-tune target, if eval demands it.** Which hard tasks (character arcs,
  multi-hop, comparison)? What training data — can we bootstrap synthetic
  (question, grounded-answer, cited-UIDs) triples from the corpus itself, the way
  the eval set was bootstrapped? And how do we avoid re-introducing format
  imitation?
- **Agentic / multi-hop retrieval loop.** "Who killed Bhishma and why" needs the
  model to *request more context* after a first pass. Should Layer 3 be a single
  shot over fixed context, or a loop where the model can call back into Layer 2?
  Single-shot for v1; the loop is the natural extension once single-shot is
  measured.
- **Chapter-summary chunking.** ~~`concept_006`'s long summary is truncated at
  bge's 512 tokens, sinking its chapter rank.~~ DONE 2026-06-17 (see Choice 3
  outcome): chunk + max-pool doubled chapter recall, but did *not* rescue
  `concept_006`. Remaining lever there: the naive-rollup summary may be a weak
  localizer signal for that scene — candidates are an LLM-written chapter
  summary (Layer 1 Choice 4's deferred upgrade), chunk overlap, or embedding the
  verse-translation text of the chapter rather than the Gemini summary.
- **Authoring reasoning-question ground truth without co-evolution.** The
  classic IR eval-drift trap, now sharper because synthesis answers are open-
  ended. Mitigation mirrors Layer 2: freeze a small reasoning-question anchor
  set (the canaries are the seed) and never edit it once written.

---

---

## Layer 4 — Translation Model

*To be written when we reach this layer. Expected topics: seq2seq for
low-resource parallel corpora, mBART vs NLLB tradeoffs, training data size
vs quality, evaluation without gold references.*
