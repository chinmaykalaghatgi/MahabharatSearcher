# Layer 3 — Synthesis Eval Run

**Timestamp:** 2026-06-17T05:57:29+00:00  
**Eval set:** `data/layer2/eval/eval_set.jsonl` (concept subset, 15 items)  
**Model:** llama3-8k:latest  
**Context:** top-3 chapters + top-8 verses  
**Faithfulness:** not judged this run (LLM-judge off — the only local model is the synthesizer, so judging is circular).  

## Summary

- **Overall:** 14/15 pass (93.3%)
- **Abstain rate:** 6.7%
- **Context recall (combined, mean):** 0.355 — via verses 0.247, via chapters 0.148
- **Items with any answer-context localized:** 13/15
- **Citation precision (mean):** 0.285

### Pass-reason breakdown

- `correct_abstain`: 1
- `grounded_answer`: 13
- `should_have_abstained`: 1

## Per-item results

| ID | Query | Pass | Reason | ctx-recall | cite-prec | chapters localized |
|---|---|---|---|---:|---:|---|
| concept_001 | Why did Arjuna refuse to fight at Kurukshetra? | pass | correct_abstain | 0.00 | 0.00 | B7_C157, B4_C36, B14_C76 |
| concept_002 | How does fate shape the course of events in the Mahabharata? | pass | grounded_answer | 0.43 | 0.20 | B6_C3, B7_C127, B8_C5 |
| concept_003 | What did Draupadi feel when she was disrobed in the Kuru assembly? | pass | grounded_answer | 0.88 | 0.00 | B2_C62, B3_C13, B2_C60 |
| concept_005 | How did Yudhishthira come to be king as the eldest Pandava? | pass | grounded_answer | 0.33 | 0.25 | B2_C42, B2_C22, B14_C3 |
| concept_006 | What motivated Bhishma to surrender his throne and worldly desires? | **FAIL** | should_have_abstained | 0.00 | 0.00 | B5_C145, B6_C95, B1_C95 |
| concept_007 | Why should we not mourn the dead? | pass | grounded_answer | 0.38 | 0.75 | B1_C145, B12_C149, B5_C71 |
| concept_008 | When is killing in battle justified? | pass | grounded_answer | 0.17 | 0.10 | B6_C1, B12_C96, B12_C97 |
| concept_009 | What happens to the soul at the moment of death? | pass | grounded_answer | 0.50 | 0.80 | B12_C286, B14_C17, B12_C179 |
| concept_010 | What is the duty owed to one's teacher? | pass | grounded_answer | 0.14 | 0.50 | B12_C234, B12_C109, B5_C29 |
| concept_011 | What is the cost of unchecked anger? | pass | grounded_answer | 0.71 | 0.33 | B3_C30, B2_C57, B12_C115 |
| concept_012 | Speaking truth even when it is costly | pass | grounded_answer | 0.38 | 0.25 | B12_C192, B12_C259, B12_C110 |
| concept_013 | Friendship that endures hardship and reversal | pass | grounded_answer | 0.50 | 0.67 | B12_C136, B5_C39, B12_C137 |
| concept_014 | What is the mark of true contentment? | pass | grounded_answer | 0.25 | 0.20 | B12_C343, B14_C38, B3_C35 |
| concept_015 | How should a king treat his subjects? | pass | grounded_answer | 0.38 | 0.22 | B5_C85, B12_C69, B12_C103 |
| concept_016 | The cost of seeking revenge | pass | grounded_answer | 0.29 | 0.00 | B12_C137, B12_C138, B12_C58 |
