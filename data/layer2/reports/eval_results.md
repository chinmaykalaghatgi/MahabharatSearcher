# Layer 2 Phase A — Eval Run

**Timestamp:** 2026-06-08T07:36:06+00:00  
**Eval set:** `data/layer2/eval/eval_set.jsonl` (52 items)  
**Retriever:** Phase A + C — route-by-shape. Facet results are corpus-ordered; lexical (quoted) queries are ranked by BM25 (Step 5); concept queries by dense bge-small (Step 6).

## Summary

- **Overall:** 49/52 pass (94.2%)
- **Router accuracy:** 100.0%
- **Union-fallback fired:** 0 item(s)

### Pass rate by shape

| Shape | Pass | Total | Rate |
|---|---:|---:|---:|
| structural_uid | 3 | 3 | 100.0% |
| structural_slice | 2 | 2 | 100.0% |
| lexical | 7 | 7 | 100.0% |
| facet | 25 | 25 | 100.0% |
| concept | 12 | 15 | 80.0% |

### Facet metrics (25 items with known_goods)

- Mean recall@10: 1.000
- Mean recall@20: 1.000
- Mean MRR: 0.651
- Facet-resolution accuracy: 100.0%
- Known-goods reachable in full result set: 100.0%

### Lexical (BM25) metrics (7 items with known_goods)

- Mean recall@10: 0.619
- Mean recall@20: 0.857
- Mean MRR: 0.592
- Known-goods reachable in BM25 top-K: 100.0%

### Concept (dense) metrics (15 items with known_goods)

- Mean recall@10: 0.312
- Mean recall@20: 0.413
- Mean MRR: 0.511
- Known-goods reachable in dense top-K: 26.7%

### Frozen subset

- 10/10 pass (100.0%)

## Per-item results

### structural_uid

| ID | Query | Pass | Router | Notes |
|---|---|---|---|---|
| struct_uid_001 | `B1_C1_S1` | pass | structural_uid | — |
| struct_uid_002 | `B6_C27_S29` | pass | structural_uid | — |
| struct_uid_003 | `B12_C1_S1` | pass | structural_uid | — |

### structural_slice

| ID | Query | Pass | Total | Router | Notes |
|---|---|---|---:|---|---|
| struct_slice_001 | `B6_C25` | pass | 43 | structural_slice | — |
| struct_slice_002 | `B17_C1` | pass | 44 | structural_slice | — |

### lexical (BM25)

| ID | Query | Pass | r@10 | r@20 | MRR | Total | Notes |
|---|---|---|---:|---:|---:|---:|---|
| lex_001 | "house of lac" | pass | 0.33 | 1.00 | 0.33 | 402 | missing=B1_C132_S10,B1_C134_S13 |
| lex_002 | "churning of the ocean" | pass | 0.33 | 1.00 | 1.00 | 784 | missing=B1_C16_S12,B1_C16_S31 |
| lex_003 | "white parasol" | pass | 1.00 | 1.00 | 1.00 | 375 | — |
| lex_004 | "war drum" | pass | 1.00 | 1.00 | 1.00 | 481 | — |
| lex_005 | "Panchajanya conch" | pass | 0.33 | 0.33 | 0.50 | 178 | missing=B5_C149_S65,B7_C3_S19 |
| lex_006 | "iron mace" | pass | 1.00 | 1.00 | 0.14 | 706 | — |
| lex_007 | "jackals howled" | pass | 0.33 | 0.67 | 0.17 | 86 | missing=B1_C1_S129,B2_C63_S22 |

### facet

| ID | Query | Pass | r@10 | r@20 | MRR | Total | Resolved | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| facet_001 | Karna and friendship | pass | 1.00 | 1.00 | 1.00 | 23 | C:[Karna] T:[friendship] | — |
| facet_002 | Draupadi and women | pass | 1.00 | 1.00 | 0.25 | 21 | C:[Draupadi] T:[women] | — |
| facet_003 | Bhishma teaches about duty | pass | 1.00 | 1.00 | 1.00 | 10 | C:[Bhishma] T:[duty] | — |
| facet_004 | Krishna and yoga | pass | 1.00 | 1.00 | 0.33 | 13 | C:[Krishna] T:[yoga] | — |
| facet_005 | Yudhishthira and truth | pass | 1.00 | 1.00 | 0.33 | 47 | C:[Yudhishthira] T:[truth] | — |
| facet_006 | Drona and teachers | pass | 1.00 | 1.00 | 1.00 | 41 | C:[Drona] T:[teachers] | — |
| facet_007 | Karna and pride | pass | 1.00 | 1.00 | 0.33 | 12 | C:[Karna] T:[pride] | — |
| facet_008 | Krishna and liberation | pass | 1.00 | 1.00 | 1.00 | 14 | C:[Krishna] T:[liberation] | — |
| facet_009 | Bhishma and time | pass | 1.00 | 1.00 | 0.25 | 14 | C:[Bhishma] T:[time] | — |
| facet_010 | Krishna and divinity | pass | 1.00 | 1.00 | 0.20 | 25 | C:[Krishna] T:[divinity] | — |
| facet_011 | Sanjaya gives counsel | pass | 1.00 | 1.00 | 0.50 | 9 | C:[Sanjaya] T:[counsel] | — |
| facet_012 | Vidura on truth | pass | 1.00 | 1.00 | 1.00 | 14 | C:[Vidura] T:[truth] | — |
| facet_013 | Bhima on the battlefield | pass | 1.00 | 1.00 | 1.00 | 22 | C:[Bhima] T:[battlefield] | — |
| facet_014 | Drona and pride | pass | 1.00 | 1.00 | 1.00 | 5 | C:[Drona] T:[pride] | — |
| facet_015 | Yudhishthira in assembly | pass | 1.00 | 1.00 | 0.50 | 46 | C:[Yudhishthira] T:[assembly] | — |
| facet_016 | Yudhishthira and Vidura | pass | 1.00 | 1.00 | 1.00 | 55 | C:[Vidura, Yudhishthira] | — |
| facet_017 | Bhima and Hidimba | pass | 1.00 | 1.00 | 1.00 | 36 | C:[Bhima, Hidimba] | — |
| facet_018 | Drona and Drupada | pass | 1.00 | 1.00 | 0.33 | 65 | C:[Drona, Drupada] | — |
| facet_019 | Krishna and Arjuna on dharma | pass | 1.00 | 1.00 | 1.00 | 29 | C:[Arjuna, Krishna] T:[dharma] | — |
| facet_020 | Yudhishthira and Krishna on dharma | pass | 1.00 | 1.00 | 0.50 | 26 | C:[Krishna, Yudhishthira] T:[dharma] | — |
| facet_021 | Drona and Arjuna teachers | pass | 1.00 | 1.00 | 1.00 | 12 | C:[Arjuna, Drona] T:[teachers] | — |
| facet_022 | Bhima and Hidimba in battle | pass | 1.00 | 1.00 | 0.50 | 6 | C:[Bhima, Hidimba] T:[battle] | — |
| facet_023 | Pandavas in hermitage | pass | 1.00 | 1.00 | 0.25 | 31 | G:[Pandavas] T:[hermitage] | — |
| facet_024 | Kauravas in war | pass | 1.00 | 1.00 | 0.50 | 71 | G:[Kauravas] T:[war] | — |
| facet_025 | Brahmins and charity | pass | 1.00 | 1.00 | 0.50 | 68 | G:[Brahmins] T:[charity] | — |

### concept (dense)

| ID | Query | Pass | r@10 | r@20 | MRR | Total | Notes |
|---|---|---|---:|---:|---:|---:|---|
| concept_001 | Why did Arjuna refuse to fight at Kurukshetra? | **FAIL** | 0.00 | 0.00 | 0.00 | 10000 | missing=B6_C24_S4,B6_C24_S5,B6_C24_S7,B6_C24_S6,B6_C24_S8,B6_C24_S9; not in full result set |
| concept_002 | How does fate shape the course of events in the Mahabharata? | pass | 0.43 | 0.57 | 1.00 | 10000 | missing=B6_C103_S92,B7_C57_S6,B7_C158_S7,B11_C25_S30 |
| concept_003 | What did Draupadi feel when she was disrobed in the Kuru assembly? | **FAIL** | 0.00 | 0.12 | 0.09 | 10000 | missing=B2_C60_S36,B2_C62_S3,B2_C72_S17,B2_C60_S21,B2_C60_S28,B2_C60_S46,B2_C62_S5,B2_C62_S8; not in full result set |
| concept_005 | How did Yudhishthira come to be king as the eldest Pandava? | pass | 0.33 | 0.33 | 0.33 | 10000 | missing=B1_C200_S6,B1_C200_S5,B1_C200_S7,B1_C200_S8; not in full result set |
| concept_006 | What motivated Bhishma to surrender his throne and worldly desires? | **FAIL** | 0.00 | 0.00 | 0.00 | 10000 | missing=B1_C94_S86,B1_C94_S93,B1_C97_S15,B1_C94_S77,B1_C94_S85,B1_C94_S87,B1_C94_S88; not in full result set |
| concept_007 | Why should we not mourn the dead? | pass | 0.50 | 0.62 | 1.00 | 10000 | missing=B6_C24_S11,B6_C24_S22,B6_C24_S25,B6_C24_S28 |
| concept_008 | When is killing in battle justified? | pass | 0.17 | 0.17 | 0.20 | 10000 | missing=B6_C24_S31,B6_C24_S32,B6_C24_S37,B6_C24_S33,B6_C24_S38; not in full result set |
| concept_009 | What happens to the soul at the moment of death? | pass | 0.50 | 0.50 | 1.00 | 10000 | missing=B6_C30_S5,B6_C30_S6,B6_C30_S13,B6_C30_S10; not in full result set |
| concept_010 | What is the duty owed to one's teacher? | pass | 0.29 | 0.57 | 0.14 | 10000 | missing=B1_C123_S1,B1_C123_S35,B1_C123_S34,B5_C44_S8,B1_C86_S2; not in full result set |
| concept_011 | What is the cost of unchecked anger? | pass | 0.43 | 0.71 | 0.50 | 10000 | missing=B6_C24_S62,B6_C24_S63,B3_C30_S4,B3_C30_S6 |
| concept_012 | Speaking truth even when it is costly | pass | 0.38 | 0.62 | 1.00 | 10000 | missing=B3_C49_S27,B5_C146_S13,B5_C186_S26,B1_C187_S6,B2_C61_S68 |
| concept_013 | Friendship that endures hardship and reversal | pass | 0.50 | 0.67 | 0.25 | 10000 | missing=B2_C18_S14,B3_C120_S7,B12_C136_S54; not in full result set |
| concept_014 | What is the mark of true contentment? | pass | 0.38 | 0.38 | 1.00 | 10000 | missing=B6_C24_S55,B6_C24_S56,B6_C24_S70,B6_C24_S57,B6_C24_S71; not in full result set |
| concept_015 | How should a king treat his subjects? | pass | 0.50 | 0.50 | 1.00 | 10000 | missing=B1_C94_S2,B1_C94_S6,B1_C94_S14,B1_C94_S16; not in full result set |
| concept_016 | The cost of seeking revenge | pass | 0.29 | 0.43 | 0.14 | 10000 | missing=B1_C155_S1,B2_C71_S38,B1_C155_S21,B1_C155_S33,B12_C137_S50; not in full result set |
