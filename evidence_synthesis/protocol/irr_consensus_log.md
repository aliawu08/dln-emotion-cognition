# IRR Consensus Log

## Date: 2026-02-22

## Participants
- Author (Ali Wu)
- Coder 1 (Yuanxi Jiang) — coded 2026-02-21
- Coder 2 (Ang Li) — coded 2026-02-22

## Hoyt (2024) — Health Outcome Domains
**No consensus needed.** All 3 pairwise comparisons yielded perfect agreement (k = 1.000, 8/8 items).

## Webb (2012) — Emotion Regulation Strategies
Initial independent coding produced 4 disagreement items (all adjacent-category, 1-step). Consensus resolved as follows:

| Item | Author (initial) | Coder 1 | Coder 2 | Consensus | Basis |
|---|---|---|---|---|---|
| situation_selection | A (dot) | B | B | **A (dot)** | Both coders initially coded B blind; after receiving one-sentence construct descriptions clarifying Webb's operationalization (reactive avoidance), both independently revised to A |
| concentration | A (dot) | B | B | **A (dot)** | Both coders initially coded B blind; after receiving one-sentence construct descriptions clarifying Webb's operationalization (rumination), both independently revised to A |
| situation_modification | B (linear) | C | B | **B (linear)** | Author + Coder 2 agreed on B; majority rule |
| acceptance | C (network) | B | C | **C (network)** | Author + Coder 2 agreed on C; majority rule |

### Effect on Webb distribution
- Before consensus: 2 dot / 4 linear / 4 network
- After consensus: **2 dot / 4 linear / 4 network** (original author codes confirmed)

### Rationale for construct-informed re-coding of situation_selection and concentration
Both independent coders, blinded to the DLN framework and author codes, initially coded these two items as B (linear). Review of their rationales revealed the disagreement stemmed from the blinded CSV providing only strategy names without construct definitions:
- **concentration**: Coders described "deliberate attentional regulation" and "attentional control" — i.e., everyday concentration. In Webb et al. (2012), this item refers to rumination: stimulus-driven fixation on emotional content, not controlled attentional deployment.
- **situation_selection**: Coders described "deliberate choice" and "structured planning." In Webb's framework, this includes reactive avoidance responses (e.g., leaving a room upon encountering a feared stimulus) — stimulus-driven withdrawal, not planned decision-making.

The author provided one-sentence construct descriptions clarifying Webb's operationalizations for the two disputed items, without revealing the DLN framework or author codes. With this additional context, both coders independently revised their codes to A (dot), agreeing that these strategies involve reactive, stimulus-driven processing rather than rule-governed deliberation.

## Greenwald (2009) — IAT Criterion Domain Topics

**Date: 2026-02-26**

**No consensus needed.** Both coders independently agreed with author coding on all 9 topic-to-DLN mappings (k = 1.000, 9/9 items). This includes the 3 mixed/unclear classifications (Personality, Drugs/tobacco, Clinical), which both coders independently flagged as ambiguous/multi-stage.

| Topic | Author | Coder 1 | Coder 2 | Agreement |
|---|---|---|---|---|
| Consumer | A (dot) | A | A | 3/3 |
| Race (Bl/Wh) | B (linear) | B | B | 3/3 |
| Politics | B (linear) | B | B | 3/3 |
| Gender/sex | B (linear) | B | B | 3/3 |
| Other intergroup | B (linear) | B | B | 3/3 |
| Relationships | C (network) | C | C | 3/3 |
| Personality | M (mixed) | M | M | 3/3 |
| Drugs/tobacco | M (mixed) | M | M | 3/3 |
| Clinical | M (mixed) | M | M | 3/3 |

### Note on mixed/unclear items
Both coders independently identified the same 3 domains as ambiguous without prompting, citing that these topics span multiple processing stages (e.g., phobic avoidance is reactive/dot-like while health behavior change is deliberative/linear-like). This convergence on mixed_unclear classification strengthens the sensitivity analysis approach of running models with and without these items.

## Final codes
All analyses use the consensus codes recorded in `webb2012_strategy_extraction.csv` (Webb), `hoyt2024_domain_extraction.csv` (Hoyt), and `greenwald2009_study_extraction.csv` (Greenwald topic-level coding).
