# Experiment 1 Protocol (Preregisterable)

## Goal
Test the core Paper 2 claim with **new data**:

> DLN stage moderates whether affective load functions as interference (linear suppression) or as informative signal (network integration).

## Design overview
- Between-subjects factor: **DLN stage** (measured, not assigned): dot / linear / network
- Within- or between-subjects factor: **Affective load**: low vs high (random assignment)
- Primary outcome: **Decision quality** in a complex, uncertain choice task (e.g., IGT-like or multi-attribute choice under uncertainty)
- Secondary outcomes:
  - interoceptive accuracy / sensibility
  - emotional granularity (performance-based if possible)
  - implicit–explicit alignment (attitudes, preferences)
  - regulation strategy use (self-report + task-based)

## Hypotheses (interaction patterns)
H1 (primary): Stage × Load interaction:
- Dot: consistently poor decision quality, weak adaptation.
- Linear: high quality under low load; sharp drop under high load (suppression brittleness).
- Network: highest quality; smaller drop and/or adaptive recalibration under high load.

H2: The same stage signatures appear across measures:
- Linear: larger implicit–explicit discrepancies, higher TAS-20, high suppression endorsement.
- Network: higher granularity and adaptive reappraisal signatures.

## Operationalization of DLN stage
Minimum viable approach:
- A structured representation task (causal mapping / relational integration / feedback reasoning)
- Scoring rubric consistent with Paper 1 computational definitions:
  - dot: minimal persistent structure
  - linear: chain-like, single-cause narratives
  - network: multi-causal, feedback-aware representations

## Sample size and power (starter)
A conservative target for a measured-moderator interaction:
- N ≈ 240–360 (e.g., 80–120 per stage band after classification)
- If stage distribution is uneven, oversample and predefine classification thresholds.

## Statistical analysis plan
Primary model:
- DecisionQuality ~ Load * Stage + covariates
- Planned contrasts:
  - (Linear high load vs linear low load) drop
  - (Network high load vs network low load) drop
  - Compare drop magnitudes

Robustness:
- Alternative stage thresholds
- Include continuous stage score (graph topology score) as sensitivity analysis

## Data and exclusions
- Predefine performance/attention checks
- Predefine exclusion rules (e.g., random responding thresholds)
