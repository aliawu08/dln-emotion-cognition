# Experiment 1 Protocol (Preregisterable)

## Goal
Test the core fusion model claim with **new data**:

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

H3 (dangerous-middle / Prediction 9): Partial integration without metacognitive monitoring yields *worse* outcomes than full compartmentalization under high affective load.
- Operationalization: Identify participants who score moderately on relational emotional knowledge (e.g., mid-range LEAS) but low on metacognitive monitoring indices (e.g., inability to revise emotional interpretations in response to disconfirming feedback; low structural-revision scores on DLN assessment tasks).
- These "Linear-Plus" individuals should show worse decision quality under high load than linear-stage individuals who fully compartmentalize.
- This is the strongest discriminating test of the DLN framework versus monotonic stage models.

## Operationalization of DLN stage
Minimum viable approach:
- A structured representation task (causal mapping / relational integration / feedback reasoning)
- Scoring rubric consistent with the DLN computational framework (Wu, 2026):
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

## Disconfirmation criteria
The following outcomes would constitute evidence against the fusion model's core claims:

1. **No Stage × Load interaction.** If network-stage individuals exhibit the same decision-quality decline under emotional load as linear-stage individuals (H1 unsupported).
2. **Monotonic integration benefit.** If partial emotional integration (Linear-Plus configuration) consistently *improves* outcomes relative to full compartmentalization, the dangerous-middle prediction (H3) is falsified.
3. **Stage classification circularity.** DLN stage must be assessed via structural representation tasks independent of the emotion–cognition outcomes being predicted. If stage classification can only be achieved through the emotion measures themselves, the framework is circular.
4. **No cross-measure coherence.** If the same individuals show contradictory stage signatures across regulation, interoception, implicit–explicit coupling, and decision quality measures (H2 unsupported).

## Data and exclusions
- Predefine performance/attention checks
- Predefine exclusion rules (e.g., random responding thresholds)
