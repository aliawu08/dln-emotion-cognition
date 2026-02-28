# DLN-Stage Coding Rubric: Desmedt et al. (2022) — Heartbeat Counting Task Validity

## Source

Desmedt, O., Van Den Houte, M., Walentynowicz, M., Dekeyser, S., Luminet, O.,
& Corneille, O. (2022). How does heartbeat counting task performance relate to
theoretically-relevant mental health outcomes? A meta-analysis. *Collabra:
Psychology, 8*(1), 33271. DOI: 10.1525/collabra.33271

## Analysis rationale

Desmedt et al. report that HCT performance is *not* significantly associated
with trait anxiety, depression, or alexithymia, but is weakly associated with
heart rate, BMI, age, and sex.  The shared variance ranges from 0.01%
(alexithymia) to 2.9% (heart rate).

The DLN framework predicts this pattern through **cross-level mismatch**: the
HCT is a dot-stage task (passive signal detection), so it should correlate most
strongly with dot-stage criteria (biological/physiological measures) and most
weakly with linear-stage criteria (psychological self-report measures).  This is
distinct from the Zanini boundary condition (all-dot, no between-stage
variation): here, variation exists in the *criterion measures*, not the task.

## Unit of analysis

Criterion measure (k = 7 outcome categories from the meta-analysis, each with
≥10 studies).  Each criterion pools many individual studies; the criterion-level
pooled effect is the unit for DLN coding.

## Task-level coding

**HCT = dot stage.**  The heartbeat counting task requires passive detection of
cardiac signals over short time intervals.  It involves:
- No cognitive mediation beyond counting
- No emotional interpretation or integration
- No relational structure between elements
- Stimulus-driven accuracy with $O(1)$ processing

## DLN-stage coding of criterion measures

### Dot stage: Biological/physiological criteria

**Criterion:** The outcome is a direct biological or physiological measure
requiring no cognitive mediation.  These are same-level (dot-dot) pairings
with HCT — both task and criterion operate through basic somatic channels.

- **Heart rate** (k=40) — Resting heart rate is a pure physiological measure.
  Higher HCT accuracy mechanistically linked to stronger cardiac signal.
- **BMI** (k=29) — Body mass index is a biological marker.  Higher BMI may
  attenuate cardiac signal through adipose tissue.
- **Age** (k=20) — Chronological age is biological.  Vascular changes with age
  affect cardiac signal strength.
- **Sex** (k=14) — Biological sex differences in cardiac physiology and
  interoceptive sensitivity.

### Linear stage: Psychological self-report criteria

**Criterion:** The outcome is a self-report psychological measure that tracks
emotional/cognitive states along single dimensions.  These are cross-level
(dot-linear) pairings — the HCT measures basic signal detection while the
criterion measures higher-order psychological constructs through
compartmentalised self-assessment.

- **Trait anxiety** (k=41) — STAI rates anxiety on a single valence/arousal
  dimension.  Self-report; no integration of bodily signals required.
- **Depression** (k=31) — BDI rates depressive symptoms along a severity
  dimension.  Cognitive-affective self-assessment.
- **Alexithymia** (k=23) — TAS-20 rates difficulty identifying/describing
  feelings.  Self-report about emotional processing deficits.

### Network stage: None

No criterion measures in the Desmedt dataset are network-stage.  This is
itself informative: the HCT literature has not examined network-stage outcomes
(e.g., emotional granularity, integrative awareness, flexible coping) because
the task was designed to measure basic signal detection.

## Key DLN prediction

| Criterion stage | Predicted HCT association | Mechanism |
|-----------------|--------------------------|-----------|
| Dot (biological) | Weak but detectable | Same-level pairing: both measure somatic channels |
| Linear (psychological) | Near-zero | Cross-level mismatch: dot task cannot predict linear-stage constructs |
| Network (none) | — | Not tested in this literature |

The critical test is whether **dot-stage criteria show systematically stronger
absolute associations** with HCT than linear-stage criteria.

## Sensitivity analyses

1. Reclassify **age** from dot → exclude (not a health outcome per se).
2. Use absolute r values (since sign direction varies across criteria).
3. Compare mean |r| for dot-stage vs linear-stage criteria.

## Data status

All 7 criterion-level r values are verified from the results section (page 4)
of Desmedt et al. (2022):

| Criterion | r [95% CI] | k | se |
|-----------|-----------|---|-----|
| Heart rate | −0.17 [−0.22, −0.10] | 40 | 0.031 |
| BMI | −0.11 [−0.18, −0.04] | 29 | 0.036 |
| Age | −0.06 [−0.14, 0.02] | 20 | 0.041 |
| Sex | −0.14 [−0.20, −0.07] | 14 | 0.033 |
| Trait anxiety | 0.03 [−0.04, 0.11] | 40 | 0.038 |
| Depression | −0.04 [−0.16, 0.08] | 31 | 0.061 |
| Alexithymia | −0.01 [−0.17, 0.15] | 23 | 0.082 |

N_approx values are effective sample sizes derived from the reported 95%
confidence intervals to reproduce the correct Fisher's z sampling variance.
DLN coding is pre-specified and independent of effect sizes.
