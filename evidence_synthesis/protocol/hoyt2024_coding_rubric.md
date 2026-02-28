# DLN-Stage Coding Rubric: Hoyt et al. (2024) — Emotional Approach Coping & Health

## Source

Hoyt, M. A., Llave, K., Wang, A. W.-T., Darabos, K., Diaz, K. G., Hoch, M.,
MacDonald, J. J., & Stanton, A. L. (2024). The utility of coping through
emotional approach: A meta-analysis. *Health Psychology, 43*(6), 397–417.
DOI: 10.1037/hea0001364

## Analysis rationale

Hoyt et al. report an overall small positive association between emotional
approach coping (EAC) and health (r = 0.05, 95% CI [0.003, 0.10]) across 86
studies.  Critically, meta-regressions reveal **domain-specific sign changes**:
EAC is linked to *better* health in biological/physiological, physical, and
resilience domains, but to *worse* outcomes in mental/emotional distress and
risk-related adjustment domains.

The DLN framework predicts this pattern: the benefit of emotional approach
coping depends on whether the health outcome domain engages dot-stage
(reactive/somatic), linear-stage (compartmentalized distress management), or
network-stage (integrative adaptive processing) cognitive architectures.

## Unit of analysis

Health outcome domain (k = 8 domains from the meta-regression moderator
analysis).  Each domain pools many individual effect sizes via the three-level
model; the domain-level pooled effect is the unit for DLN coding.

## DLN-stage coding decision tree

### Dot stage: Somatic/reactive outcomes

**Criterion:** Outcome is a direct bodily or behavioural response with minimal
cognitive mediation.  Emotional approach coping acts through basic
attention-to-body or stimulus-driven health behaviour, not through reflective
integration.

- **Biological/physiological** — Biomarkers (cortisol, inflammatory markers,
  immune function, cardiovascular reactivity).  These are downstream somatic
  responses, not cognitively mediated health constructs.
- **Physical health** — Self-reported symptoms, functional status, pain
  intensity.  These track somatic state rather than cognitive interpretation of
  health status.

### Linear stage: Unidimensional distress outcomes

**Criterion:** Outcome measures assess single-valence symptom severity along
isolated dimensions.  Measurement instruments (e.g., BDI, STAI, IES subscales)
treat emotional states as unidimensional severity ratings without relational,
contextual, or cross-domain integration components.  Items do not require
respondents to integrate across life domains or evaluate multiple interacting
dimensions simultaneously.

- **Mental/emotional distress** — Depression, anxiety, negative affect, general
  distress scales.  These index single-dimension severity without relational
  structure.
- **Risk-related psychological adjustment** — Psychological risk factors
  (intrusive thoughts, avoidance, hyperarousal, maladaptive schemas).  These
  assess isolated symptom dimensions without cross-domain integration; items
  are structurally independent.

### Network stage: Integrative/adaptive outcomes

**Criterion:** Outcome requires flexible integration of emotional information
with broader cognitive and social context.  Successful performance on these
outcomes involves using emotional signals *as data* rather than treating them as
noise — precisely the network-stage architecture.

- **Resilience-related psychological adjustment** — Post-traumatic growth,
  benefit finding, sense of coherence, adaptive coping self-efficacy.  These
  require active meaning-making that integrates emotional experience with
  cognitive reappraisal.
- **Positive psychological health** — Subjective well-being, life satisfaction,
  positive affect, flourishing.  These reflect emotion-cognition integration
  rather than distress minimisation.
- **Social functioning** — Interpersonal relationship quality, social support
  utilisation, social adjustment.  These require multi-dimensional integration
  of emotional signals within relational contexts.

### Ambiguous case

- **Behavioral** — Health behaviours (exercise, medication adherence, sleep
  hygiene).  These span dot (simple stimulus–response habits) to linear
  (rule-following adherence) to network (flexible self-regulation).  Coded as
  **dot** (most parsimonious: majority of measured health behaviours are simple
  action-level outcomes) but flagged for sensitivity analysis.

## Key DLN prediction

| Stage   | Predicted EAC–health direction | Mechanism |
|---------|-------------------------------|-----------|
| Dot     | Small positive (+)            | Emotional processing brings basic attention to somatic signals |
| Linear  | Near-zero or negative (−)     | Emotional processing amplifies distress without integrative resolution; dangerous-middle pattern |
| Network | Moderate positive (+)         | Emotional signals integrated as adaptive information; meaning-making and flexible deployment |

The critical test is the **sign reversal between linear and network domains**:
the same coping strategy (emotional approach) produces opposite effects
depending on whether the outcome domain engages compartmentalised or
integrative processing.

## Sensitivity analyses

1. Reclassify **behavioral** from dot → linear; check impact on tau² reduction.
2. Reclassify **physical health** from dot → linear (if construed as cognitive
   appraisal of health status rather than somatic tracking); check impact.
3. Run with only the 5 domains where effect-size direction is unambiguous
   (biological, mental distress, risk-adjustment, resilience, positive psych).

## Data status

All 8 domain-level EAC pooled correlations (r) are verified from Table 3 of
Hoyt et al. (2024).  Domain-level k (number of studies) are verified from
Figure 2.  N_approx values in the extraction CSV are effective sample sizes
derived from the reported robust standard errors (se computed from CI width)
to reproduce the correct Fisher's z sampling variance; they do not represent
literal participant counts.  DLN coding is pre-specified and independent of
effect sizes.

### Verified domain-level values (EAC construct)

| Domain | r [95% CI] | k | se |
|--------|-----------|---|-----|
| Biological/Physiological | −0.02 [−0.25, 0.21] | 8 | 0.117 |
| Physical | 0.02 [−0.08, 0.12] | 21 | 0.051 |
| Behavioral | 0.14 [−0.21, 0.45] | 8 | 0.168 |
| Mental/Emotional | −0.11 [−0.19, −0.03] | 66 | 0.041 |
| Risk | −0.18 [−0.28, −0.07] | 15 | 0.054 |
| Positive | 0.29 [0.19, 0.38] | 25 | 0.048 |
| Social | 0.28 [0.18, 0.38] | 15 | 0.051 |
| Resilience | 0.31 [0.23, 0.39] | 22 | 0.041 |
