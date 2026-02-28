# Evidence Synthesis Protocol (Umbrella + Moderator Meta-Analysis)

## Scope and purpose
This paper uses a **targeted umbrella synthesis** as an empirical scaffold. The synthesis is intentionally selective (theory-driven) and focuses on **quantitative meta-analyses and systematic reviews** in the emotion–cognition landscape where (a) effect sizes are available and (b) **between-study heterogeneity** is reported or clearly implied.

The core scientific claim to be evaluated is:

> **DLN stage is a hidden moderator** that predicts when affect functions primarily as noise/bias (dot / linear regimes) versus signal/information (network regime), and therefore explains persistent heterogeneity and apparent sign reversals across emotion–cognition literatures.

This protocol operationalizes that claim in a way that can be executed as a formal moderator meta-analysis.

## Review questions
1. Across canonical emotion–cognition paradigms, how large are average effects and how large is unexplained heterogeneity after conventional moderators?
2. Does coding each effect size for **DLN-dominant architecture** (dot / linear / network) reduce heterogeneity and recover predicted sign/size differences?

## Eligibility criteria

### Included
- Quantitative meta-analyses and systematic reviews that report standardized effect sizes (e.g., *d*, *g*, *r*, OR) **or** provide sufficient information to derive them.
- Topics that map directly onto the fusion model's mechanisms:
  - emotion regulation strategies (suppression, reappraisal, attentional deployment, etc.)
  - implicit vs explicit measures / dissociations
  - interoception and emotional awareness (including alexithymia)
  - affective decision-making tasks (e.g., IGT variants; stress/affect load)

### Excluded
- Narrative reviews with no quantitative synthesis
- Meta-analyses that do not provide enough information to compute an effect size and sampling variance
- Studies where the task/method is not describable at a level sufficient to code DLN architecture

## Search strategy (umbrella layer)
Minimum search sources:
- PsycINFO / PubMed / Google Scholar
- Backward/forward citation chasing from seed meta-analyses already referenced in the manuscript

Example search strings:
- ("emotion regulation" AND meta-analysis AND suppression AND reappraisal)
- ("implicit association test" AND meta-analysis AND predictive validity)
- (interoception AND alexithymia AND meta-analysis)
- ("Iowa Gambling Task" AND meta-analysis)

## Data extraction
Extraction units:
- Primary: each **effect size** extracted from each included quantitative synthesis.
- Secondary: synthesis-level descriptors.

Required fields (see templates in `../extraction/`):
- study_id, effect_id
- paradigm_family
- outcome_type
- effect_size_metric
- yi (effect size in a common metric, e.g., Fisher-z for correlations)
- vi (sampling variance of yi)
- population descriptors (clinical, age band, expertise domain)
- manipulation descriptors (incidental vs integral emotion; stress/arousal; valence)
- moderators already used in the original synthesis
- heterogeneity statistics when available (Q, I², tau²)

## DLN stage coding (moderator layer)
DLN stage is coded **at the effect-size level**, using the rubric in `dln_stage_coding_manual.md`.

Interrater plan:
- Two independent coders
- Agreement reported as Cohen's κ (categorical) and/or percent agreement
- Disagreements adjudicated and logged

## Statistical analysis plan
Baseline:
- Random-effects model per paradigm family.

Moderator model:
- Random-effects meta-regression with DLN stage as a categorical moderator.
- Primary tests:
  1. Δtau² (heterogeneity reduction) when adding stage.
  2. Stage-specific pooled effects and predicted sign differences.
  3. Residual heterogeneity after stage.

Robustness:
- Influence diagnostics (leave-one-out, Cook’s distance analogs)
- Sensitivity to coding assumptions (conservative vs liberal stage assignments)
- Publication-bias sensitivity where feasible (Egger-type tests / selection sensitivity)

## Outputs (for the paper + repo)
- A versioned extraction sheet (CSV)
- A versioned coding log (CSV/JSON)
- Scripts that reproduce:
  - pooled effects by paradigm
  - pooled effects by stage within paradigm
  - heterogeneity reduction summaries
  - stage-by-paradigm figure(s)

