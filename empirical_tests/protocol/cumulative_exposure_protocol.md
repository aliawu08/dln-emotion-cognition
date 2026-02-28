# Deposited Empirical Protocol: Cumulative Exposure Sensitivity
# as a DLN-Stage-Specific Prediction

## Status
Pre-registered protocol. No data collected.

## Background and Rationale

The DLN computational framework (Wu, 2026) identifies cumulative
exposure tracking via the cross-term 2E_t * b_i as the mechanism that
separates Network-DLN agents from all other agent classes. Under this
mechanism, the marginal impact of an action depends not only on its
immediate payoff but on the agent's cumulative exposure state. Agents
that lack exposure tracking (Linear, Linear-Plus, Network-standard)
cannot represent this cross-term and collapse under stakes; critically,
Linear-Plus agents (factor compression without exposure tracking)
collapse *worst* because efficient factor-level exploitation accelerates
exposure accumulation without hedging.

The meta-analytic evidence presented in the companion manuscript is
consistent with the dangerous-middle prediction (Hoyt et al., 2024:
V-shaped pattern; interoception: sign reversal) but cannot directly
test the within-person cumulative exposure mechanism because
meta-analytic moderator analyses operate at the between-study level.

This protocol designs a prospective experimental test of the cumulative
exposure mechanism at the individual level.

## Design

### Overview
Between-subjects (DLN stage) x within-subjects (cumulative exposure
phase) mixed design. Participants complete a multi-round decision task
in which emotional consequences accumulate over trials.

### Independent Variables

**DLN stage (between-subjects):** Assessed via structural
representation task administered before the decision task. Classification
into dot, linear, and network stages based on whether participants
maintain (a) no persistent structure, (b) independent per-option
estimates, or (c) factor-level representations with structural revision.

The structural representation task must be independent of the emotional
decision task to avoid circularity. Candidate tasks:
- Factor structure learning task (Wu, 2026, Section 3.2)
- Multi-cue probability learning with structural transfer
- Hierarchical concept formation with revision

**Cumulative emotional exposure (within-subjects):** Operationalised
as increasing cumulative stakes across trial blocks. Early blocks have
low cumulative consequences (each decision is approximately independent).
Later blocks introduce compounding consequences (outcomes depend on
cumulative history).

### Dependent Variables

1. **Decision quality:** Proportion of optimal choices in each block
2. **Strategy adaptation:** Change in choice pattern as cumulative
   exposure increases (assessed via logistic regression coefficients)
3. **Response time:** Indexing processing demands (O(K) vs O(F) scaling)

### Predictions

1. **Network-DLN participants:** Adjust strategy based on cumulative
   exposure state. Decision quality remains stable or improves across
   blocks. Response times scale with factor structure (O(F)), not number
   of options.

2. **Linear participants:** Fixed choice pattern regardless of
   cumulative exposure. Decision quality degrades proportionally to
   exposure. Response times scale with K.

3. **Partial integration participants (the critical test):** Participants
   classified as having relational emotional knowledge (above-median
   factor structure scores) but low metacognitive monitoring (below-median
   revision scores) should show the *steepest* decline in decision
   quality under high cumulative load. This is the dangerous-middle
   prediction: efficient factor-level exploitation without exposure
   tracking accelerates accumulation without hedging.

4. **Dot participants:** Reactive, no systematic pattern. Decision
   quality is low throughout but does not decline systematically.

### Primary Analysis

Mixed-effects model:
  Decision_quality ~ DLN_stage * Block + (1 | participant)

Key test: DLN_stage x Block interaction, with specific contrast testing
that partial-integration participants show a steeper negative slope
than linear participants.

### Sample Size Justification

Based on the Hoyt et al. (2024) dangerous-middle effect size
(b = -0.10, SE = 0.023), the standardised effect for the stage x load
interaction is approximately d = 0.43. For 80% power at alpha = 0.05,
the required sample per group is approximately n = 45, yielding a
total target N = 180 (four groups: dot, linear, partial integration,
network).

### Randomisation
Participants are classified into DLN stage groups based on the
structural representation task. The decision task is identical for all
participants; cumulative exposure builds within the task in a fixed
sequence (not randomised). Block order is fixed because cumulative
exposure is inherently sequential.

### Exclusion Criteria
- Failure to complete the structural representation task
- Fewer than 80% of decision task trials completed
- Uniform responding (same choice on >95% of trials)
- DLN stage classification confidence below pre-specified threshold

### Analysis Plan

1. Primary: Mixed-effects model with DLN stage x Block interaction
2. Secondary: Pairwise contrasts (partial integration vs linear;
   network vs linear; network vs partial integration)
3. Sensitivity: Continuous DLN stage scores (factor structure and
   revision scores as separate predictors) rather than categorical
   classification
4. Robustness: Bayesian mixed model with weakly informative priors
   (BF > 3 as evidence threshold)

## Deposited Materials

- This protocol document
- Structural representation task specification (below)
- Decision task specification (below)
- Analysis script template (see `empirical_tests/analysis/analyze_experiment1.py`
  for the general template; the cumulative exposure extension is specified below)


---

## Appendix A: Structural Representation Task Specification

### Purpose
Classify participants into DLN stage groups *independently* of the
emotional decision task. The task assesses whether participants maintain
(a) no persistent structure (dot), (b) independent per-option estimates
(linear), or (c) factor-level representations with structural revision
(network).

### Task: Multi-Cue Factor Structure Learning

**Stimuli.** K = 8 options (presented as abstract choice objects, e.g.,
coloured tokens or labelled cards), with outcomes governed by F = 2
latent factors. Each option loads on one or both factors:

| Option | Factor 1 loading | Factor 2 loading | Expected outcome |
|--------|-----------------|-----------------|------------------|
| A      | 1.0             | 0.0             | m_1              |
| B      | 1.0             | 0.0             | m_1              |
| C      | 0.0             | 1.0             | m_2              |
| D      | 0.0             | 1.0             | m_2              |
| E      | 1.0             | 1.0             | m_1 + m_2        |
| F      | 1.0             | 1.0             | m_1 + m_2        |
| G      | 0.5             | 0.5             | 0.5(m_1 + m_2)   |
| H      | 0.0             | 0.0             | 0 (noise only)   |

Factor means m_1 and m_2 are drawn from N(50, 10) and shift every
20 trials (phase change), with Gaussian noise (SD = 5) on each
observation.

**Procedure.** 80 trials total (4 phases of 20 trials). On each trial,
participant selects one option and observes the outcome (numerical
value). After each phase, participants provide:
1. **Outcome predictions** for all 8 options (point estimates)
2. **Grouping task:** "Which options behave similarly?" (free grouping)
3. **Transfer test:** A novel option (I, loading = [0.7, 0.3]) is
   introduced; participants predict its outcome.

**Classification criteria:**

| Criterion | Dot | Linear | Network |
|-----------|-----|--------|---------|
| Prediction accuracy after phase change | Chance | Correct for previously sampled options only | Correct for all options sharing factor structure |
| Grouping task | No consistent grouping or random | Groups by surface features or recent outcomes | Groups by latent factor structure |
| Transfer test | Chance prediction | No transfer (prediction = overall mean) | Weighted prediction reflecting factor loadings |
| Structural revision score | N/A | 0 (no revision across phases) | > 0 (updates grouping after phase change) |

**Scoring.** Continuous scores on two dimensions:
- **Factor structure score** (0-1): normalised accuracy of grouping
  task relative to true factor structure (Rand index against ground
  truth partition).
- **Structural revision score** (0-1): proportion of phase transitions
  where grouping task changes in the correct direction after factor
  mean shift.

**Stage classification:**
- Dot: factor structure score < 0.3 AND revision score < 0.3
- Linear: factor structure score < 0.3 AND revision score < 0.3
  (distinguished from dot by prediction accuracy for sampled options)
- Network: factor structure score >= 0.5 AND revision score >= 0.3
- Partial integration: factor structure score >= 0.5 AND revision
  score < 0.3 (the critical dangerous-middle group)

Note: Continuous scores are used for the sensitivity analysis (Analysis
Plan item 3); categorical classification is used for the primary analysis.


---

## Appendix B: Decision Task Specification

### Purpose
Test whether cumulative emotional exposure produces stage-dependent
performance trajectories, with the dangerous-middle group showing the
steepest decline.

### Task: Emotional Consequence Accumulation Task (ECAT)

**Cover story.** Participants manage a simulated portfolio of
well-being investments across life domains (health, social,
financial, emotional). Each round, they allocate resources across
K = 6 domains. Outcomes in each domain depend on the allocation and
on a cumulative emotional exposure state that builds across rounds.

**Formal structure.** Adapted from the DLN foundation paper (Wu, 2026,
Section 4.2):

- K = 6 choice options (domains)
- F = 2 latent outcome factors (underlying shared structure)
- Each domain loads on factors with known loadings b_i
- Immediate payoff per round: r_t = sum_i(a_i * mu_i) + noise
- Cumulative exposure: E_t = E_{t-1} + sum_i(a_i * b_i)
- Penalty (applied from Block 3 onward):
  P_t = lambda * (E_t^2)
  where lambda controls penalty severity
- Net outcome: V_t = r_t - P_t
- The marginal impact of action a_i on penalty is:
  dP/da_i = lambda * 2 * E_t * b_i (the cross-term)

**Block structure (5 blocks, 20 rounds each):**

| Block | Rounds | lambda | Exposure effect |
|-------|--------|--------|-----------------|
| 1     | 1-20   | 0.00   | None (pure learning phase) |
| 2     | 21-40  | 0.01   | Mild (E_t small) |
| 3     | 41-60  | 0.03   | Moderate (cross-term becomes relevant) |
| 4     | 61-80  | 0.05   | Strong (requires exposure tracking) |
| 5     | 81-100 | 0.05   | Sustained (tests adaptation vs collapse) |

**Emotional valence manipulation.** Outcomes in each round are
accompanied by brief emotional feedback:
- Positive outcome (V_t > median): brief positive image (IAPS)
  or rewarding sound
- Negative outcome (V_t < median): brief negative image or
  aversive sound
- Cumulative exposure is framed as "emotional load" with a visible
  accumulation meter

**Optimal strategy.** The optimal allocation shifts across blocks:
- Blocks 1-2: allocate to highest-return domains (exploitation)
- Blocks 3-5: allocate to balance return against exposure
  accumulation (requires tracking 2E_t * b_i)

Network-DLN participants who track cumulative exposure should shift
strategy; linear participants who do not track exposure should maintain
Block 1-2 strategy and accumulate penalty; partial integration
participants should accumulate penalty fastest because factor-level
exploitation is efficient at increasing returns *and* exposure
simultaneously.

**Control conditions:**
- **No-accumulation control:** Same task with lambda = 0 throughout
  (verifies stage differences are specific to cumulative exposure,
  not general ability)
- **Explicit feedback condition:** E_t displayed on screen (tests
  whether providing exposure information eliminates stage differences)


---

## Appendix C: Analysis Script Specification

### Primary model

```
library(lme4)
library(lmerTest)

# decision_quality: proportion optimal in each block (0-1)
# stage: factor with levels dot, linear, partial, network
# block: numeric 1-5
# pid: participant ID

m1 <- lmer(decision_quality ~ stage * block + (1 + block | pid),
           data = dat)

# Primary test: stage x block interaction
anova(m1, type = 3)

# Specific contrasts (Helmert-coded):
# C1: partial vs linear (dangerous-middle test)
# C2: network vs partial (recovery test)
# C3: network vs linear (basic DLN ordering)
library(emmeans)
emm <- emtrends(m1, ~ stage, var = "block")
pairs(emm, adjust = "holm")
```

### Secondary analyses

```
# Continuous predictors (sensitivity analysis)
m2 <- lmer(decision_quality ~ factor_score * block +
           revision_score * block +
           factor_score:revision_score:block +
           (1 + block | pid), data = dat)

# The three-way interaction tests whether high factor structure
# combined with low revision (partial integration) produces
# steeper decline than low factor structure (linear)
```

### Bayesian robustness check

```
library(brms)
m3 <- brm(decision_quality ~ stage * block + (1 + block | pid),
          data = dat,
          prior = c(
            prior(normal(0, 0.5), class = "b"),
            prior(student_t(3, 0, 0.5), class = "sd"),
            prior(lkj(2), class = "cor")
          ),
          iter = 4000, warmup = 1000, chains = 4)

# Report BF for stage x block interaction
hypothesis(m3, "stagelinear:block < stagepartial:block")
```

### Stopping rule

Data collection proceeds until N = 180 (45 per group) is reached or
until a Bayesian sequential analysis (conducted after each batch of
20 participants) yields BF > 6 or BF < 1/6 for the primary contrast
(partial vs linear slope difference).

### Data exclusion pipeline

```python
# Implemented in Python for consistency with the meta-analysis pipeline
# 1. Remove incomplete participants (< 80% trials)
# 2. Remove uniform responders (> 95% same choice)
# 3. Remove classification-uncertain participants
#    (factor_score and revision_score both in [0.3, 0.5] range)
# 4. Report: N excluded, N per group, classification distribution
```

## References

Wu, A. (2026). Compression efficiency and structural learning as a
computational model of DLN cognitive stages. bioRxiv.
doi:10.64898/2026.02.01.703168

Hoyt, M. A. et al. (2024). The utility of coping through emotional
approach: A meta-analysis. Health Psychology, 43(6), 397-417.
