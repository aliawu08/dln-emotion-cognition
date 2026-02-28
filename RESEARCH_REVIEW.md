# Head of Research Review: DLN Emotion-Cognition Repository

**Paper:** "Cognitive architecture moderates emotion-cognition relationships: cross-domain evidence from meta-analytic reanalysis"
**Author:** Alia Wu (wut08@nyu.edu)
**Target venue:** Nature Human Behaviour (Article format)
**Review date:** 2026-02-28
**Reviewer scope:** Full codebase audit (261 tests, ~140 source files, manuscript, supplementary, all extraction data, all protocols, metafor cross-validation, blinded IRR materials)

---

## 1. EXECUTIVE SUMMARY

This repository accompanies a theory-testing paper proposing that Dot-Linear-Network (DLN) cognitive architecture is a **hidden structural moderator** that explains why emotion-cognition findings appear contradictory across research literatures. The project reanalyses six published meta-analyses spanning four domains (emotion regulation, implicit cognition, interoception, health psychology), supplemented by agent-based simulations and a deposited experimental protocol.

**Overall verdict:** This is a genuinely strong research project --- well-organized, transparently documented, and methodologically rigorous at a level that substantially exceeds the field norm. The central theoretical contribution (cognitive architecture as hidden moderator) is novel and the cross-domain convergence is compelling. Two prior reviews identified and resolved all critical blockers. This review identifies **no new blockers** but raises several substantive concerns about theoretical framing, statistical interpretation, and scope that deserve attention.

**Codebase status:** 261/261 tests passing (pytest, 13.8s). Clean modular structure. Python 3.8+ with pinned scientific stack (numpy==2.4.2, scipy==1.17.1, pandas==3.0.1, matplotlib==3.10.8, statsmodels==0.14.6).

---

## 2. WHAT THE PAPER CLAIMS

The DLN framework models cognitive development as shifts in belief-dependency graph topology:

| Stage | Architecture | Emotion-Cognition Pattern |
|-------|-------------|--------------------------|
| **Dot** | Empty belief graph; O(1) memory | Reactive separation --- stimulus-driven, no integration |
| **Linear** | Null graph on K nodes; O(K) scaling | Suppressive compartmentalisation --- emotion as interference |
| **Network** | Bipartite factor DAG; O(F) scaling | Integrative fusion --- emotion as information |

The key testable claim: coding published meta-analytic data by DLN stage should reduce between-study heterogeneity that conventional moderators leave unexplained. A distinctive prediction --- the **dangerous middle** --- holds that partial integration without metacognitive monitoring produces *worse* outcomes than full compartmentalisation.

---

## 3. VERIFIED RESULTS

| Dataset | k | Domain | tau2 reduction | Perm. p | LOO R2 | Key pattern |
|---------|---|--------|---------------|---------|--------|-------------|
| Webb (strategy) | 10 | Emotion regulation | 69.9% | 0.025 | 0.19 | Network > Linear > Dot |
| Webb (comparison) | 306 | Emotion regulation | ~10% | <0.001 | 0.054 | Same ordering, cluster-robust |
| Interoception | 8 | Interoception | 84.7% | 0.002 | 0.74 | Sign reversal (lin+, net-) |
| Hoyt | 8 | Health psychology | 100% | 0.008 | 0.89 | V-shape (dangerous middle) |
| Desmedt | 7 | Heartbeat counting | 82.6% | 0.111 | 0.44 | Dot |r| > Linear |r| |
| Greenwald (4-level) | 184 | Implicit cognition | 32.9% | <0.001 | N/A | Suppression fingerprint |
| Zanini (boundary) | 110 | Affective decision | N/A | N/A | N/A | Null moderation (all dot) |

DLN achieved the highest tau2 reduction in every head-to-head comparison with established alternative moderators. No alternative moderator generalises across domains.

I verified these results by running the full test suite (261/261 pass), reading all extraction CSVs, auditing the REML implementation against standard formulations, and tracing manuscript claims back to analysis output tables.

---

## 4. STRENGTHS

### 4.1 Cross-domain convergence (the paper's strongest card)
The same three-question decision tree reduces heterogeneity across four unrelated domains. Competing moderators (Gross process model, cognitive effort, Garfinkel trichotomy, construct valence, temporal frame) are domain-specific; only DLN generalises. This is the single most persuasive element of the paper and the one most likely to shift reviewer opinion, because it cannot be easily dismissed as post-hoc fitting to a single dataset.

### 4.2 Dangerous-middle prediction
The V-shaped pattern in Hoyt (linear *worse* than dot and network) is a direction-specific prediction unique to DLN. No monotonic stage model (LEAS, constructionism, integrative complexity, ACT) generates it. This is theoretically distinctive and, if replicated at the individual level, would constitute strong evidence for the framework.

### 4.3 Boundary-condition test
Zanini (k=110, all dot) correctly predicts where the framework should *not* work. This scope-delimiting test is rare in theory papers and significantly adds credibility.

### 4.4 Falsification infrastructure
The paper explicitly defines disconfirmation criteria at multiple levels:
- Kappa < 0.60 disconfirms coding reliability
- No Stage x Load interaction disconfirms the framework
- Monotonic improvement disconfirms the dangerous-middle prediction
- Stage classification dependent on outcomes disconfirms independence

Supplementary Table 1 lists six specific falsifiable predictions with named measures and expected directions. This exceeds the norm.

### 4.5 Statistical safeguards
- **Exhaustive permutation tests** using Stirling number enumerations for small k; Monte Carlo for large k
- **LOO cross-validation** with 10,000 null-model simulations
- **Competing moderator analysis** with 2-3 alternatives per dataset
- **47/47 sensitivity coding scenarios** preserving qualitative patterns
- **Profile-likelihood CIs** for tau2 under small-k conditions

### 4.6 Transparency and reproducibility
- All extraction CSVs deposited with source annotations and coding rationales
- Blinded coding rubrics, response sheets, and consensus log included
- 261-test suite validates CI coverage, REML convergence, permutation logic, agent invariants, numerical stability
- Metafor cross-validation completed (198 parameters, 144 PASS, 54 known formula variants, 0 FAIL)
- Blinded IRR completed for Webb (10/10), Hoyt (8/8), and Greenwald topic-level (9/9) = 27/27 items

### 4.7 Manuscript quality
Clear structure, precise technical writing, appropriate for NHB format. The contradiction framing (Introduction) effectively motivates the paper. Limitations are handled with unusual candour for a single-author paper. The cover letter leads with concrete results.

---

## 5. CONCERNS AND RECOMMENDATIONS

### 5.1 The circularity question needs stronger prophylaxis (MAJOR)

The paper's central operation is: take existing data organised by existing categories (strategies, measures, outcome domains), reclassify those categories using DLN labels, and show that the new labels reduce heterogeneity better. The critical question any reviewer will ask is: **is this reclassification doing anything beyond reorganising already-known groupings in a way that happens to align with effect sizes?**

The paper addresses this concern in several places (structural coding rationale, permutation tests, sensitivity analyses, competing moderator comparison, blinded IRR). These are all appropriate. However, the argument could be strengthened:

**Recommendation:** Add a brief paragraph to the Discussion explicitly addressing the strongest version of the circularity objection. Something like: "The most challenging version of the circularity concern is not that effect sizes informed coding (they did not --- codes were assigned based on structural features of measures), but that anyone with domain knowledge could construct a grouping that aligns with known effect-size patterns, and DLN provides post-hoc labels for such a grouping. Three observations argue against this interpretation: (1) the competing moderator analysis shows that obvious domain-specific groupings (Gross process model, Garfinkel trichotomy) do *not* systematically align with effect sizes; (2) no single alternative moderator generalises across domains, whereas DLN does; and (3) the DLN coding produces direction-specific predictions (sign reversals, V-shapes) that a mere relabelling of known groupings would not generate."

### 5.2 The "70-100%" framing risks overclaiming (MODERATE)

The abstract opens with "DLN stage reduces between-study heterogeneity by 70-100% across four analyses." This is technically accurate (Webb strategy-family=70%, interoception=85%, Desmedt=83%, Hoyt=100%), but it front-loads the small-k results (k=7-10) without adequate context. The comparison-level Webb analysis (k=306) --- which has real statistical power --- achieves only 10% reduction. The Greenwald study-level analysis (k=184) achieves 33%.

The reader's first impression is of a framework that explains 70-100% of heterogeneity. The reality is that this holds only at the category level (k=7-10) where model saturation is a genuine concern, while at the study level (where power is adequate) the reduction is 10-33%.

**Recommendation:** The abstract currently handles this reasonably by separately noting the k=184 result, but consider reframing to lead with the convergent pattern across domains rather than the percentage range. The persuasive unit is the cross-domain convergence, not the magnitude of any single reduction.

### 5.3 The Hoyt "100% reduction" warrants stronger caveats (MODERATE)

Three parameters fitting eight data points will often achieve near-perfect fit regardless of the model's theoretical validity. The permutation test (p=0.008) mitigates this concern, as does the profile-likelihood CI. However, the Discussion could more explicitly note that 100% tau2 reduction at k=8 is expected under moderate model saturation and should not be interpreted as indicating the framework explains all between-study variance.

The LOO R2 = 0.89 for Hoyt is impressively high, but given k=8, this means each leave-one-out fold removes 12.5% of the data --- the model is fitting on 7 points, which for a 3-parameter model leaves df=4. The high LOO-R2 is consistent with the framework being meaningful, but the precision of the estimate is low.

**Recommendation:** Add a sentence in the Hoyt discussion: "The 100% tau2 reduction should be interpreted as reflecting near-complete separation of stage groups in this small dataset, not as a claim that DLN explains all true heterogeneity. The profile-likelihood 95% CI for residual tau2 ([0.000, 0.007]) is consistent with both zero and small residual heterogeneity."

### 5.4 The computational model is underspecified relative to its claims (MODERATE)

The agent-based simulation (`computational/`) is described as "intentionally minimal" and as a "reproducible computational scaffold." This framing is appropriate. However, a few issues deserve attention:

1. **Learning rates are not calibrated.** LinearAgent lr=0.15 and NetworkAgent lr=0.12 are chosen without justification. The lr difference (0.15 vs 0.12) means the linear agent learns faster per step, which could influence the comparison independently of the architectural differences.

2. **Cost scaling is arbitrary.** The cost function `net = (reward - 0.01 * cost) / T` uses a 0.01 multiplier that is not derived from theory. Given that LinearAgent cost = K and NetworkAgent cost = F + structural_updates, the 0.01 weighting effectively makes cost negligible. The simulation results would be qualitatively identical with cost=0.

3. **The structural learning trigger is a heuristic.** NetworkAgent triggers structural updates when |PE| > 0.35, paying a cost of 0.5*F. This is a simplification of the "hypothesis-test-update" cycle described in the companion paper, but the specific thresholds are not derived.

4. **No parameter sensitivity sweep exists.** The README mentions the simulation is for "generating qualitative regime diagrams," but no analysis confirms that the qualitative ordering (network > linear when K >> F) holds across parameter space.

**Recommendation:** Add a parameter sensitivity analysis (sweep learning rates 0.05-0.30, structure threshold 0.2-0.5, cost weight 0.0-0.1) and report whether the qualitative regime ordering is robust. This is a few hours of work and would substantially strengthen the computational layer. Alternatively, if the ordering always holds regardless of parameters, state this explicitly as a structural property of the architecture (which is the stronger claim).

### 5.5 Interoception variance formula needs clearer justification (MINOR)

The interoception analysis uses `vi = 1/(N_total - 3*k)` rather than the standard Fisher-z variance `1/(N-3)`. The Methods now document this as "adjusting the standard Fisher-z variance for the aggregation level," but the statistical derivation is not provided. For a reviewer familiar with meta-analytic methods, this will raise a flag.

The formula is conservative (it produces larger variances than `1/(N-3)` would), so it works *against* the paper's findings. But the lack of a formal justification is a gap.

**Recommendation:** Either (a) derive the formula briefly in the Methods (showing it follows from averaging k Fisher-z values from studies of approximately equal size), or (b) include a sensitivity analysis using standard `1/(N-3)` variance and showing the results are qualitatively unchanged.

### 5.6 Blinded IRR covers only 27 items total (MINOR)

The blinded inter-rater reliability now covers all three coded datasets: Webb (10/10), Hoyt (8/8), and Greenwald topic-level (9/9) = 27/27 items with 100% agreement. This is a strong result.

However, the Greenwald IRR covers the 9 *topic-to-DLN mappings* (domain level), not the 184 *study-level* codes. The study-level coding involves assigning each individual study to dot/linear/linear-plus/network based on the criterion behaviour. While most of this follows mechanically from the domain mapping, the linear vs. linear-plus distinction within the intergroup domain (97 vs. 43 studies) was done by the author alone.

The 31-scenario sensitivity analysis covers this distinction adequately (tau2 reduction ranges from 28.7% to 34.7%), but a reviewer may still note that the linear/linear-plus boundary is the weakest-validated coding decision in the paper.

**Recommendation:** Acknowledge this explicitly: "The blinded coding validated the domain-to-stage mapping; the within-domain linear/linear-plus boundary assignment was conducted by the author and validated via the 31-scenario sensitivity analysis (Section X)."

### 5.7 The Desmedt result (p=0.111) weakens the convergent narrative (MINOR)

The Desmedt permutation test is non-significant (p=0.111). The manuscript handles this well, noting the limited combinatorial resolution (63 partitions). However, including a non-significant result in a "convergent evidence" narrative requires careful framing.

The current text says DLN coding "outperformed 89% of random assignments" --- this is an accurate restatement of p=0.111, but it reads as spin. A clearer framing would be: "The Desmedt analysis showed a large tau2 reduction (83%) consistent with the predicted pattern (same-level correspondence), but the permutation test did not reach significance (p=0.111), reflecting the limited combinatorial resolution of 63 possible two-way partitions."

### 5.8 Measurement-objectivity confound in Desmedt (MINOR)

The competing moderator analysis reveals that the DLN coding for Desmedt is *exactly equivalent* to a measurement-objectivity coding (objective vs. self-report). The manuscript acknowledges this ("the measurement-objectivity coding is exactly equivalent to the DLN coding for this dataset"). This is intellectually honest but raises a question: is DLN doing anything more than distinguishing objective from self-report measures?

The answer the paper gives is correct --- DLN generalises across domains while measurement objectivity does not --- but this argument could be stated more forcefully in the Desmedt paragraph.

### 5.9 No individual-level evidence exists (ACKNOWLEDGED)

The paper codes *tasks and measures*, not *participants*. All moderator analyses operate at the between-study level. The DLN framework's claim that individuals at different cognitive stages process emotion differently has not been tested. The deposited protocol (cumulative exposure protocol) addresses this gap in principle, but no data have been collected.

This is not a flaw --- it is a scope limitation appropriate for a theory-testing paper using meta-analytic data. The manuscript acknowledges it clearly (Discussion, deposited protocol). No recommendation needed beyond what is already present.

---

## 6. STATISTICAL METHODOLOGY ASSESSMENT

### 6.1 Core meta-analysis pipeline (meta_pipeline.py) --- SOUND

- REML via bounded scalar optimisation: correctly implemented
- Knapp-Hartung small-sample correction with t-based CIs: correct; the QE floor at 1.0 is conservative (prevents deflation of SEs)
- Profile-likelihood CIs for tau2 via bisection on REML log-likelihood: mathematically sound, 100 bisection iterations gives precision ~10^-30
- Cochran's Q and I2: standard formulations using fixed-effects weights
- Egger's regression: properly specified (precision as predictor, t-test on intercept)
- AICc with small-sample correction: correct formula (Hurvich & Tsai 1989)
- QM test: computed as Q_baseline - Q_residual with chi2 df; standard meta-analytic definition

### 6.2 Three-level model (multilevel_meta.py) --- SOUND WITH CAVEAT

- V matrix correctly implements compound symmetry within clusters
- REML objective via Cholesky decomposition: correct
- CR2 cluster-robust sandwich estimator: well-implemented with eigenvalue regularisation
- **Caveat:** Satterthwaite df uses `n_clusters - p` uniformly for all coefficients. The correct Satterthwaite approximation should vary by coefficient. This is conservative (underestimates df, widens CIs) and therefore does not threaten the reported results.

### 6.3 Permutation tests --- WELL-DESIGNED

- Exhaustive enumeration for small k with correct Stirling number counts (verified by tests)
- Canonical relabelling prevents redundant computation
- Monte Carlo for large k with fixed group sizes
- The Webb comparison-level permutation (k=306, 10,000 draws) is adequate
- The Greenwald domain-level permutation (S(9,4) = 7,770) is a genuine addition that tests at the honest effective k

### 6.4 Metafor cross-validation --- COMPLETE

The R scripts exist and have been executed. 198 parameters compared across 6 datasets:
- 144 PASS (exact agreement within tolerance)
- 54 VARIANT (known formula differences: I2 formula, KH SE scaling, QM chi2 vs F, CR2 df method, profile-likelihood bounds)
- 0 FAIL

This pre-empts the single most predictable reviewer objection.

### 6.5 LOO cross-validation --- APPROPRIATE

- Out-of-sample R2 with 10,000 null-model simulations: correct design
- Null median LOO-R2 is negative in every case (random partitions predict worse than the mean), confirming the DLN coding carries genuine information
- The LOO-R2 values (0.19-0.89) are appropriately interpreted as varying with stage separation clarity

---

## 7. COMPUTATIONAL MODEL ASSESSMENT

### 7.1 Agent architectures --- WELL-DIFFERENTIATED

| Agent | State | Learning | Cost | Key property |
|-------|-------|----------|------|-------------|
| DotAgent | last_reward only | None (softmax on recency) | O(1) | Reactive; no cross-option learning |
| LinearAgent | K independent Q-values | TD learning (lr=0.15) | O(K) | Compartmentalised; verified isolation |
| NetworkAgent | F shared factor beliefs | Factor-gradient update (lr=0.12) | O(F) + structural | Integrative; verified cross-option generalisation |

### 7.2 Verified properties
- DotAgent: no cross-option learning (updating one arm does not change others)
- LinearAgent: Q-values converge independently (verified in test_agents.py)
- NetworkAgent: updating one option influences others sharing factor loadings (verified)
- Regime ordering: Network > Linear when K >> F and latent structure exists (verified across seeds)

### 7.3 Assessment
The computational layer is appropriately framed as a minimal scaffold for generating qualitative regime predictions. It is not a full reproduction of the DLN computational framework. The qualitative ordering (network > linear under shared structure) follows from the architectural difference (shared vs. independent representations) and is robust across the seeds tested. A parameter sensitivity sweep would strengthen confidence.

---

## 8. TEST SUITE ASSESSMENT

### 8.1 Coverage --- COMPREHENSIVE

14 test files, 261 tests. Coverage includes:
- **Statistical correctness**: CI coverage tests (400 simulations each for small-k and large-k), REML convergence, heterogeneity reduction
- **Numerical stability**: Softmax overflow/underflow, extreme variances, large rewards, zero temperature
- **Agent invariants**: DotAgent isolation, LinearAgent Q-convergence, NetworkAgent cross-option generalisation, regime orderings
- **Real data validation**: All extraction CSVs verified for schema compliance, DLN coding consistency, published-value matching
- **Permutation validity**: Stirling number counts, canonical relabelling, random-grouping non-significance
- **LOO cross-validation**: Positive OOS R2 vs null distribution
- **Blind coding consistency**: Agreement between author codes and validated codes
- **Fisher-z guards**: Bounds checking for correlation-to-Fisher-z conversions

### 8.2 Notable gaps
- No parametric sweep tests for the computational model (vary K, F, T, lr, threshold)
- No edge-case tests for the three-level model (e.g., singleton clusters, perfectly collinear moderators)
- No test of agent behaviour under non-stationarity (changing factor structure)
- No regret bounds or convergence-rate validation for agents
- Tests validate statistical consistency but not the external validity of DLN coding decisions

---

## 9. REPOSITORY STRUCTURE AND HYGIENE

### 9.1 Architecture
Three cleanly separated research layers:
- **Evidence synthesis** (`evidence_synthesis/`): 38 analysis scripts, 13+ protocol documents, 12+ extraction CSVs, 19+ output figures, 20+ output tables
- **Empirical tests** (`empirical_tests/`): Protocol, preregistration template, synthetic data, analysis script
- **Computational** (`computational/`): Three agent classes, factor-bandit environment, simulation grid

### 9.2 Code quality
- Modular pipeline: `meta_pipeline.py` provides core REML/forest/Egger infrastructure; dataset-specific scripts wire it to extraction data
- Explicit random seeds throughout (deterministic reproduction)
- Path management via `pathlib.Path`
- Dependencies pinned to exact versions
- Output directories created with `mkdir(parents=True, exist_ok=True)`
- No circular imports, no global mutable state

### 9.3 Remaining hygiene issues
- No Dockerfile or conda environment file (mentioned in prior reviews but still absent --- low priority)
- Hard-coded paths to repo structure (not problematic for reproducibility but limits portability)
- The `REVIEW.md`, `RESEARCH_REVIEW.md`, `BLOCKER_FIX_PLAN.md`, and `NHB_SUBMISSION_READINESS.md` files are internal review documents. Consider whether these should be excluded from the public-facing repository at submission time, or retained as transparency artifacts.

---

## 10. COMPARISON WITH PRIOR REVIEWS

Two prior reviews exist in this repository (REVIEW.md and the previous RESEARCH_REVIEW.md). Both identified critical blockers:

| Blocker | Status | Verification |
|---------|--------|-------------|
| Metafor cross-validation | RESOLVED | 198 parameters, 0 FAIL |
| Greenwald blinded IRR | RESOLVED | 9/9 items, both coders agreed |
| Numerical errors (lines 333, 620) | RESOLVED | Verified in current manuscript |
| Mixed/unclear handling documentation | RESOLVED | Added to Methods M2 |
| Interoception variance formula documentation | RESOLVED | Added to Methods M7 |
| Silent exception handling in permutation code | RESOLVED | Tightened to ValueError/LinAlgError only |
| Dependency pinning | RESOLVED | Exact versions in requirements.txt |

All blockers and advisories from prior reviews have been addressed. This review identifies no new blockers.

---

## 11. ANTICIPATED REVIEWER OBJECTIONS AND PREPAREDNESS

| Objection | Preparedness | Key evidence |
|-----------|-------------|-------------|
| Custom REML, not metafor | HIGH | 198-parameter cross-validation, 0 FAIL |
| Author-coded data | HIGH | 27/27 blinded IRR match, 47/47 sensitivity scenarios |
| Small k (7-10) | HIGH | Permutation tests (4/5 sig), LOO-CV (all positive), convergent pattern, k=306 anchor |
| Post-hoc coding | HIGH | Blinded IRR, sensitivity analysis, permutation tests, structural rationales |
| Circularity | HIGH | Codes based on structural features, competing moderator comparison, cross-domain generalisation |
| Mixed/unclear handling | HIGH | Documented in Methods, sensitivity analyses reported |
| No individual-level data | MEDIUM | Deposited protocol with power analysis; acknowledged as scope limitation |
| DLN companion paper under review | HIGH | bioRxiv preprint available; present paper is independent empirical test |
| Single-author paper | MEDIUM | IRR validates coding; all data deposited; metafor cross-validates pipeline |
| Computational model ad hoc | MEDIUM | Framed as scaffold; no quantitative claims from simulations |

---

## 12. PRIORITISED RECOMMENDATIONS

### Should-do before submission

1. **Strengthen the circularity discussion.** Add 3-4 sentences explicitly addressing the strongest version of the objection (Section 5.1 above).

2. **Soften the 100% reduction interpretation for Hoyt.** Add a sentence noting model saturation at k=8 (Section 5.3 above).

3. **Acknowledge the linear/linear-plus boundary as author-only coding.** One sentence in the Greenwald discussion (Section 5.6 above).

4. **Run computational parameter sensitivity sweep.** Vary lr, threshold, cost_weight; report whether qualitative ordering holds. This is a strong addition for ~2 hours of work.

### Nice-to-have

5. **Add a Dockerfile** for containerised reproducibility.

6. **Derive or sensitivity-check the interoception variance formula** (Section 5.5 above).

7. **Report funnel plots** alongside Egger's test results for visual assessment.

8. **Clean up internal review documents** (REVIEW.md, BLOCKER_FIX_PLAN.md, etc.) before public release --- either remove or consolidate into a single transparency log.

9. **Complete the preregistration template** (`prereg_template.md` is still a skeleton). The cumulative exposure protocol is fully specified, but the generic template should be filled in if it will be part of the public deposit.

---

## 13. BOTTOM LINE

This is an excellent research project. The core claim --- that cognitive architecture moderates emotion-cognition relationships --- is supported by convergent evidence across four domains with appropriate statistical safeguards. The dangerous-middle prediction is theoretically distinctive. The transparency infrastructure (261 passing tests, deposited materials, blinded IRR, metafor cross-validation, 47/47 sensitivity scenarios) substantially exceeds field norms.

The project has addressed all critical issues raised by prior reviews. The remaining concerns are moderate (circularity framing, Hoyt interpretation, computational parameter sensitivity) and can be addressed with targeted revisions.

**Recommendation:** Suitable for Nature Human Behaviour submission. The four "should-do" recommendations above can be addressed in 1-2 days. The cross-domain convergence and dangerous-middle prediction should remain foregrounded as the paper's most compelling contributions.

**Key risk:** The most likely outcome at NHB is a favourable R1 with requests for (a) additional blinded coding of the Greenwald study-level linear/linear-plus boundary, (b) individual-level pilot data, and/or (c) a formal pre-registration of the cumulative exposure experiment. Items (a) and (c) are feasible within a revision cycle; item (b) would require a longer timeline.
