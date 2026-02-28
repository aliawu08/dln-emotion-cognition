# Head of Research Review: DLN Emotion-Cognition Repository

**Paper:** "Cognitive architecture moderates emotion-cognition relationships: cross-domain evidence from meta-analytic reanalysis"
**Author:** Alia Wu (Redline Rising / Risk Efficacy; wut08@nyu.edu)
**Target venue:** Nature Human Behaviour
**Review date:** 2026-02-24
**Repo health:** 261/261 tests passing; single initial commit; well-organized three-layer structure

---

## 1. EXECUTIVE SUMMARY

This repository accompanies a paper proposing that the Dot-Linear-Network (DLN) cognitive architecture framework functions as a **hidden structural moderator** explaining why emotion-cognition findings appear contradictory across literatures. The project reanalyses six published meta-analyses spanning four domains (emotion regulation, implicit cognition, interoception, health psychology), supplemented by agent-based simulations and a deposited empirical protocol.

The cross-domain convergence is the paper's strongest card: DLN stage coding reduces heterogeneity by 70-100% in four category-level datasets, produces theoretically predicted sign reversals and directional orderings in five, and correctly predicts null moderation in a boundary-condition dataset. Permutation tests confirm non-arbitrary grouping in four of five datasets. A competing moderator analysis shows DLN outperforms established alternatives in every dataset --- and, critically, no alternative moderator generalises across domains.

This is a well-executed, transparent, and methodologically rigorous project with a genuinely novel theoretical contribution. The main vulnerabilities are small-k datasets, single-author coding for the largest analysis, and absence of individual-level empirical data.

---

## 2. WHAT THE PROJECT CLAIMS

The DLN framework models cognitive development as shifts in belief-dependency graph topology:

| Stage | Architecture | Emotion-Cognition Pattern |
|-------|-------------|--------------------------|
| **Dot** | Empty belief graph; O(1) memory | Reactive separation --- stimulus-driven, no integration |
| **Linear** | Null graph on K nodes; O(K) scaling | Suppressive compartmentalisation --- emotion as interference |
| **Network** | Bipartite factor DAG; O(F) scaling | Integrative fusion --- emotion as information |

The key testable claim: coding published meta-analytic data for DLN stage should reduce between-study heterogeneity that conventional moderators leave unexplained. A distinctive prediction --- the **dangerous middle** --- holds that partial integration without metacognitive monitoring produces *worse* outcomes than full compartmentalisation.

---

## 3. REPOSITORY STRUCTURE AND QUALITY

### 3.1 Architecture

Three cleanly separated research layers:

- **Evidence synthesis** (`evidence_synthesis/`): 38 Python analysis scripts, 13+ protocol/rubric documents, 56 CSV extraction files, 19+ output figures
- **Empirical tests** (`empirical_tests/`): Preregistered protocol, synthetic demo data (N=360), analysis script
- **Computational** (`computational/`): Three agent classes (DotAgent, LinearAgent, NetworkAgent), factor-bandit environment, simulation grid

### 3.2 Codebase health

- **261 tests, all passing** (14 test files covering REML correctness, CI coverage, agent invariants, permutation nulls, numerical stability)
- Explicit random seeds throughout (deterministic reproduction)
- Modular pipeline: `meta_pipeline.py` provides core REML/forest/Egger infrastructure; dataset-specific scripts wire it to extraction data
- Path management via `pathlib.Path` (though all paths are hard-coded to the repo structure)
- Output directories created with `mkdir(parents=True, exist_ok=True)`
- Dependencies: numpy, scipy, pandas, matplotlib, statsmodels --- standard scientific Python stack

### 3.3 Reproducibility

**Strengths:**
- All extraction files deposited with source annotations
- All coding rationales documented
- Blinded coding materials, response sheets, and consensus log included
- Test suite validates analytical properties (CI coverage at k=5 and k=30)

**Gaps:**
- Loose version pinning in `requirements.txt` (e.g., `numpy>=1.24`) --- no lockfile
- No Docker/conda environment specification
- Hard-coded directory paths limit portability
- REML implementation is custom (not `metafor`); `validate_reml.py` exists but cross-validation with established packages should be demonstrated

---

## 4. STATISTICAL METHODOLOGY

### 4.1 Core pipeline

The meta-analysis pipeline (`meta_pipeline.py`) implements:

- **REML random-effects meta-regression** with bounded scalar optimisation
- **Knapp-Hartung small-sample correction** (t-distribution CIs, appropriate for small k)
- **Cochran's Q, I-squared, tau-squared** with profile-likelihood CIs
- **Egger's regression** for publication bias
- **Cluster-robust (CR2 sandwich) standard errors** for nested comparisons
- **Three-level hierarchical models** (`multilevel_meta.py`)

### 4.2 Validation infrastructure

- **Exhaustive permutation tests**: Stirling number enumerations for small k (all S(n,k) partitions); 10,000-draw Monte Carlo for large k. This directly tests whether any arbitrary grouping achieves comparable variance reduction.
- **Leave-one-out cross-validation**: Out-of-sample R-squared with null-model simulation (10,000 random partitions)
- **Competing moderator analysis**: 2-3 alternatives per dataset (Gross process model, cognitive effort, Garfinkel trichotomy, construct valence, temporal frame, measurement objectivity, clinical relevance)
- **Coding sensitivity analysis**: 47 alternative coding scenarios across all datasets; qualitative pattern preserved in 47/47

### 4.3 Assessment

The statistical approach is appropriate and thorough. The permutation tests are the right safeguard for small-k analyses. The competing moderator comparison is well-designed. The LOO cross-validation with null simulation is a strong addition.

The main methodological concern is the custom REML implementation. While validated by the test suite, a reviewer will likely request `metafor` replication of key results.

---

## 5. KEY RESULTS

| Dataset | k | Domain | tau-squared reduction | Permutation p | Key pattern |
|---------|---|--------|-----------------------|---------------|-------------|
| Webb (strategy-family) | 10 | Emotion regulation | 70% | 0.025 | Network > Linear > Dot |
| Webb (comparison-level) | 306 | Emotion regulation | 10% | <0.001 | Same ordering, cluster-robust |
| Interoception | 8 | Interoception | 85% | 0.002 | Sign reversal (linear +, network -) |
| Hoyt | 8 | Health psychology | 100% | 0.008 | V-shape (dangerous middle) |
| Desmedt | 7 | Heartbeat counting | 83% | 0.111 | Same-level correspondence |
| Greenwald (4-level) | 184 | Implicit cognition | 33% | <0.001 | Suppression fingerprint |
| Zanini (boundary) | 110 | Affective decision-making | N/A | N/A | All dot --- null moderation confirmed |

DLN achieved the highest tau-squared reduction in every dataset in head-to-head comparison with established alternative moderators.

---

## 6. STRENGTHS

### 6.1 Rigorous falsification infrastructure
The paper explicitly defines disconfirmation criteria at multiple levels: for the coding rubric (kappa < 0.60), for the framework (no Stage x Load interaction), for circularity (stage must be assessed independently of outcomes). Supplementary Table 1 lists six specific falsifiable predictions with named measures and expected directions. This exceeds the norm for theory papers.

### 6.2 Cross-domain convergence
The same coding framework reduces heterogeneity across emotion regulation, interoception, health psychology, and implicit cognition. No alternative moderator achieves this generality. This is the paper's strongest argument and the one most likely to persuade reviewers.

### 6.3 Boundary-condition test
The Zanini dataset (k=110, all studies coded as dot) is a genuine scope-delimiting test. The framework correctly predicts where it should *not* work. This is rare in theory-driven papers and adds credibility.

### 6.4 Dangerous-middle prediction
The V-shaped pattern in Hoyt (linear worse than dot *and* network) is a direction-specific prediction not generated by monotonic stage models (LEAS, constructionism, integrative complexity, ACT). This is the paper's most theoretically distinctive contribution.

### 6.5 Transparency
All extraction files, coding rationales, blinded rubrics, response sheets, consensus logs, and sensitivity analyses are deposited. The 261-test suite validates the custom implementation. This substantially exceeds the field norm.

### 6.6 Blinded inter-rater reliability
Two naive coders replicated the author's codes on all 18 items (Webb + Hoyt). The IRR consensus log (`evidence_synthesis/protocol/irr_consensus_log.md`) is transparent about four initial disagreements and their resolution.

---

## 7. CONCERNS AND LIMITATIONS

### 7.1 Small-k vulnerability (MAJOR)

Four of five moderator analyses operate at k=7-10. With 2-3 parameters fitting 7-8 data points, model saturation is a genuine concern. The 100% tau-squared reduction in Hoyt (3 parameters, 8 points) should be interpreted cautiously despite the significant permutation test (p=0.008).

**Mitigations in manuscript:** Profile-likelihood CIs, LOO cross-validation, permutation tests. These are appropriate.

**Assessment:** The permutation tests are the right safeguard, and four of five being significant is meaningful. The comparison-level Webb analysis (k=306) provides the high-powered anchor. But the field should not treat the individual small-k results as independently persuasive --- the paper's strength is the convergent pattern.

### 7.2 Single-author coding (MAJOR)

All DLN stage assignments were initially made by the framework's creator. The blinded coding validation covers only Webb and Hoyt (18 items total), not the more complex Greenwald (184 study-level codes) or interoception (8 measures).

**Assessment:** The Greenwald coding is the weakest link: 184 study-level codes assigned by a single coder with no blinded replication, and the network stage contains only political behaviour studies (k=11). Independent blinded coding of the Greenwald dataset should be a high priority.

### 7.3 IRR consensus process (MODERATE)

The consensus log reveals that on two of four Webb disagreements (situation_selection, concentration), both coders initially coded B (linear), then revised to A (dot) after receiving "one-sentence construct descriptions" from the author. The construct clarifications appear legitimate (distinguishing rumination from controlled attention; reactive avoidance from deliberate planning), but they introduce potential demand characteristics.

**Assessment:** Reported transparently. The methodology is defensible --- providing construct definitions is standard practice in coding reliability studies. The key test is that the clarifications were factual descriptions of Webb's operationalisations, not theoretical arguments about DLN.

### 7.4 Post-hoc vs. confirmatory status (MODERATE)

Webb, interoception, Desmedt, and Hoyt analyses are explicitly acknowledged as exploratory. Only Greenwald and Zanini rubrics were pre-specified. The manuscript is transparent about this (Methods M5), but the cross-domain narrative can read more confirmatory than the evidence warrants.

**Assessment:** Handled well. The absence of any datasets where DLN coding was tried and *failed* is a concern the paper cannot fully address, but the permutation tests provide a formal safeguard against post-hoc selection bias.

### 7.5 Custom REML implementation (MODERATE)

The meta-analysis pipeline is hand-rolled in Python rather than using `metafor` (R). While the test suite validates CI coverage and analytical properties, a reviewer will likely request independent replication.

**Assessment:** `validate_reml.py` exists, suggesting this was planned. Completing the metafor cross-validation before submission would pre-empt the most likely methodological objection.

### 7.6 Agent model parameter sensitivity (MINOR)

The computational simulations use specific learning rates (LinearAgent lr=0.15, NetworkAgent lr=0.12) and a structure threshold (0.35) with no empirical calibration or sensitivity analysis.

**Assessment:** The computational layer is framed as generating qualitative regime predictions, not quantitative fits. This is appropriate. A brief sweep over hyperparameters would strengthen the argument.

### 7.7 Greenwald network-stage generalisability (MINOR)

Network stage in the Greenwald analysis is exclusively political behaviour (k=11). This limits the generalisability of the network-stage estimate for implicit cognition. Noted in the manuscript.

### 7.8 Empirical layer is prospective (MINOR)

The experimental protocol is deposited but unfilled. The synthetic demo data demonstrates what the analysis *would* look like, but no human data exists for individual-level predictions. This is acknowledged and appropriate for a theory-testing paper.

---

## 8. ASSESSMENT FOR NATURE HUMAN BEHAVIOUR

**Novelty:** High. The proposal that cognitive architecture (formalised as DLN) is a hidden moderator explaining contradictory emotion-cognition findings is original. The dangerous-middle prediction is genuinely distinctive --- no competing framework generates it.

**Evidence quality:** Moderate-to-strong. Cross-domain convergence across four domains with permutation validation is compelling. The comparison-level Webb analysis (k=306) provides adequate statistical power. Small-k analyses are individually suggestive rather than definitive, but the convergent pattern is meaningful.

**Presentation:** Well-written, appropriately structured for NHB format. Methods section is thorough. Limitations are handled candidly. The manuscript does not overstate its claims.

**Anticipated reviewer concerns:**
1. A methodologist will question the custom REML and request metafor replication
2. A meta-analysis specialist will focus on small-k and single-coder issues
3. A theory reviewer may question differentiation from simpler alternatives (partially addressed by competing moderator analysis)
4. Any reviewer may ask why the empirical experiment hasn't been run

---

## 9. RECOMMENDATIONS

### High priority

1. **Obtain blinded IRR for the Greenwald dataset.** This is the paper's largest analysis (k=184) and has no independent coding validation. The 31-scenario sensitivity analysis is helpful but does not substitute for independent coding.

2. **Cross-validate key results with metafor in R.** Complete the `validate_reml.py` script and report exact numerical agreement. This pre-empts the single most predictable methodological objection.

3. **Frame small-k results as a convergent pattern.** The paper already does this to some extent, but the abstract and Results could more explicitly emphasise that the persuasive unit is the *cross-domain convergence*, not any single small-k analysis.

### Medium priority

4. **Add parameter sensitivity analysis for agent simulations.** Sweep learning rates (0.05-0.30), temperature (0.5-1.5), and structure threshold (0.2-0.5); report whether the qualitative ordering (network > linear under shared structure) is robust.

5. **Complete the preregistration template.** The current `prereg_template.md` is a skeleton. Filling it in --- even before data collection begins --- signals methodological seriousness.

6. **Pin exact dependency versions.** Add a `requirements-lock.txt` or `conda-lock.yml` for exact reproducibility.

### Lower priority

7. **Add a Dockerfile** or `conda environment.yml` for containerised reproducibility.

8. **Report funnel plots** alongside Egger's test results for visual assessment of publication bias.

9. **Reframe the Desmedt result (p=0.111).** Present it as a qualitative pattern consistent with the framework rather than a formal test. The limited combinatorial resolution (63 possible partitions) makes significance testing uninformative.

10. **Consider a comparison-level or study-level analysis for Hoyt and interoception** (similar to the Webb k=306 analysis) to address the small-k concern, if the source data permit it.

---

## 10. BOTTOM LINE

This is a well-executed, transparent, and methodologically rigorous research project with a genuinely novel theoretical contribution. The core claim --- that cognitive architecture moderates emotion-cognition relationships --- is supported by convergent evidence across four domains, with appropriate safeguards against arbitrary grouping (permutation tests), coding bias (blinded IRR, sensitivity analysis), and alternative explanations (competing moderator comparison). The codebase quality (261 passing tests, modular structure, deposited materials) substantially exceeds the field norm.

The main vulnerabilities are the small-k datasets, single-author coding for the Greenwald analysis, and absence of individual-level empirical data. These are addressable limitations, not fatal flaws.

**Recommendation:** The project is suitable for a high-impact venue, contingent on addressing the Greenwald IRR gap and providing metafor cross-validation of key results. The dangerous-middle prediction and cross-domain convergence are the paper's most compelling contributions and should be foregrounded in the framing.
