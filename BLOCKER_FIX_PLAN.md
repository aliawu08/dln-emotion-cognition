# Blocker Fix Plan

## Execution tiers

### TIER 1: Immediate manuscript fixes (can do now, ~20 min)

These are text edits to `manuscript/nhb_analysis.tex` and `manuscript/cover_letter_nhb.tex`.

**Fix 3a — Line 333: Remove false superlative**

Current (line 332-334):
```latex
($S(7,2) = 63$ two-way partitions; $p = 0.111$) achieved the
strongest absolute $\hat\tau^2$ reduction (83\%) among the
cross-domain datasets, but with only 63 possible partitions
```

Change to:
```latex
($S(7,2) = 63$ two-way partitions; $p = 0.111$) achieved a
large absolute $\hat\tau^2$ reduction (83\%) among the
cross-domain datasets, but with only 63 possible partitions
```

Rationale: Hoyt has 100% and interoception has 85%. Desmedt's 83% is third, not first.

**Fix 3b — Line 620: Correct 84% to 83%**

Current (line 620):
```latex
100\% vs.\ 71\% for temporal frame in Hoyt; and 84\% tied with
```

Change to:
```latex
100\% vs.\ 71\% for temporal frame in Hoyt; and 83\% tied with
```

Rationale: Data shows 82.6%, Table 5 rounds to 83%. Text should match the table.

**Fix 1-partial — Lines 1078-1080: Correct metafor reference**

Current (lines 1078-1080):
```latex
levels for both small and large $k$; see test suite). R code for
replication with the metafor package\cite{Viechtbauer2010} is provided
in \texttt{validate\_reml.py}.
```

Change to:
```latex
levels for both small and large $k$; see test suite). R scripts for
independent replication with the metafor package\cite{Viechtbauer2010}
are deposited in the project repository
(\texttt{validate\_metafor/*.R}).
```

Rationale: `validate_reml.py` is the Python comparison script, not the R code. The R scripts are at `validate_metafor/validate_*.R`. This also softens the claim from "validated against metafor" (which hasn't been done yet) to "deposited for independent replication" (which is true).

**Fix advisory 1 — Methods M2, after line 1027: Document mixed_unclear treatment**

After line 1027 (`subjected to sensitivity analysis.`), insert:

```latex
In the Greenwald domain-level analysis, items from topics coded
mixed/unclear were retained in the model; the design matrix assigns
these to the reference category (dot), functionally treating them as
unmoderated observations. Sensitivity analyses excluding these items
($k = 125$) and reclassifying them as linear are reported in
the Greenwald summary tables.
```

**Fix advisory 2 — Methods M7, interoception subsection (line ~1220): Document variance formula**

After line 1220 (`multi-dimensional integrative awareness (network).`), insert:

```latex
Because each row represents an aggregated measure family ($k_i$
studies, $N_{\text{total},i}$ participants), sampling variance was
approximated as $v_i = 1/(N_{\text{total},i} - 3k_i)$, adjusting the
standard Fisher-$z$ variance for the aggregation level.
```

**Fix advisory 5 — Cover letter line 36: Greenwald framing**

Current (line 36):
```latex
heterogeneity ($\hat\tau^2$) by 70--100\% in four analyses.
```

Change to:
```latex
heterogeneity ($\hat\tau^2$) by 70--100\% in four category-level
analyses and by 32.9\% at the study level in the largest dataset
($k = 184$, implicit cognition).
```

### TIER 2: Metafor cross-validation (requires R environment, ~2 hours)

Cannot be done in this environment (R not installed). Requires:

1. Install R with `metafor` and `clubSandwich` packages
2. Run: `Rscript evidence_synthesis/analysis/validate_metafor/run_all.R`
   - This executes 6 dataset-specific R scripts
   - Each writes a CSV to `validate_metafor/output_*.csv`
3. Run: `python evidence_synthesis/analysis/validate_reml.py`
   - Reads the R output CSVs, compares against Python values
   - Writes updated `validation_report.csv` with PASS/FAIL per parameter
   - Tolerance thresholds: tau2 < 0.001, beta < 0.005, SE < 0.005, CI < 0.01

Expected outcome: All 200 parameters should PASS within tolerance. If any FAIL, the Python implementation needs debugging before submission.

After metafor validation passes, update the manuscript Software section (line 1076-1080) to state:
```latex
The implementation was validated against the metafor R
package\cite{Viechtbauer2010} across all six datasets; all 200
parameters agreed within pre-specified tolerances (see project
repository). R replication scripts are deposited at
\texttt{validate\_metafor/*.R}.
```

### TIER 3: Greenwald blinded IRR (requires human coders, ~1-2 weeks)

Cannot be done computationally. Requires:

1. Recruit 2 independent coders naive to DLN
2. Provide them with:
   - The coding rubric at `evidence_synthesis/protocol/greenwald2009_coding_rubric.md`
   - Blinded extraction data (9 Greenwald topic domains, stripped of DLN codes and effect sizes)
   - The 3-question decision tree and neutral category labels (A/B/C/D for 4-level coding)
3. Collect independent codes, compute Cohen's kappa
4. Resolve disagreements via consensus discussion
5. Compare consensus to author's original codes

Infrastructure already exists: the rubric is deposited, the blinding/consensus protocol matches what was done for Webb and Hoyt. This is 9 items — small enough for a single session.

After completion, update Methods M10 to include Greenwald IRR results and remove the "future work" qualification at line 901-903.

## Execution order

```
TIER 1 (now)
  ├── Fix 3a: line 333 superlative          ✎ nhb_analysis.tex
  ├── Fix 3b: line 620 percentage           ✎ nhb_analysis.tex
  ├── Fix 1-partial: metafor reference      ✎ nhb_analysis.tex
  ├── Advisory 1: mixed_unclear sentence    ✎ nhb_analysis.tex
  ├── Advisory 2: interoception variance    ✎ nhb_analysis.tex
  └── Advisory 5: cover letter Greenwald    ✎ cover_letter_nhb.tex

TIER 2 (next, with R)
  └── Run metafor validation suite          ⚙ R + Python
      └── Update manuscript Software claim  ✎ nhb_analysis.tex

TIER 3 (parallel, with people)
  └── Commission Greenwald blinded IRR      👥 2 coders
      └── Update Methods M10                ✎ nhb_analysis.tex
```

## Submit decision

- After Tier 1 alone: Submittable with caveats. The manuscript honestly discloses both limitations. Risk: reviewers ask for metafor validation and Greenwald IRR, adding a revision cycle.
- After Tiers 1+2: Stronger. Metafor validation defuses the #1 methodological objection. Greenwald IRR remains a likely R1 request.
- After all three tiers: Optimal. Preempts both major reviewer concerns.
