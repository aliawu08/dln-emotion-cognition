# Evidence synthesis analysis

This folder contains a lightweight, runnable pipeline for the **DLN-stage moderator meta-analysis** program.

## Quick start (synthetic demo)
From the repo root:

```bash
python evidence_synthesis/analysis/run_meta_example.py
```

Outputs:
- `evidence_synthesis/outputs/tables/meta_summary_example.csv`
- `evidence_synthesis/outputs/figures/stage_effects_example.png`

## Replace with real extraction data
1. Fill `evidence_synthesis/extraction/study_level_extraction_template.csv`
2. Export a long-form effect size table with at least:
   - `paradigm_family`, `dln_stage_code`, `yi`, `vi`
3. Point `run_meta_example.py` to your real CSV

## Extensions you likely want next
- Three-level / correlated-effects models (multiple effects per study)
- Robust variance estimation (RVE)
- Bias/sensitivity analyses

The provided code is a transparent starting point; it is not intended to be the final statistical engine for a publication-grade meta-analysis.
