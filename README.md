# DLN Emotion–Cognition: Evidence Synthesis and Implementation

This repository contains **submission-ready manuscript files** and a complete set of **open-science deliverables** that implement the three-layer research program described in the manuscript:

1. **Evidence synthesis layer**: umbrella + moderator meta-analysis templates, extraction sheets, and runnable analysis code.
2. **Empirical layer**: preregistration-ready experiment protocol + analysis template (with synthetic demo data).
3. **Computational layer**: runnable DLN-style simulations (dot vs linear vs network) that generate regime predictions.

## Canonical source of truth
- **arXiv/LaTeX submission:** `manuscript/paper2.tex` (builds to `manuscript/paper2.pdf`)
- **Word/APA submission:** `manuscript/Paper2_Emotion_Cognition_SubmissionReady.docx` (with PDF export)

## Manuscript
- `manuscript/Paper2_Emotion_Cognition_SubmissionReady.docx`
- `manuscript/Paper2_Emotion_Cognition_SubmissionReady.pdf`
- `manuscript/paper2.tex` (LaTeX version matching the DLN-series style)

## Quick start (run the deliverables)
### 1) Evidence synthesis demo
```bash
python evidence_synthesis/analysis/run_meta_example.py
```

### 2) Empirical test demo
```bash
python empirical_tests/analysis/analyze_experiment1.py
```

### 3) Computational demo
```bash
python computational/run_simulation.py
```

### 4) Regenerate Figure 1 (contradiction map)
```bash
python figures/source/contradiction_map.py
```

## Replace synthetic demo data with real extraction / study data
- Evidence synthesis:
  - Fill: `evidence_synthesis/extraction/study_level_extraction_template.csv`
  - Ensure you produce a long-form table with `yi`, `vi`, `dln_stage_code`, `paradigm_family`.
- Empirical:
  - Replace `empirical_tests/data/synthetic_experiment1_demo.csv` with real data in the same schema.
- Computational:
  - Adjust `K`, `F`, noise parameters, and learning hyperparameters in `computational/run_simulation.py`.

## Citation for the DLN computational framework
Wu, A. (2026). *Dot-Linear-Network: A Computational Framework for Cognitive Architecture*. DOI: 10.13140/RG.2.2.11937.26728

## License
This repo is released under CC BY 4.0 unless otherwise noted (see `LICENSE`).
