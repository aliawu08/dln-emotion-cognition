# Cognitive Architecture as Hidden Moderator: Reconciling Contradictory Emotion–Cognition Findings with the DLN Framework

This repository accompanies the paper *"Cognitive Architecture as Hidden Moderator: Reconciling Contradictory Emotion–Cognition Findings with the Dot–Linear–Network (DLN) Framework"* (Wu, 2026).

The paper proposes that DLN cognitive stage acts as a **hidden moderator** explaining why emotion–cognition findings appear contradictory: emotion functions as noise under linear suppression but as signal under network integration.

**Preprint:** [PsychArchives](https://psycharchives.org/en/item/b4dc1d67-acb7-4327-939e-bc2ca0ef54b4) | DOI: [10.23668/psycharchives.21641](https://doi.org/10.23668/psycharchives.21641)

**Contact:** Alia Wu — wut08@nyu.edu
**ORCID:** [0009-0005-4424-102X](https://orcid.org/0009-0005-4424-102X)

---

## Key Claims

| DLN Stage | Emotion–Cognition Pattern | Prediction |
|-----------|---------------------------|------------|
| **Dot** | Reactive separation | Stimulus-driven reactivity; emotional opacity; poor decision quality across load levels |
| **Linear** | Suppressive compartmentalization | Emotion treated as interference; suppression costs; brittleness under high affective load |
| **Network** | Integrative fusion | Emotion as information; cognitive contextualization; flexible deployment; smallest implicit–explicit gaps |

---

## Repository Structure

```
├── manuscript/
│   ├── nhb_analysis.tex             # LaTeX source (canonical)
│   ├── nhb_analysis.pdf             # Compiled PDF
│   ├── nhb_supplementary.tex/pdf    # Supplementary materials
│   ├── cover_letter_nhb.tex/pdf     # Cover letter
│   └── references.bib               # Bibliography
├── evidence_synthesis/
│   ├── protocol/                    # Coding manuals, rubrics, screening tools (22 files)
│   ├── extraction/                  # Study-level data extraction (11 CSVs)
│   │   └── blinded/                 # Blinded coding sheets for IRR
│   ├── analysis/                    # Meta-analysis pipeline (35 scripts)
│   │   └── validate_metafor/        # R cross-validation against metafor
│   └── outputs/
│       ├── tables/                  # 37 result CSVs
│       └── figures/                 # 19 forest/permutation plots
├── empirical_tests/
│   ├── design/                      # Experiment protocol
│   ├── preregistration/             # OSF-style prereg template
│   ├── data/                        # Synthetic demo + published supplementary data
│   ├── analysis/                    # Analysis scripts
│   ├── protocol/                    # Cumulative exposure protocol
│   └── outputs/                     # Generated results
├── computational/
│   ├── agents.py                    # Dot / Linear / Network agent implementations
│   ├── environment.py               # Simulation environment
│   ├── run_simulation.py            # Main simulation script
│   └── outputs/                     # Simulation results
├── figures/
│   ├── scripts/
│   │   ├── contradiction_map.py     # Figure 1 generation
│   │   └── suppression_fingerprint.py  # Waterfall + fingerprint figures
│   └── export/                      # 6 files (PDF + PNG for each figure)
├── tests/                           # 12 test modules, 261 tests (pytest)
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
└── LICENSE, LICENSE-CODE
```

---

## Quickstart

**Requirements:** Python ≥ 3.8

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run evidence synthesis demo
python evidence_synthesis/analysis/run_meta_example.py

# Run empirical analysis demo
python empirical_tests/analysis/analyze_experiment1.py

# Run computational simulation
python computational/run_simulation.py

# Regenerate Figure 1 (contradiction map)
python figures/scripts/contradiction_map.py

# Regenerate suppression fingerprint + model waterfall figures
python figures/scripts/suppression_fingerprint.py
```

---

## Three-Layer Research Program

1. **Evidence synthesis layer** — Re-analysis of six published datasets (Webb 2012, Greenwald 2009, Hoyt 2024, De Smedt 2022, interoception meta-analysis, Zanini 2025) with 22 protocol documents, 11 extraction CSVs, blinded IRR coding, and 35 analysis scripts producing 37 result tables and 19 forest/permutation plots.

2. **Empirical layer** — Preregistration-ready experiment protocol testing the Stage × Affective Load interaction prediction, with synthetic demo data and published supplementary datasets for secondary analysis.

3. **Computational layer** — Runnable DLN-style simulations (dot vs. linear vs. network agents) that generate regime predictions consistent with the fusion model, with full pytest coverage of agent behaviour and numerical stability.

---

## Validation

**Test suite:** 12 test modules, 261 tests (run `pytest` from repo root). Covers agent behaviour, meta-analysis numerics, permutation inference, leave-one-out cross-validation, competing moderators, and profile confidence intervals.

**Cross-validation against R/metafor:** 7 R scripts in `evidence_synthesis/analysis/validate_metafor/` independently replicate the Python meta-analysis pipeline against the R `metafor` package. Results: 198 parameters compared, 0 failures, 144 exact matches (PASS), 54 documented methodological variants (e.g., Knapp–Hartung vs. Wald CI). See `validation_report.csv` for the full comparison.

---

## Citation

```bibtex
@misc{wu_dln_emotion_cognition_2026,
  title  = {Cognitive Architecture as Hidden Moderator:
            Reconciling Contradictory Emotion–Cognition Findings
            with the Dot–Linear–Network (DLN) Framework},
  author = {Wu, Alia},
  year   = {2026},
  doi    = {10.23668/psycharchives.21641}
}
```

---

## Related Work

This paper builds on the DLN computational framework:

> Wu, A. (2026). *Compression Efficiency and Structural Learning as a Computational Model of DLN Cognitive Stages*.
> bioRxiv: [10.64898/2026.02.01.703168](https://www.biorxiv.org/content/10.64898/2026.02.01.703168v2)
> Repository: [github.com/aliawu08/dln-compression-model](https://github.com/aliawu08/dln-compression-model)

---

## License

- **Code** (Python scripts, analysis pipelines): [MIT License](LICENSE-CODE)
- **Manuscript, documentation, and data**: [CC-BY-4.0](LICENSE)
