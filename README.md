# Cognitive Architecture as Hidden Moderator: Reconciling Contradictory Emotion–Cognition Findings with the DLN Framework

This repository accompanies the paper *"Cognitive Architecture as Hidden Moderator: Reconciling Contradictory Emotion–Cognition Findings with the Dot–Linear–Network (DLN) Framework"* (Wu, 2026).

The paper proposes that DLN cognitive stage acts as a **hidden moderator** explaining why emotion–cognition findings appear contradictory: emotion functions as noise under linear suppression but as signal under network integration.

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
│   ├── paper2.tex                   # LaTeX source (canonical)
│   └── paper2.pdf                   # Compiled PDF
├── evidence_synthesis/
│   ├── protocol/                    # Meta-analysis protocol and coding manual
│   ├── extraction/                  # Data extraction templates
│   ├── analysis/                    # Meta-analysis pipeline
│   └── outputs/                     # Generated tables and figures
├── empirical_tests/
│   ├── design/                      # Experiment protocol
│   ├── preregistration/             # OSF-style prereg template
│   ├── data/                        # Synthetic demo data
│   ├── analysis/                    # Analysis scripts
│   └── outputs/                     # Generated results
├── computational/
│   ├── src/                         # Agent implementations
│   ├── run_simulation.py            # Main simulation script
│   └── outputs/                     # Simulation results
├── figures/
│   ├── source/                      # Figure generation code
│   └── export/                      # PNG and PDF exports
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Quickstart

**Requirements:** Python ≥ 3.8

```bash
# Install dependencies
pip install -r requirements.txt

# Run evidence synthesis demo
python evidence_synthesis/analysis/run_meta_example.py

# Run empirical analysis demo
python empirical_tests/analysis/analyze_experiment1.py

# Run computational simulation
python computational/run_simulation.py

# Regenerate Figure 1 (contradiction map)
python figures/source/contradiction_map.py
```

---

## Three-Layer Research Program

1. **Evidence synthesis layer**: Umbrella + moderator meta-analysis templates, extraction sheets, and runnable analysis code demonstrating how DLN stage can be coded as a moderator of heterogeneity.

2. **Empirical layer**: Preregistration-ready experiment protocol testing the Stage × Affective Load interaction prediction (with synthetic demo data).

3. **Computational layer**: Runnable DLN-style simulations (dot vs. linear vs. network agents) that generate regime predictions consistent with the fusion model.

---

## Citation

```bibtex
@misc{wu_dln_emotion_cognition_2026,
  title  = {Cognitive Architecture as Hidden Moderator:
            Reconciling Contradictory Emotion–Cognition Findings
            with the Dot–Linear–Network (DLN) Framework},
  author = {Wu, Alia},
  year   = {2026}
}
```

---

## Related Work

This paper builds on the DLN computational framework:

> Wu, A. (2026). *Compression Efficiency and Structural Learning as a Computational Model of DLN Cognitive Stages*.
> DOI: [10.13140/RG.2.2.11937.26728](https://doi.org/10.13140/RG.2.2.11937.26728)
> Repository: [github.com/aliawu08/dln-compression-model](https://github.com/aliawu08/dln-compression-model)

---

## License

- **Code** (Python scripts, analysis pipelines): [MIT License](LICENSE-CODE)
- **Manuscript, documentation, and data**: [CC-BY-4.0](LICENSE)
