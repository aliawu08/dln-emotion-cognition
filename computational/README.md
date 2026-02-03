# Computational layer

This folder provides a lightweight, runnable simulation that mirrors the DLN computational mapping used in the DLN series:

- **Dot**: reactive, minimal state
- **Linear**: K independent option-value estimates (no sharing)
- **Network**: shared latent structure (factorized representation) + a simplified structural learning cycle

## Run
From the repo root:

```bash
python computational/run_simulation.py
```

Outputs:
- `computational/outputs/simulation_summary.csv`
- `computational/outputs/simulation_plot.png`

## How this connects to Paper 2
Paper 2's core emotion–cognition claim is that different architectures yield different roles for affect:
- dot: reactive signals dominate, weak contextual integration
- linear: compartmentalization can preserve sequential reasoning but produces brittleness and “influence without awareness”
- network: integration supports flexible use of affect as information

The simulation here is not a full reproduction of Paper 1; it is a reproducible computational scaffold for generating testable regime predictions that can be aligned to emotion–cognition paradigms.
