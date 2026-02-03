"""Run DLN simulations (dot vs linear vs network) and write outputs.

This script produces synthetic regime evidence showing when network (integrative)
processing outperforms linear (compartmentalized) processing under shared structure,
supporting the fusion model's predictions.

Outputs:
- computational/outputs/simulation_summary.csv
- computational/outputs/simulation_plot.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment import EnvConfig, FactorBanditEnv
from src.agents import DotAgent, LinearAgent, NetworkAgent


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_once(K: int, F: int, T: int, seed: int = 0) -> dict:
    env = FactorBanditEnv(EnvConfig(K=K, F=F, seed=seed))
    rng = np.random.default_rng(seed + 123)

    dot = DotAgent(K=K, rng=rng)
    lin = LinearAgent(K=K, rng=rng)
    net = NetworkAgent(K=K, F=F, option_loadings=env.L, rng=rng)

    agents = {"dot": dot, "linear": lin, "network": net}

    out = {"K": K, "F": F, "T": T, "seed": seed}
    for name, agent in agents.items():
        total_reward = 0.0
        total_cost = 0.0
        for _ in range(T):
            a = agent.choose()
            r = env.step(a)
            step = agent.update(a, r)
            total_reward += step.reward
            total_cost += step.cost
        out[f"reward_{name}"] = total_reward / T
        out[f"cost_{name}"] = total_cost / T
        out[f"net_{name}"] = (total_reward - 0.01 * total_cost) / T  # small cost penalty
    out["structural_updates_network"] = getattr(net, "structural_updates", 0)
    return out


def main():
    # Grid: when K grows while structure size F stays modest, network should gain advantage.
    grid = []
    for K in [4, 8, 16, 32]:
        for F in [2, 4, 8]:
            for seed in range(10):
                grid.append(run_once(K=K, F=F, T=400, seed=seed))

    df = pd.DataFrame(grid)
    out_csv = OUT_DIR / "simulation_summary.csv"
    df.to_csv(out_csv, index=False)

    # Plot: net utility vs K for each agent (averaged over seeds), for a representative F
    F_plot = 4
    sub = df[df["F"] == F_plot].groupby("K").mean(numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(sub["K"], sub["net_dot"], marker="o", label="dot")
    ax.plot(sub["K"], sub["net_linear"], marker="o", label="linear")
    ax.plot(sub["K"], sub["net_network"], marker="o", label="network")

    ax.set_xlabel("Number of options (K)")
    ax.set_ylabel("Net utility (reward - cost penalty)")
    ax.set_title(f"DLN simulation (F={F_plot})")
    ax.legend()
    fig.tight_layout()

    out_fig = OUT_DIR / "simulation_plot.png"
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    print(f"Wrote: {out_csv}\nWrote: {out_fig}")


if __name__ == "__main__":
    main()
