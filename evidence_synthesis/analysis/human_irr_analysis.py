"""Inter-rater reliability analysis for human blinded DLN coding.

Computes agreement between independent human coders and author-original
codes for the Webb (2012) and Hoyt (2024) datasets.

Reads coder response sheets from evidence_synthesis/protocol/ and
author codes from the extraction CSVs.  Produces:
  - Console report with agreement tables, Cohen's kappa, weighted kappa
  - CSV output: evidence_synthesis/outputs/tables/human_blind_coding_results.csv

Usage
-----
  python evidence_synthesis/analysis/human_irr_analysis.py

When only one coder has returned data, the script reports coder-vs-author
agreement.  When both coders are available it additionally reports the
primary inter-rater metric (coder-1 vs coder-2 kappa).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "evidence_synthesis" / "protocol"
EXTRACTION = ROOT / "evidence_synthesis" / "extraction"
OUT_DIR = ROOT / "evidence_synthesis" / "outputs" / "tables"

# Mapping from blinded labels to DLN stage codes
LABEL_TO_DLN = {"A": "dot", "B": "linear", "C": "network"}
DLN_TO_LABEL = {v: k for k, v in LABEL_TO_DLN.items()}
ORDERED_LABELS = ["A", "B", "C"]


# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

def load_author_codes() -> dict[str, dict[str, str]]:
    """Load author-original DLN codes, return as {dataset: {item: label}}."""
    author = {}

    # Webb
    webb = pd.read_csv(EXTRACTION / "webb2012_strategy_extraction.csv")
    author["Webb"] = {
        row["strategy_sub"]: DLN_TO_LABEL[row["dln_stage_code"]]
        for _, row in webb.iterrows()
    }

    # Hoyt
    hoyt = pd.read_csv(EXTRACTION / "hoyt2024_domain_extraction.csv")
    author["Hoyt"] = {
        row["health_domain"]: DLN_TO_LABEL[row["dln_stage_code"]]
        for _, row in hoyt.iterrows()
    }

    return author


def _read_response_csv(path: Path) -> pd.DataFrame:
    """Read a coder response CSV, handling both comma and semicolon delimiters
    and optional title rows."""
    # Read raw first line to detect title row / delimiter
    with open(path) as f:
        first_line = f.readline().strip()

    # If first line doesn't look like a header (no separator or no known column),
    # skip it (some files have a title row like "response_sheet_hoyt").
    skip = 0
    if "coder_name" not in first_line:
        skip = 1

    # Detect delimiter from the header row
    with open(path) as f:
        for _ in range(skip):
            f.readline()
        header_line = f.readline()
    sep = ";" if header_line.count(";") > header_line.count(",") else ","

    return pd.read_csv(path, sep=sep, skiprows=skip)


def load_coder_responses(dataset: str) -> list[pd.DataFrame]:
    """Load all coder response sheets for a dataset.

    Discovers files matching response_sheet_{dataset}*.csv in the protocol
    directory, loading each as a separate coder.  Handles comma and semicolon
    delimiters and optional title rows.

    Returns list of DataFrames, one per coder.
    """
    ds_lower = dataset.lower()
    if dataset == "Webb":
        item_col = "strategy_sub"
    elif dataset == "Hoyt":
        item_col = "health_domain"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Find all response sheets for this dataset
    paths = sorted(PROTOCOL.glob(f"response_sheet_{ds_lower}*.csv"))
    if not paths:
        return []

    coders = []
    for path in paths:
        df = _read_response_csv(path)
        if df.empty or "coder_name" not in df.columns:
            continue

        # Normalise category assignment
        df["category_assignment"] = df["category_assignment"].str.strip().str.upper()

        # Split by coder (a single file could contain multiple coders)
        for name in df["coder_name"].unique():
            coder_df = df[df["coder_name"] == name].copy()
            coder_df = coder_df.set_index(item_col)
            coders.append(coder_df)

    return coders


# ═══════════════════════════════════════════════════════════════════
# AGREEMENT METRICS
# ═══════════════════════════════════════════════════════════════════

def cohens_kappa(codes_a: list[str], codes_b: list[str],
                 categories: list[str] | None = None) -> float:
    """Cohen's kappa for two lists of categorical codes."""
    n = len(codes_a)
    if n == 0:
        return 0.0

    if categories is None:
        categories = sorted(set(codes_a) | set(codes_b))
    cat_idx = {c: i for i, c in enumerate(categories)}
    m = len(categories)

    cm = np.zeros((m, m), dtype=int)
    for a, b in zip(codes_a, codes_b):
        cm[cat_idx[a], cat_idx[b]] += 1

    p_o = np.trace(cm) / n
    p_e = sum(cm[i, :].sum() * cm[:, i].sum() for i in range(m)) / (n * n)

    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def weighted_kappa(codes_a: list[str], codes_b: list[str],
                   categories: list[str] | None = None) -> float:
    """Quadratic-weighted kappa for ordinal categories."""
    n = len(codes_a)
    if n == 0:
        return 0.0

    if categories is None:
        categories = sorted(set(codes_a) | set(codes_b))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    observed = np.zeros((k, k))
    for a, b in zip(codes_a, codes_b):
        observed[cat_idx[a], cat_idx[b]] += 1
    observed /= n

    row_marg = observed.sum(axis=1)
    col_marg = observed.sum(axis=0)
    expected = np.outer(row_marg, col_marg)

    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i][j] = (i - j) ** 2 / (k - 1) ** 2

    num = np.sum(weights * observed)
    den = np.sum(weights * expected)

    if den == 0:
        return 1.0
    return 1.0 - num / den


def confusion_matrix(codes_a: list[str], codes_b: list[str],
                     categories: list[str] = ORDERED_LABELS) -> np.ndarray:
    """Return confusion matrix as numpy array."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    m = len(categories)
    cm = np.zeros((m, m), dtype=int)
    for a, b in zip(codes_a, codes_b):
        cm[cat_idx[a], cat_idx[b]] += 1
    return cm


# ═══════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════

def print_comparison(label_a: str, label_b: str,
                     items: list[str], codes_a: list[str],
                     codes_b: list[str], dataset: str,
                     confidence: list[str] | None = None):
    """Print detailed comparison table and agreement metrics."""
    n = len(items)
    agree = sum(1 for a, b in zip(codes_a, codes_b) if a == b)
    kappa = cohens_kappa(codes_a, codes_b, ORDERED_LABELS)
    wkappa = weighted_kappa(codes_a, codes_b, ORDERED_LABELS)

    print(f"\n{'=' * 70}")
    print(f"{dataset}: {label_a} vs {label_b}")
    print(f"{'=' * 70}")

    # Item-level table
    header = f"{'Item':<35} {label_a:>8} {label_b:>8} {'Match':>5}"
    if confidence:
        header += f" {'Conf':>6}"
    print(header)
    print("-" * len(header))

    for i, item in enumerate(items):
        match = "Y" if codes_a[i] == codes_b[i] else ""
        row = f"{item:<35} {codes_a[i]:>8} {codes_b[i]:>8} {match:>5}"
        if confidence:
            row += f" {confidence[i]:>6}"
        print(row)

    # Summary
    print(f"\nRaw agreement:        {agree}/{n} ({agree/n:.0%})")
    print(f"Cohen's kappa:        {kappa:.3f}")
    print(f"Weighted kappa (q):   {wkappa:.3f}")

    # Interpret kappa (Landis & Koch 1977)
    if kappa >= 0.81:
        interp = "almost perfect"
    elif kappa >= 0.61:
        interp = "substantial"
    elif kappa >= 0.41:
        interp = "moderate"
    elif kappa >= 0.21:
        interp = "fair"
    else:
        interp = "slight/poor"
    print(f"Interpretation:       {interp} (Landis & Koch)")

    threshold = 0.70
    if kappa >= threshold:
        print(f"Threshold (kappa >= {threshold}): PASSED")
    else:
        print(f"Threshold (kappa >= {threshold}): NOT YET MET")

    # Confusion matrix
    cm = confusion_matrix(codes_a, codes_b)
    print(f"\nConfusion matrix ({label_a} rows x {label_b} cols):")
    print(f"{'':>8} {'A':>5} {'B':>5} {'C':>5}")
    for i, cat in enumerate(ORDERED_LABELS):
        print(f"{cat:>8} {cm[i,0]:>5} {cm[i,1]:>5} {cm[i,2]:>5}")

    # Disagreement details
    disagreements = [
        (items[i], codes_a[i], codes_b[i])
        for i in range(n) if codes_a[i] != codes_b[i]
    ]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for item, ca, cb in disagreements:
            dist = abs(ORDERED_LABELS.index(ca) - ORDERED_LABELS.index(cb))
            adj = " (adjacent)" if dist == 1 else f" ({dist}-step)"
            direction = "UP" if ORDERED_LABELS.index(cb) > ORDERED_LABELS.index(ca) else "DOWN"
            print(f"  {item:<30} {ca} -> {cb}  {direction}{adj}")
    else:
        print("\nNo disagreements — perfect agreement.")

    return {
        "dataset": dataset,
        "comparison": f"{label_a}_vs_{label_b}",
        "n_items": n,
        "agree": agree,
        "pct_agree": agree / n,
        "cohens_kappa": kappa,
        "weighted_kappa": wkappa,
        "interpretation": interp,
    }


# ═══════════════════════════════════════════════════════════════════
# OUTPUT CSV
# ═══════════════════════════════════════════════════════════════════

def build_results_table(author: dict, coders_by_dataset: dict) -> pd.DataFrame:
    """Build item-level results table for CSV output."""
    rows = []
    for dataset in ["Webb", "Hoyt"]:
        author_codes = author[dataset]
        coders = coders_by_dataset.get(dataset, [])

        for item in author_codes:
            row = {
                "dataset": dataset,
                "item": item,
                "author_code": author_codes[item],
                "author_dln": LABEL_TO_DLN[author_codes[item]],
            }

            for ci, coder_df in enumerate(coders, 1):
                if item in coder_df.index:
                    coder_row = coder_df.loc[item]
                    row[f"coder{ci}_code"] = coder_row["category_assignment"]
                    row[f"coder{ci}_confidence"] = coder_row.get("confidence", "")
                    row[f"coder{ci}_rationale"] = coder_row.get("rationale", "")
                    row[f"coder{ci}_agree_author"] = (
                        coder_row["category_assignment"] == author_codes[item]
                    )

            rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    author = load_author_codes()
    coders_by_dataset = {}
    summary_rows = []

    print("=" * 70)
    print("HUMAN BLINDED CODING — INTER-RATER RELIABILITY ANALYSIS")
    print("=" * 70)

    for dataset in ["Webb", "Hoyt"]:
        coders = load_coder_responses(dataset)
        coders_by_dataset[dataset] = coders
        items = list(author[dataset].keys())
        author_codes = [author[dataset][it] for it in items]

        if not coders:
            print(f"\n{dataset}: No coder responses found — skipping.")
            continue

        print(f"\n{'#' * 70}")
        print(f"# {dataset.upper()} — {len(items)} items, {len(coders)} coder(s)")
        print(f"{'#' * 70}")

        for ci, coder_df in enumerate(coders, 1):
            coder_name = coder_df["coder_name"].iloc[0]
            coder_codes = [coder_df.loc[it, "category_assignment"] for it in items]
            confidence = [
                str(coder_df.loc[it, "confidence"]) if "confidence" in coder_df.columns else ""
                for it in items
            ]

            result = print_comparison(
                "Author", f"Coder{ci} ({coder_name})",
                items, author_codes, coder_codes, dataset,
                confidence=confidence,
            )
            summary_rows.append(result)

        # Inter-rater: coder vs coder (when 2+ coders available)
        if len(coders) >= 2:
            c1_codes = [coders[0].loc[it, "category_assignment"] for it in items]
            c2_codes = [coders[1].loc[it, "category_assignment"] for it in items]
            c1_name = coders[0]["coder_name"].iloc[0]
            c2_name = coders[1]["coder_name"].iloc[0]

            result = print_comparison(
                f"Coder1 ({c1_name})", f"Coder2 ({c2_name})",
                items, c1_codes, c2_codes, dataset,
            )
            summary_rows.append(result)

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    if summary_rows:
        fmt = "{:<12} {:<30} {:>5} {:>8} {:>8} {:>8}  {:<15}"
        print(fmt.format("Dataset", "Comparison", "N", "%Agree",
                         "Kappa", "w-Kappa", "Interpretation"))
        print("-" * 90)
        for r in summary_rows:
            print(fmt.format(
                r["dataset"], r["comparison"], r["n_items"],
                f"{r['pct_agree']:.0%}", f"{r['cohens_kappa']:.3f}",
                f"{r['weighted_kappa']:.3f}", r["interpretation"],
            ))

        # Threshold check
        print()
        for r in summary_rows:
            status = "PASS" if r["cohens_kappa"] >= 0.70 else "PENDING"
            print(f"  {r['dataset']} {r['comparison']}: kappa={r['cohens_kappa']:.3f} [{status}]")
    else:
        print("No coder data available yet.")

    # ── Save CSV ──────────────────────────────────────────────────
    results_df = build_results_table(author, coders_by_dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "human_blind_coding_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

    return results_df


if __name__ == "__main__":
    main()
