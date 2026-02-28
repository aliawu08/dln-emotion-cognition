"""Prepare blinded extraction files for human inter-rater coding.

Reads each extraction CSV and strips DLN stage codes, coding rationales,
and effect-size columns, producing blinded versions suitable for
independent coders naive to the DLN framework.

See evidence_synthesis/protocol/human_blind_coding_protocol.md for the
full coding protocol.

Outputs
-------
- evidence_synthesis/extraction/blinded/webb2012_blinded.csv
- evidence_synthesis/extraction/blinded/hoyt2024_blinded.csv

Usage
-----
  python evidence_synthesis/analysis/prepare_blind_coding_materials.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_DIR = ROOT / "evidence_synthesis" / "extraction"
BLINDED_DIR = EXTRACTION_DIR / "blinded"

# Columns to strip: DLN codes, effect sizes, coding rationales
STRIP_COLUMNS = {
    "dln_stage_code",
    "coding_rationale",
    "d_plus",
    "r_pooled",
    "r_approx",
    "yi",
    "vi",
    "se",
    "se_d",
    "se_r",
    "ci_lo",
    "ci_hi",
    "abs_r",
}


def blind_csv(input_path: Path, output_path: Path, extra_strip: set | None = None):
    """Read a CSV, strip sensitive columns, and write a blinded version."""
    df = pd.read_csv(input_path)
    to_strip = STRIP_COLUMNS | (extra_strip or set())
    cols_to_drop = [c for c in df.columns if c in to_strip]

    df_blinded = df.drop(columns=cols_to_drop, errors="ignore")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_blinded.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")
    print(f"  Original columns: {list(df.columns)}")
    print(f"  Stripped: {cols_to_drop}")
    print(f"  Retained: {list(df_blinded.columns)}")
    print()


def main():
    print("Preparing blinded extraction files for human coding\n")
    print("=" * 60)

    # Webb (2012): primary coding dataset
    blind_csv(
        EXTRACTION_DIR / "webb2012_strategy_extraction.csv",
        BLINDED_DIR / "webb2012_blinded.csv",
    )

    # Hoyt (2024): generalization dataset
    blind_csv(
        EXTRACTION_DIR / "hoyt2024_domain_extraction.csv",
        BLINDED_DIR / "hoyt2024_blinded.csv",
        extra_strip={"N_approx"},
    )

    print("=" * 60)
    print("Done. Blinded files are in:")
    print(f"  {BLINDED_DIR}")
    print()
    print("Next steps:")
    print("  1. Give blinded CSVs + rubrics to independent coders")
    print("  2. See protocol: evidence_synthesis/protocol/"
          "human_blind_coding_protocol.md")


if __name__ == "__main__":
    main()
