"""Zanini et al. (2025) — DLN boundary-condition analysis.

Applies the pre-specified DLN coding rubric (zanini2025_coding_rubric.md)
to the Iowa Gambling Task sex-differences meta-analysis (k=110 studies).

Finding: All 110 studies used the standard 100-trial IGT with healthy naive
participants, placing every study at dot stage.  No between-stage variation
exists, so DLN predicts NO moderation—heterogeneity is entirely within-stage.

This script:
  1. Loads the boundary-analysis extraction.
  2. Verifies the coding (all dot, zero linear/network).
  3. Confirms the null prediction against Zanini et al.'s reported moderator
     results (none significant).
  4. Writes a structured summary to outputs/.

Outputs:
  - evidence_synthesis/outputs/tables/zanini2025_boundary_summary.csv
  - evidence_synthesis/outputs/tables/zanini2025_boundary_report.txt

Usage:
  python evidence_synthesis/analysis/run_zanini2025_boundary.py
"""

from pathlib import Path
import csv
import textwrap

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "zanini2025_boundary_analysis.csv"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "zanini2025_boundary_summary.csv"
OUT_REPORT = ROOT / "evidence_synthesis" / "outputs" / "tables" / "zanini2025_boundary_report.txt"


def main():
    # ------------------------------------------------------------------
    # 1. Load extraction
    # ------------------------------------------------------------------
    with open(DATA, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    rec = rows[0]

    # ------------------------------------------------------------------
    # 2. Verify DLN coding
    # ------------------------------------------------------------------
    k_total = int(rec["k_total"])
    k_dot = int(rec["k_dot"])
    k_linear = int(rec["k_linear"])
    k_network = int(rec["k_network"])
    k_mixed = int(rec["k_mixed"])

    assert k_total == 110, f"Expected k=110, got {k_total}"
    assert k_dot == 110, f"Expected all dot, got k_dot={k_dot}"
    assert k_linear == 0, f"Expected 0 linear, got {k_linear}"
    assert k_network == 0, f"Expected 0 network, got {k_network}"
    assert k_mixed == 0, f"Expected 0 mixed, got {k_mixed}"
    assert rec["dln_variation"] == "none"

    print(f"DLN coding verified: {k_total} studies, all dot stage.")
    print(f"  k_dot={k_dot}, k_linear={k_linear}, k_network={k_network}")
    print(f"  Between-stage variation: {rec['dln_variation']}")

    # ------------------------------------------------------------------
    # 3. Verify null prediction against reported moderators
    # ------------------------------------------------------------------
    assert rec["any_moderator_significant"] == "no", (
        "Expected no significant moderators in Zanini et al."
    )
    moderators = rec["moderators_tested"].split("; ")
    print(f"\nZanini et al. tested {len(moderators)} moderators: {moderators}")
    print(f"Any significant: {rec['any_moderator_significant']}")

    # ------------------------------------------------------------------
    # 4. Report overall effect and heterogeneity
    # ------------------------------------------------------------------
    umd = float(rec["overall_umd"])
    tau2 = float(rec["tau2"])
    q = float(rec["q_stat"])
    q_df = int(rec["q_df"])

    print(f"\nOverall effect: UMD = {umd:.3f} (SE = {rec['overall_se']})")
    print(f"  95% CI [{rec['overall_ci_lo']}, {rec['overall_ci_hi']}]")
    print(f"Heterogeneity: Q({q_df}) = {q:.3f}, tau2 = {tau2:.3f}")
    print(f"  tau2 95% CI [{rec['tau2_ci_lo']}, {rec['tau2_ci_hi']}]")

    # ------------------------------------------------------------------
    # 5. DLN interpretation
    # ------------------------------------------------------------------
    interpretation = textwrap.dedent("""\
    DLN BOUNDARY-CONDITION ANALYSIS: Zanini et al. (2025)
    =====================================================

    CODING OUTCOME
    All 110 studies used the standard 100-trial IGT with healthy naive
    participants.  Per the pre-specified rubric (Step 3: standard IGT with
    naive participants -> dot), every study is coded as dot stage.

      k_dot     = 110
      k_linear  =   0
      k_network =   0
      Between-stage variation: NONE

    DLN PREDICTION
    Because all studies occupy a single DLN stage, there is no between-stage
    variation for DLN to explain.  DLN predicts:
      - No heterogeneity reduction from stage coding (tau2 reduction = 0%)
      - Residual heterogeneity reflects within-stage sources (individual
        differences, cultural factors, sample composition) that DLN is not
        designed to capture

    CONSISTENCY CHECK
    Zanini et al. tested 7 moderators (mean age, publication year, sample
    size, study quality, monetary reward, task version, region).
    Result: NONE significant.

    This is consistent with the DLN boundary prediction: when task demands
    are homogeneous within a single stage, conventional moderators that do
    not index processing architecture should not resolve heterogeneity, and
    DLN stage coding (which requires between-stage variation) cannot reduce
    it either.

    THEORETICAL IMPLICATION
    This dataset tests a boundary condition of the DLN framework.  The
    theory predicts moderation only when task demands or sample
    characteristics create between-stage variation (e.g., a meta-analysis
    that includes both standard and modified IGT variants, or both naive
    and experienced participants).  A meta-analysis restricted to a single
    task variant with a single population type provides no leverage for
    DLN stage coding.

    The significant heterogeneity (Q=206.001, tau2=20.782) that persists
    in this all-dot dataset likely reflects:
      (a) Variation in somatic marker sensitivity within dot-stage processing
      (b) Cultural and demographic factors affecting IGT strategy adoption
      (c) Methodological variation (lab conditions, instructions) that does
          not cross DLN stage boundaries
    These are within-stage sources of variability, not the between-stage
    architectural differences that DLN is designed to detect.

    QUANTITATIVE SUMMARY
      Overall effect:  UMD = {umd:.3f} (males > females)
      Heterogeneity:   Q(109) = {q:.3f}, p < 0.001
                       tau2 = {tau2:.3f}, 95% CI [{tau2_lo}, {tau2_hi}]
      DLN stage coding: Not applicable (no between-stage variation)
      DLN prediction:   Null (no moderation expected) -- CONFIRMED
    """).format(
        umd=umd, q=q, tau2=tau2,
        tau2_lo=rec["tau2_ci_lo"], tau2_hi=rec["tau2_ci_hi"],
    )
    print("\n" + interpretation)

    # ------------------------------------------------------------------
    # 6. Write outputs
    # ------------------------------------------------------------------
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_rows = [
        ["metric", "value"],
        ["meta_analysis", "Zanini, Picano, & Spitoni (2025)"],
        ["k_total", str(k_total)],
        ["task_variant", "Standard 100-trial IGT only"],
        ["sample", "Healthy participants only"],
        ["dln_stage_all", "dot"],
        ["k_dot", str(k_dot)],
        ["k_linear", str(k_linear)],
        ["k_network", str(k_network)],
        ["between_stage_variation", "none"],
        ["overall_umd", f"{umd:.3f}"],
        ["overall_se", rec["overall_se"]],
        ["q_stat", f"{q:.3f}"],
        ["q_df", str(q_df)],
        ["tau2", f"{tau2:.3f}"],
        ["moderators_tested", str(len(moderators))],
        ["any_moderator_significant", "no"],
        ["dln_prediction", "null (no between-stage variation)"],
        ["dln_prediction_confirmed", "yes"],
    ]
    with open(OUT_TABLE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)
    print(f"Summary table written to {OUT_TABLE}")

    # Full report
    with open(OUT_REPORT, "w") as f:
        f.write(interpretation)
    print(f"Full report written to {OUT_REPORT}")


if __name__ == "__main__":
    main()
