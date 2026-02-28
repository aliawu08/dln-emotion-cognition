"""AI-assisted blind coding validation for DLN stage assignments.

For each primary dataset, this script:
  1. Strips effect sizes and DLN codes from extraction data
  2. Constructs a blind coding prompt using only the rubric decision tree
     and item descriptions (no effect sizes, no expected patterns)
  3. Runs N independent AI coding passes
  4. Computes agreement between AI modal coding and author coding
     (Cohen's kappa, percent agreement, Fleiss' kappa across passes)

Datasets validated: Webb (k=10), Interoception (k=8), Desmedt (k=7),
Hoyt (k=8), Greenwald (9 topic domains).

Usage
-----
  # Dry-run: shows prompts, uses author codes as stand-in
  python blind_coding_validation.py --dry-run

  # Full run (requires ANTHROPIC_API_KEY environment variable)
  python blind_coding_validation.py --n-passes 10

Outputs
-------
- evidence_synthesis/outputs/tables/blind_coding_validation.csv
- Console summary with per-dataset kappa values
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_TABLE = (
    ROOT / "evidence_synthesis" / "outputs" / "tables"
    / "blind_coding_validation.csv"
)

# ── Generic DLN decision tree (shared across datasets) ──────────
GENERIC_DECISION_TREE = """\
Apply the following three-question decision tree to each item:

1. Does the task, measure, or domain require integration across multiple
   cues with context updating, feedback, meaning-making, or
   multi-dimensional appraisal?
   → If YES: code as **network**

2. Is the task, measure, or domain fundamentally sequential,
   rule-governed, single-dimensional, or designed to suppress/ignore
   affective information?
   → If YES: code as **linear**

3. Is the task, measure, or domain stimulus-driven with minimal
   relational structure, involving reactive responses without
   cognitive mediation?
   → If YES: code as **dot**

If none of the above clearly applies: code as **mixed_unclear**
"""


# ═══════════════════════════════════════════════════════════════════
# DATASET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

def _load_webb():
    """Webb: blind item descriptions and author codes."""
    csv = ROOT / "evidence_synthesis" / "extraction" / "webb2012_strategy_extraction.csv"
    df = pd.read_csv(csv)
    items = []
    for _, row in df.iterrows():
        items.append({
            "item_id": row["strategy_sub"],
            "description": (
                f"Emotion regulation strategy: {row['strategy_sub'].replace('_', ' ')}. "
                f"Family: {row['strategy_family'].replace('_', ' ')}."
            ),
            "author_code": row["dln_stage_code"],
        })
    rubric_context = (
        "Domain: Emotion regulation strategies from Webb et al. (2012).\n"
        "Each item is a strategy sub-family used in experimental emotion "
        "regulation studies. Code each strategy based on the cognitive "
        "processing demands it places on the participant:\n"
        "- Strategies requiring only reactive, stimulus-driven responding → dot\n"
        "- Strategies requiring rule-governed, sequential control or active "
        "suppression of affect → linear\n"
        "- Strategies requiring integrative reframing, multi-perspective "
        "understanding, or meaning-level engagement with emotion → network\n"
    )
    return "Webb", items, rubric_context


def _load_interoception():
    """Interoception: blind item descriptions and author codes."""
    csv = ROOT / "evidence_synthesis" / "extraction" / "interoception_measure_extraction.csv"
    df = pd.read_csv(csv)
    items = []
    for _, row in df.iterrows():
        items.append({
            "item_id": row["measure_family"],
            "description": (
                f"Interoceptive measure: {row['measure_family']}. "
                f"Description: {row['measure_desc']}."
            ),
            "author_code": row["dln_stage_code"],
        })
    rubric_context = (
        "Domain: Interoceptive awareness measures.\n"
        "Each item is a measure or task used to assess interoceptive "
        "processing. Code each measure based on the representational "
        "complexity of body-cognition coupling it captures:\n"
        "- Raw signal detection (heartbeat counting, basic body noticing) → dot\n"
        "- Single-dimension body signal tracking (reactivity intensity, "
        "confusion, distress monitoring) without emotion integration → linear\n"
        "- Multi-dimensional body-emotion integration (metacognitive "
        "calibration, cross-domain awareness, emotional awareness, "
        "self-regulation) → network\n"
    )
    return "Interoception", items, rubric_context


def _load_desmedt():
    """Desmedt: blind item descriptions and author codes."""
    csv = ROOT / "evidence_synthesis" / "extraction" / "desmedt2022_criterion_extraction.csv"
    df = pd.read_csv(csv)
    items = []
    for _, row in df.iterrows():
        items.append({
            "item_id": row["criterion"],
            "description": (
                f"HCT criterion: {row['criterion']}. "
                f"Description: {row['criterion_desc']}."
            ),
            "author_code": row["dln_stage_code"],
        })
    rubric_context = (
        "Domain: Criterion measures for heartbeat counting task (HCT) "
        "validity from Desmedt et al. (2022).\n"
        "The HCT is a simple cardiac signal detection task. Each item is "
        "a criterion variable correlated with HCT performance. Code each "
        "criterion based on whether it is:\n"
        "- A biological/physiological variable (direct somatic measurement "
        "without cognitive mediation) → dot\n"
        "- A psychological self-report measure (single-dimension symptom "
        "severity or cognitive-affective self-assessment) → linear\n"
        "- A multi-dimensional integrative construct → network\n"
    )
    return "Desmedt", items, rubric_context


def _load_hoyt():
    """Hoyt: blind item descriptions and author codes."""
    csv = ROOT / "evidence_synthesis" / "extraction" / "hoyt2024_domain_extraction.csv"
    df = pd.read_csv(csv)
    items = []
    for _, row in df.iterrows():
        items.append({
            "item_id": row["health_domain"],
            "description": (
                f"Health outcome domain: {row['health_domain'].replace('_', ' ')}. "
                f"Description: {row['domain_desc']}."
            ),
            "author_code": row["dln_stage_code"],
        })
    rubric_context = (
        "Domain: Health outcome domains from Hoyt et al. (2024) "
        "meta-analysis of emotional approach coping.\n"
        "Each item is a health outcome domain. Code each domain based on "
        "the structural features of its measurement instruments:\n"
        "- Single-modality somatic outcomes or simple action-level "
        "behaviours → dot\n"
        "- Unidimensional distress severity measures (single-valence "
        "symptom scales without relational or contextual components) → linear\n"
        "- Multi-domain integrative appraisals (measures requiring "
        "respondents to integrate across life domains) → network\n"
    )
    return "Hoyt", items, rubric_context


def _load_greenwald():
    """Greenwald: topic-level coding (9 unique topics)."""
    csv = ROOT / "evidence_synthesis" / "extraction" / "greenwald2009_study_extraction.csv"
    df = pd.read_csv(csv)
    # DLN is coded at topic level
    topic_to_dln = {
        "Consumer": "dot",
        "Race (Bl/Wh)": "linear",
        "Politics": "linear",
        "Gender/sex": "linear",
        "Other intergroup": "linear",
        "Relationships": "network",
        "Personality": "mixed_unclear",
        "Drugs/tobacco": "mixed_unclear",
        "Clinical": "mixed_unclear",
    }
    topic_descs = {
        "Consumer": "Consumer preferences: brand choices, product attitudes, purchasing",
        "Race (Bl/Wh)": "Racial attitudes: Black-White intergroup attitudes and discrimination",
        "Politics": "Political attitudes: candidate preferences, policy support, political identity",
        "Gender/sex": "Gender/sex attitudes: gender stereotypes, sexism, gender role attitudes",
        "Other intergroup": "Other intergroup attitudes: age, weight, nationality, religion",
        "Relationships": "Close relationships: partner preferences, attachment, relationship quality",
        "Personality": "Personality traits: self-esteem, anxiety, extraversion, agreeableness",
        "Drugs/tobacco": "Substance use: drug and tobacco attitudes, consumption behaviour",
        "Clinical": "Clinical: phobias, psychopathology symptoms, clinical outcomes",
    }
    items = []
    for topic in sorted(topic_to_dln.keys()):
        items.append({
            "item_id": topic,
            "description": f"IAT criterion domain: {topic}. {topic_descs[topic]}.",
            "author_code": topic_to_dln[topic],
        })
    rubric_context = (
        "Domain: IAT (Implicit Association Test) criterion domains from "
        "Greenwald et al. (2009).\n"
        "Each item is a topic domain in which IAT predictive validity "
        "was assessed. Code each domain based on the cognitive processing "
        "demands of the criterion behaviour:\n"
        "- Reflexive/automatic criterion behaviour (snap judgments, "
        "approach-avoidance) → dot\n"
        "- Deliberative criterion behaviour under social-desirability "
        "pressure (intergroup, political — where explicit and implicit "
        "may diverge due to suppression) → linear\n"
        "- Well-practised integrative criterion behaviour (sustained "
        "relational contexts, intimate relationships) → network\n"
        "- Ambiguous or spanning multiple processing levels → mixed_unclear\n"
    )
    return "Greenwald", items, rubric_context


DATASET_LOADERS = [
    _load_webb,
    _load_interoception,
    _load_desmedt,
    _load_hoyt,
    _load_greenwald,
]


# ═══════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

def build_blind_prompt(rubric_context: str, items: list[dict]) -> str:
    """Build a prompt for blind DLN coding with NO effect sizes."""
    item_block = "\n".join(
        f"  {i+1}. [{item['item_id']}] {item['description']}"
        for i, item in enumerate(items)
    )
    return (
        "You are a research assistant coding items for a meta-analytic "
        "study. You will classify each item into one of three cognitive "
        "processing stages (dot, linear, network) or mixed_unclear.\n\n"
        f"## General Decision Tree\n\n{GENERIC_DECISION_TREE}\n"
        f"## Domain-Specific Guidance\n\n{rubric_context}\n"
        f"## Items to Code\n\n{item_block}\n\n"
        "## Instructions\n\n"
        "For each item, provide your coding in this exact format:\n"
        "  ITEM_ID: STAGE (1-2 sentence rationale)\n\n"
        "Where STAGE is one of: dot, linear, network, mixed_unclear\n"
        "Code each item based ONLY on the structural/processing features "
        "described above. Do not consider what result you expect."
    )


# ═══════════════════════════════════════════════════════════════════
# RESPONSE PARSING
# ═══════════════════════════════════════════════════════════════════

VALID_STAGES = {"dot", "linear", "network", "mixed_unclear"}


def parse_ai_response(response_text: str, item_ids: list[str]) -> dict[str, str]:
    """Parse AI coding response into {item_id: stage} mapping."""
    codes = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Try pattern: ITEM_ID: STAGE (rationale)
        # or: N. ITEM_ID: STAGE ...
        for item_id in item_ids:
            # Escape special chars for regex
            escaped = re.escape(item_id)
            pattern = rf"(?:^|\d+\.\s*)\[?{escaped}\]?\s*[:=\-]\s*(dot|linear|network|mixed_unclear)"
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                codes[item_id] = match.group(1).lower()
                break
    return codes


# ═══════════════════════════════════════════════════════════════════
# AGREEMENT METRICS
# ═══════════════════════════════════════════════════════════════════

def cohens_kappa(codes_a: list[str], codes_b: list[str]) -> float:
    """Compute Cohen's kappa for two lists of categorical codes."""
    assert len(codes_a) == len(codes_b)
    n = len(codes_a)
    if n == 0:
        return 0.0

    all_labels = sorted(set(codes_a) | set(codes_b))
    label_idx = {l: i for i, l in enumerate(all_labels)}
    m = len(all_labels)

    # Confusion matrix
    cm = np.zeros((m, m), dtype=int)
    for a, b in zip(codes_a, codes_b):
        cm[label_idx[a], label_idx[b]] += 1

    p_o = np.trace(cm) / n
    p_e = sum(cm[i, :].sum() * cm[:, i].sum() for i in range(m)) / (n * n)

    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa for multiple raters.

    ratings_matrix: (n_items, n_categories) — count of raters assigning
    each category to each item.
    """
    n_items, n_cats = ratings_matrix.shape
    n_raters = ratings_matrix[0].sum()
    if n_raters <= 1:
        return 0.0

    # Proportion per category
    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)

    # Per-item agreement
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (
        n_raters * (n_raters - 1)
    )
    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)

    if P_e >= 1.0:
        return 1.0 if P_bar == 1.0 else 0.0
    return (P_bar - P_e) / (1.0 - P_e)


# ═══════════════════════════════════════════════════════════════════
# AI API CALL
# ═══════════════════════════════════════════════════════════════════

def call_ai_coder(prompt: str, temperature: float = 0.7) -> str:
    """Call Anthropic API for one blind coding pass."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required. Install with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ═══════════════════════════════════════════════════════════════════
# DRY RUN: use author codes as stand-in
# ═══════════════════════════════════════════════════════════════════

def dry_run_coding(items: list[dict]) -> dict[str, str]:
    """Return author codes as a stand-in for AI coding (dry run)."""
    return {item["item_id"]: item["author_code"] for item in items}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_validation(n_passes: int = 10, dry_run: bool = False):
    """Run the full blind coding validation pipeline."""
    all_rows = []

    print("=" * 80)
    print("AI-ASSISTED BLIND CODING VALIDATION")
    print(f"Mode: {'DRY RUN (author codes as stand-in)' if dry_run else f'LIVE ({n_passes} passes)'}")
    print("=" * 80)

    for loader in DATASET_LOADERS:
        dataset_name, items, rubric_context = loader()
        item_ids = [item["item_id"] for item in items]
        author_codes = [item["author_code"] for item in items]
        k = len(items)

        print(f"\n--- {dataset_name} (k={k}) ---")

        prompt = build_blind_prompt(rubric_context, items)
        if dry_run:
            print(f"  Prompt length: {len(prompt)} chars")
            print(f"  First 200 chars: {prompt[:200]}...")

        # Run N passes
        all_pass_codes = []
        for pass_i in range(n_passes):
            if dry_run:
                codes = dry_run_coding(items)
            else:
                temp = 0.3 + (pass_i / max(n_passes - 1, 1)) * 0.7
                response = call_ai_coder(prompt, temperature=temp)
                codes = parse_ai_response(response, item_ids)

            pass_codes = [codes.get(iid, "MISSING") for iid in item_ids]
            all_pass_codes.append(pass_codes)

            if not dry_run:
                match = sum(
                    1 for a, p in zip(author_codes, pass_codes)
                    if a == p
                )
                print(f"  Pass {pass_i+1}: {match}/{k} match "
                      f"({match/k*100:.0f}%)")

        # Modal AI coding
        modal_codes = []
        for j in range(k):
            votes = [all_pass_codes[p][j] for p in range(n_passes)]
            from collections import Counter
            ctr = Counter(votes)
            modal_codes.append(ctr.most_common(1)[0][0])

        # Agreement rates per item
        for j, item in enumerate(items):
            votes = [all_pass_codes[p][j] for p in range(n_passes)]
            agreement_rate = sum(
                1 for v in votes if v == modal_codes[j]
            ) / n_passes

            all_rows.append({
                "dataset": dataset_name,
                "item": item["item_id"],
                "author_code": item["author_code"],
                "ai_modal_code": modal_codes[j],
                "agree_author": item["author_code"] == modal_codes[j],
                "ai_consistency": round(agreement_rate, 2),
            })

        # Cohen's kappa: author vs AI modal
        kappa = cohens_kappa(author_codes, modal_codes)
        pct_agree = sum(
            1 for a, m in zip(author_codes, modal_codes) if a == m
        ) / k

        # Fleiss' kappa across passes
        all_labels = sorted(set(author_codes) | VALID_STAGES)
        label_idx = {l: i for i, l in enumerate(all_labels)}
        ratings = np.zeros((k, len(all_labels)), dtype=int)
        for p in range(n_passes):
            for j in range(k):
                code = all_pass_codes[p][j]
                if code in label_idx:
                    ratings[j, label_idx[code]] += 1
        fk = fleiss_kappa(ratings)

        # Disagreements
        disagree_items = [
            f"{items[j]['item_id']} (author={author_codes[j]}, "
            f"ai={modal_codes[j]})"
            for j in range(k) if author_codes[j] != modal_codes[j]
        ]

        print(f"  Cohen's kappa (author vs. AI modal): {kappa:.3f}")
        print(f"  Percent agreement: {pct_agree:.1%}")
        print(f"  Fleiss' kappa (across {n_passes} passes): {fk:.3f}")
        if disagree_items:
            print(f"  Disagreements: {'; '.join(disagree_items)}")
        else:
            print("  No disagreements")

    # Save results
    results_df = pd.DataFrame(all_rows)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_TABLE, index=False)
    print(f"\nWrote: {OUT_TABLE}")

    # Summary
    n_total = len(results_df)
    n_agree = results_df["agree_author"].sum()
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"  Total items: {n_total}")
    print(f"  Author-AI agreement: {n_agree}/{n_total} ({n_agree/n_total*100:.0f}%)")

    for ds in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == ds]
        n = len(sub)
        ag = sub["agree_author"].sum()
        kap = cohens_kappa(
            sub["author_code"].tolist(), sub["ai_modal_code"].tolist()
        )
        print(f"  {ds}: {ag}/{n} ({ag/n*100:.0f}%), kappa={kap:.3f}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="AI-assisted blind coding validation"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run: show prompts, use author codes as stand-in",
    )
    parser.add_argument(
        "--n-passes", type=int, default=10,
        help="Number of independent AI coding passes (default: 10)",
    )
    args = parser.parse_args()
    run_validation(n_passes=args.n_passes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
