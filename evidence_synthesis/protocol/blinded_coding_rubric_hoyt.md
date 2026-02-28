# Coding Rubric — Health Outcome Domain Classification

## Your task

You will see a list of 8 health outcome domains from a meta-analysis of
emotional approach coping and health. For each domain, assign it to
**one** of three processing categories (A, B, or C) defined below.

Base your judgment on **what kind of processing the outcome measurement
primarily captures** — not on whether the outcome is "good" or "bad."

---

## Category definitions

### Category A — Reactive / somatic

The outcome domain primarily captures **direct bodily responses or
simple action-level behaviours** with minimal cognitive mediation.
Measurement instruments assess physiological states, somatic symptoms,
or concrete behaviours rather than subjective evaluations requiring
reflective integration.

*Key question:* Does this outcome primarily reflect bodily processes,
somatic states, or concrete behaviours rather than cognitively mediated
evaluations?

### Category B — Unidimensional / compartmentalised

The outcome domain primarily captures **single-dimension symptom
severity or distress** measured along isolated scales. Measurement
instruments assess one psychological dimension at a time (e.g., how
depressed, how anxious, how intrusive) without requiring respondents to
integrate across life domains or evaluate multiple interacting
dimensions.

*Key question:* Does this outcome primarily measure severity on a single
psychological dimension, without requiring the respondent to integrate
across multiple life domains or evaluate trade-offs?

### Category C — Integrative / relational

The outcome domain primarily captures **outcomes that require flexible
integration of emotional, cognitive, and/or social information**.
Measurement instruments assess constructs that inherently involve
relating emotional experience to broader context — meaning-making,
relational functioning, or multi-dimensional well-being.

*Key question:* Does this outcome require the respondent to integrate
emotional information with broader cognitive, social, or existential
context?

---

## Decision procedure

For each health domain:

1. Read the domain name and description.
2. Ask the three key questions above in order (A → B → C).
3. Assign the **first category whose key question you answer "Yes" to**.
4. If more than one seems to fit, choose the one that best captures the
   **primary** processing demand of the typical measurement instruments
   used for that domain.
5. Record your assignment and a brief rationale (1–2 sentences).
6. Rate your **confidence** in the assignment (high / medium / low):
   - **High**: The category clearly fits; no serious alternative.
   - **Medium**: The category fits but another was considered.
   - **Low**: Uncertain; the item could plausibly belong to a different category.

---

## The domains to code

You will receive a CSV file (`hoyt2024_blinded.csv`) with 8 rows.
Each row has:

- **health_domain**: Short label for the outcome domain
- **domain_desc**: Description of what the domain includes
- **k**: Number of studies in the meta-analysis (for context only;
  do not use this to inform your coding)
- **source**: Where the data comes from
- **estimate_status**: Whether values are verified from original tables

For each row, assign a category (A, B, or C) and write a rationale.

---

## Examples (not in the dataset — for training only)

| Health domain | Category | Rationale | Confidence |
|---------------|----------|-----------|------------|
| Blood pressure reactivity to stress | A | Direct physiological measurement; no cognitive mediation required from the participant | High |
| Trait anger severity (e.g., STAXI) | B | Single-dimension severity rating along one affective dimension | High |
| Perceived quality of life across physical, psychological, social, and environmental domains (e.g., WHOQOL-BREF) | C | Requires respondent to evaluate and integrate across multiple life domains | High |
