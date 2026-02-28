# Coding Rubric — Emotion-Regulation Strategy Classification

## Your task

You will see a list of 10 emotion-regulation strategy families. For each
strategy, assign it to **one** of three processing categories (A, B, or C)
defined below. Base your judgment on the strategy description and what
you know about how the strategy works, not on how effective it is.

---

## Category definitions

### Category A — Reactive / stimulus-driven

The strategy operates through **immediate, stimulus-driven responding**
with minimal reflective or relational processing. The person reacts to
the situation or stimulus directly, without sequential reasoning or
multi-dimensional integration.

*Key question:* Does the strategy primarily rely on direct reactions to
stimuli or situations, without requiring deliberate chains of reasoning?

### Category B — Sequential / rule-governed

The strategy involves **deliberate, sequential, or rule-based processing**.
The person follows a specific procedure, suppresses or redirects
attention, or applies a rule to manage the emotional response. Processing
is structured but does not require integrating information across
multiple interacting dimensions simultaneously.

*Key question:* Does the strategy require the person to follow a
procedure, apply a rule, or deliberately redirect/suppress a response?

### Category C — Integrative / relational

The strategy requires **flexible integration of information across
multiple dimensions**. The person must hold multiple perspectives,
re-interpret meaning, or relate emotional information to broader
context. Processing involves combining or transforming information
rather than following a fixed procedure.

*Key question:* Does the strategy require the person to integrate across
multiple perspectives, reframe meaning, or flexibly relate emotional
information to context?

---

## Decision procedure

For each strategy:

1. Read the strategy name and description.
2. Ask the three key questions above in order (A → B → C).
3. Assign the **first category whose key question you answer "Yes" to**.
4. If more than one seems to fit, choose the one that best captures the
   **primary** processing demand of the strategy as typically studied.
5. Record your assignment and a brief rationale (1–2 sentences).
6. Rate your **confidence** in the assignment (high / medium / low):
   - **High**: The category clearly fits; no serious alternative.
   - **Medium**: The category fits but another was considered.
   - **Low**: Uncertain; the item could plausibly belong to a different category.

---

## The strategies to code

You will receive a CSV file (`webb2012_blinded.csv`) with 10 rows.
Each row has:

- **strategy_family**: The broad family the strategy belongs to
  (e.g., attentional_deployment, cognitive_change)
- **strategy_sub**: The specific strategy name
  (e.g., distraction, reappraisal, expressive_suppression)
- **k**: Number of studies in the meta-analysis (for context only;
  do not use this to inform your coding)

For each row, assign a category (A, B, or C) and write a rationale.

---

## Examples (not in the dataset — for training only)

| Strategy | Category | Rationale | Confidence |
|----------|----------|-----------|------------|
| Startle-blink reflex measurement | A | Immediate physiological response to a stimulus; no deliberate reasoning required | High |
| Counting backwards by sevens | B | Sequential, rule-governed procedure; requires maintaining a rule but not integrating multiple perspectives | High |
| Writing about how a stressful event changed one's life priorities | C | Requires integrating emotional experience with broader meaning; multiple perspectives and reframing involved | High |
