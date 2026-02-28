# DLN Stage Coding Manual (Effect-Size Level)

## Purpose
This rubric operationalizes **DLN-dominant architecture** as a study-level or effect-size-level moderator that can be coded from method sections and task structure.

Coding target:
- **Dot**: element-wise, reactive responding; minimal integration across cues/time.
- **Linear**: sequential chains with bottlenecks; suppression/compartmentalization is plausible or required; rule-based reasoning.
- **Network**: explicitly relational integration; feedback/loops; context updating; multi-constraint evaluation.

## How to code
### Step 1: Identify the unit being pooled
- If the effect size comes from a single task condition: code that condition.
- If it pools across multiple tasks/conditions: split into multiple effect sizes if possible; otherwise code as **Mixed/Unclear** and document.

### Step 2: Code using observable task features
Use the decision questions below.

#### Decision questions (quick rubric)
1. Does the task demand integration across multiple cues/relations, including context updating or feedback?
   - Yes → likely **Network**
2. Is the task fundamentally sequential, rule-governed, or designed to suppress/ignore affective information?
   - Yes → likely **Linear**
3. Is the task stimulus-driven with minimal relational structure, relying on immediate reactions?
   - Yes → likely **Dot**

### Step 3: Apply override rules (when needed)
- **Network override**: if the task explicitly requires integrating tradeoffs among multiple interacting variables (including social dynamics) *and* participants can revise hypotheses with feedback, prefer Network even if there are sequential steps.
- **Linear override**: if the task encourages “objective” reasoning that explicitly excludes affect, or relies on suppression/compartmentalization, prefer Linear even if multiple variables are present.
- **Dot override**: if responses are dominated by immediate salience or affective reaction with minimal reflective integration, prefer Dot.

## Examples (illustrative)
- Heartbeat counting accuracy → Dot/Linear boundary depends on whether task uses reflective monitoring and calibration; document assumptions.
- IAT vs self-report discrepancies → often **Linear** (explicit) with implicit influence; but coding can be effect-specific (implicit measure as signal vs explicit as control).
- Reappraisal training effects → often **Network** because it recruits cognitive contextualization and flexible integration.
- Suppression instruction effects → **Linear** by design.

## Coding categories
Allowed values:
- `dot`
- `linear`
- `network`
- `mixed_unclear` (must include rationale)

## Relationship to manuscript differentiation subsection
The coding categories above should be interpreted in light of the DLN framework's formal computational definitions (manuscript Section 2.5):
- **Dot**: Empty belief graph, $O(1)$ memory, no revision graph.
- **Linear**: Null graph on $K$ independent nodes, $O(K)$ scaling, fixed-point revision graph. Cannot represent covariance between elements.
- **Network**: Bipartite factor DAG, $O(F)$ scaling, cyclic revision graph with structural learning cycle (test–update–expand–contract).

The DLN framework is differentiated from LEAS, constructionism, integrative complexity, and ACT by the specific computational mechanisms it proposes (see manuscript Section 5.2). When coding, the critical distinction is **representational topology** (how information propagates), not merely the richness of emotional vocabulary or the number of constructs differentiated.

## Coding a "Linear-Plus" (dangerous-middle) configuration
Some effects may involve tasks or populations that exhibit partial integration (factor-level structure linking affective and cognitive variables) without metacognitive monitoring (no structural revision capacity). In such cases:
- Code as `linear` with a note: "Linear-Plus: partial integration without metacognition."
- This configuration is predicted to produce *worse* outcomes than full compartmentalization under high affective load (manuscript Prediction 9, Section 3.4).
- Document the basis for inferring partial integration (e.g., relational knowledge present but no feedback revision).

## Disconfirmation criteria for coding validity
The following outcomes would indicate the coding rubric is not functioning as intended:
1. Inter-rater reliability (Cohen's kappa) < 0.60 on a calibration sample.
2. DLN stage coding does not reduce between-study $\tau^2$ beyond conventional moderators in any domain.
3. Stage codes are achievable only through the emotion measures being predicted (circularity).
4. Stage coding produces identical distributions across substantively different domains.

## Documentation requirements
Every coded effect must include:
- short rationale (1–3 sentences)
- key method quote/snippet paraphrase (no long quotations)
- coder_id
- date

