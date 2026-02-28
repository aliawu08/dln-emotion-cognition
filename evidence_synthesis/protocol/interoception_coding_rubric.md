# Interoception–Alexithymia DLN Stage Coding Rubric

## Sources
- Trevisan, D. A., et al. (2019). A meta-analysis on the relationship between
  interoceptive awareness and alexithymia: Distinguishing interoceptive accuracy
  and sensibility. *Journal of Abnormal Psychology*, 128(8), 765–776.
  doi:10.1037/abn0000454
- Van Bael, K., et al. (2024). A systematic review and meta-analysis of the
  relationship between subjective interoception and alexithymia: Implications
  for construct definitions and measurement. *PLOS ONE*, 19(11), e0310411.
  doi:10.1371/journal.pone.0310411

## Unit of Analysis
**Interoceptive measure-family level**. Each measure instrument used to assess
interoceptive awareness is coded for DLN stage based on the representational
topology required to process body signals as the measure operationalizes them.

## Coding Target
What is being coded: the **representational complexity of the body–cognition
coupling** that each measure captures.

The DLN framework specifies three processing topologies (Wu, 2026):
- **Dot**: Empty belief graph; O(1) memory; reactive body signal detection
  without persistent tracking or interpretation
- **Linear**: Null graph on K independent nodes; O(K) memory; body signals
  tracked along a single dimension without cross-element integration into
  emotion concepts
- **Network**: Bipartite factor DAG; O(F) memory; multi-dimensional integration
  linking body signals to emotion concepts, regulatory strategies, and
  metacognitive calibration

The critical distinction is **whether and how body signals propagate to
emotion-cognition representations**.

## Decision Tree

### Step 1: Does the measure capture multi-dimensional body–emotion integration?
Does the construct require:
- Linking body signals to emotion concepts (emotional awareness)?
- Metacognitive calibration between perceived and actual interoception?
- Using body signals for self-regulation across domains?
- Multi-scale awareness (noticing + interpreting + trusting + regulating)?

- **Yes → Network**
- No → Step 2

### Step 2: Does the measure capture single-dimension body signal tracking?
Does the construct involve:
- Tracking autonomic reactivity on an intensity dimension without interpreting it?
- Reporting body sensations as symptoms (anxiety-driven) rather than information?
- Monitoring a single channel without cross-referencing emotion or cognition?
- Confusion or distress about body signals (signals present but not integrated)?

- **Yes → Linear**
- No → Step 3

### Step 3: Does the measure capture raw signal detection?
Is the measured construct:
- Simple heartbeat counting/detection (stimulus-driven accuracy)?
- Basic body awareness (noticing without tracking or interpreting)?
- Passive registration of somatic events without dimensional structure?

- **Yes → Dot**
- If none of the above → **Mixed/Unclear**

## Pre-specified Measure-Family Coding

| Measure Family | DLN Stage | Rationale |
|---|---|---|
| **Heartbeat tasks (HCT/HPT)** | Dot | Objective signal detection: count/detect heartbeats. Raw stimulus-driven accuracy task with no cognitive mediation, emotional interpretation, or persistent belief structure. O(1) processing: detect-or-not per trial. |
| **BPQ-BA (Body Awareness)** | Dot | Reports basic awareness of body sensations (e.g., "I notice when my muscles are tense"). Simple noticing without interpretation, regulation, or emotion-linking. Consistently shows near-zero association with alexithymia (r ≈ -0.05, ns), indicating it captures a construct independent of emotion-cognition integration. |
| **ICQ (Interoceptive Confusion)** | Linear | Measures confusion and distress about body signals: signals are registered (tracked) but cannot be interpreted or integrated with emotion concepts. Classic linear compartmentalization: the body signal node is active but has no edges to emotion nodes. Positive association with alexithymia (r ≈ 0.57) reflects that more confusion (failed integration) tracks with more alexithymia. |
| **BPQ-R (Reactivity)** | Linear | Autonomic nervous system reactivity subscales: reports intensity of somatic responses (sweating, heart racing, stomach churning). Tracks body signals on a single intensity dimension without linking to emotion concepts. Mehling (2016) characterized BPQ-R items as anxiety-driven hypervigilance. Positive alexithymia association (r ≈ 0.36) indicates amplified body monitoring without emotional interpretation. |
| **EDI-IAw (Interoceptive Awareness)** | Linear | Eating Disorder Inventory interoceptive awareness: captures anxiety about and confusion over body signals in clinical eating disorder context. Single-dimension distress monitoring. Positive alexithymia association (r ≈ 0.26). |
| **IAS (Interoceptive Accuracy Scale)** | Network | Self-report interoceptive accuracy requiring metacognitive calibration: comparing perceived accuracy with actual body signals across domains. Multi-dimensional (accuracy across cardiovascular, gastrointestinal, respiratory systems). Negative alexithymia association (r ≈ -0.30) aligns with integrative measures. |
| **MAIA-Total (Multidimensional Assessment)** | Network | Explicitly multi-dimensional: 8 subscales spanning Noticing, Not-Distracting, Not-Worrying, Attention Regulation, Emotional Awareness, Self-Regulation, Body Listening, and Trusting. Core network-stage construct: integrates body signals with emotion concepts, regulatory strategies, and metacognitive awareness. Strongest negative alexithymia association (r ≈ -0.41). |
| **BAQ (Body Awareness Questionnaire)** | Network | Integrates body awareness with daily functioning and health behaviors across situations. Multi-contextual body-cognition coupling. Negative alexithymia association (r ≈ -0.17). |

## DLN Predictions for Interoception–Alexithymia

### Primary prediction: DLN stage moderates the direction and magnitude of the interoception–alexithymia association

This analysis generates a unique signature: the interoception–alexithymia association
**reverses sign** across DLN stages.

- **Dot measures** (heartbeat tasks, BPQ-BA): Near-zero association. Raw body signal
  detection operates independently of the emotion-cognition coupling that alexithymia
  disrupts. Dot-stage processing neither requires nor reflects emotion integration,
  so it should be unrelated to alexithymia.

- **Linear measures** (ICQ, BPQ-R, EDI-IAw): **Positive** association with
  alexithymia. Body signals are tracked on a single dimension (intensity, distress)
  but compartmentalized from emotion concepts. Higher alexithymia is associated with
  MORE of this type of "awareness" — because linear architecture amplifies body
  signals as noise/anxiety/confusion without routing them through emotion integration.
  This is the somatic amplification pattern: body awareness without meaning.

- **Network measures** (MAIA, IAS, BAQ): **Negative** association with alexithymia.
  Multi-dimensional body–emotion integration is exactly what alexithymia disrupts.
  Higher alexithymia predicts LESS integrative awareness because the network-stage
  processing required for body–emotion coupling is impaired.

### Theoretical significance
This sign reversal is a strong test of DLN because:
1. No conventional moderator (clinical status, sample size, measurement error)
   predicts a sign change between measure types
2. The reversal follows directly from the topology of body–cognition coupling:
   isolated tracking (linear) amplifies signals; integrated processing (network)
   contextualizes them
3. It explains Trevisan et al.'s (2019) puzzling finding that "interoceptive
   sensibility" showed no overall association with alexithymia — because sensibility
   measures pull in opposite directions depending on their DLN stage

## Sensitivity Analyses
1. Reclassify IAS as linear (boundary case: self-report vs metacognitive) → rerun
2. Drop measures with k < 4 (BAQ, IATS) → rerun
3. Use only Van Bael 2024 data (drop Trevisan heartbeat estimate) → rerun
4. Report all alongside primary analysis
