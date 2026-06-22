# TakeMeter — Planning

A fine-tuned text classifier that evaluates **discourse quality** in the r/anime
community by sorting comments along a "reasons → asserts → feels" spectrum:
`analysis` / `hot_take` / `reaction`.

This document is the spec. It is written **before** any example is collected or
labeled, and addresses the six required questions (Community, Labels, Hard edge
cases, Data collection, Evaluation metrics, Definition of success) plus an
AI Tool Plan.

---

## 1. Community

**Choice: r/anime**, one of the largest anime discussion forums online.

**Why it's a good fit for a classification task.** The quality of discourse on
r/anime varies enormously. The same Episode Discussion thread contains, side by
side: raw emotional outbursts, confident unsupported verdicts, and genuinely
structured arguments about direction, adaptation choices, and theme. That spread
is exactly what makes the task interesting — the labels aren't about *topic*
but about *form*, so the model has to learn something subtle rather than 
keyword-match a subject. The community also *already* makes this distinction 
implicitly: it upvotes effortful comments, flairs "Rewatch" and discussion posts, 
and mods enforce effort requirements in some threads. We're formalizing a judgment 
regulars already make in the community

---

## 2. Labels

Three labels forming a single **quality spectrum** (reasons → asserts → feels).
The decision boundary is *the form of the claim*, not its sentiment or subject.

### `analysis` — *it reasons*
> Makes a structured argument about an anime's writing, themes, direction,
> pacing, or animation, backed by **specific, verifiable** evidence (a named
> scene, a directing/animation choice, source-material context, a production
> fact).

- "Episode 12's reveal works because the foreshadowing was planted in episode 3
  — the recurring broken-clock shot tracked exactly when each character stopped
  moving forward. The payoff lands because the visual grammar stayed consistent
  across the cour."
- "People calling the pacing bad miss that this faithfully adapts the manga's
  third arc, which deliberately slows down. The director even cut two action
  beats from the source to give the funeral scene room to breathe."

### `hot_take` — *it asserts*
> A bold, confident **opinion or evaluative claim** stated **without** real
> supporting evidence — it might be correct, but it asserts rather than argues.

- "This is the best-written anime of the last ten years and it's not even close."
- "Isekai is creatively bankrupt — every single one is the same power-fantasy
  garbage."

### `reaction` — *it feels*
> An immediate **emotional response** to a specific moment or episode, with
> little or no argument — expressing a feeling in the moment.

- "I AM NOT OKAY AFTER THAT EPISODE 😭😭 they did NOT have to do him like that"
- "okay that fight was actually insane, my jaw is on the floor, peak"

**Why two annotators would mostly agree:** each label is keyed to an observable
surface feature — *is there verifiable evidence?* (analysis), *is it a flat
evaluative claim?* (hot_take), *is it an in-the-moment feeling?* (reaction) —
rather than a subjective "is this good?" judgment.

---

## 3. Hard edge cases & decision rules

A comment is assigned to **exactly one** label via this ordered procedure:

1. Presents specific, verifiable evidence supporting a claim? → **analysis**
2. Else, primarily a flat evaluative claim about the work/genre/creator? → **hot_take**
3. Else, primarily an emotional response to a moment/episode? → **reaction**

### Edge case A — `analysis` vs `hot_take` (the "decorative evidence" post) — *primary*
> "Season 2 is objectively worse — the studio changed and you can tell, the
> budget clearly got slashed."

Cites a verifiable fact (studio change) but the conclusion is unsupported
assertion; the fact is *decorative*.
**Rule:** keep `analysis` only if the evidence survives stripping the opinion
framing — i.e. it genuinely does the reasoning. If it's vague, cherry-picked, or
decorative, it's `hot_take`. → **hot_take.**

### Edge case B — `reaction` vs `hot_take` (the short verdict) — *secondary*
> "this episode was garbage" vs "this episode was a masterpiece"

**Rule:** in-the-moment feeling about a *specific event* → `reaction`;
*generalizing* evaluative claim about the work/genre/creator → `hot_take`.
**Tiebreaker for bare superlatives** (surfaced during stress-testing, §7): a
one-line verdict on a specific episode with emotional markers (caps, emoji,
"I", exclamation, slang like "peak") → `reaction`; a flat, declarative,
generalizing verdict → `hot_take`. So "this episode was a masterpiece" (flat,
specific episode) leans `reaction`; "this show is a masterpiece of the genre"
(generalizing) → `hot_take`.

### Edge case C — `reaction` vs `analysis` (feeling with a detail)
> "I cried when he died — the way they cut to the empty chair afterward got me."

Mentions a directing choice but is **not arguing anything** — the detail serves
the feeling. **Rule:** if the comment's purpose is to convey emotion and the
detail is illustrative rather than evidentiary, it's `reaction`. → **reaction.**

**Catch-all policy:** comments that are pure questions, off-topic, single emoji,
or memes with no take are **excluded at collection time** (not labeled), so the
3 labels stay ≥90% exhaustive over what remains.

### Three real hard cases encountered during annotation

These are actual r/anime comments from the dataset that genuinely sat between two
labels, with the call made:

1. **`reaction` vs `analysis` —** *"The final part of the episode did bring a tear
   to my eye… Season 1 has been pretty solid. The pacing felt rushed and they cut
   some small bits out, but I suppose that's part of setting up the Zenza arc.
   Hopefully future seasons have more breathing room."* — Mixes a real adaptation
   observation (cut content, pacing serving arc setup) with strong emotion ("tear
   to my eye", "easily my favourite part"). **Decided `reaction`:** the purpose is
   to express how the episode *felt*; the pacing note is a passing aside, not a
   built-out argument (Edge case C).

2. **`hot_take` vs `analysis` —** *"I'm not a huge fan of how the MC is written. He
   has little to no agency… the daemons don't immediately outclass him, that stunt
   pulls me out."* — Long and structured, reads analytical, but every claim is
   unsupported personal taste ("I find his character bland"). **Decided
   `hot_take`:** strip the structured framing and no verifiable evidence remains
   (Edge case A — decorative structure, not real reasoning). *(The model later
   miscalled this exact comment `analysis` at 0.97 confidence — see README.)*

3. **`hot_take` vs `reaction` —** *"25 minutes of straight fight animation, we
   finally have some budget."* — A sarcastic claim about the production (budget),
   but phrased as a short, casual quip. **Decided `hot_take`:** it asserts an
   evaluative claim about the show's production rather than just emoting; contrast
   a pure reaction like "that fight went so hard" (Edge case B / the sarcasm makes
   it borderline, and short sarcastic production-claims are a known weak spot).

---

## 4. Data collection plan

- **Source:** r/anime **Episode Discussion threads** and **discussion/review
  posts** (NOT recommendation-request threads). Collected via PRAW (Reddit's
  official API, free script app) or the public `.json` endpoints as fallback.
- **Volume target:** ≥ **240** labeled comments (buffer above the 200 minimum to
  survive exclusions), split **70/15/15** train/val/test, **stratified** by label.
- **Per-label target:** ≥ **20% per class**. Natural r/anime distribution skews
  hard toward `reaction`, so `analysis` is the bottleneck.
- **If a label is underrepresented after the first pass:** `analysis` is the
  expected shortfall. Mitigation, in order: (1) **oversample** from review /
  "Rewatch" / theory-discussion posts where structured argument concentrates;
  (2) pull from highly-upvoted top-level comments (effort correlates with
  upvotes); (3) widen the time window across more episode threads. If still
  short, document the final distribution honestly and report **macro-F1** (which
  doesn't reward majority-class bias) rather than papering over imbalance.

---

## 5. Evaluation metrics

Accuracy alone is insufficient because the classes are **imbalanced** (reaction
dominates) — a model that always predicts `reaction` could score deceptively
high on accuracy while being useless. We report:

| Metric | Why it's the right one for *this* task |
|---|---|
| **Overall accuracy** | Required baseline number; comparable across both models. |
| **Macro-F1** (primary headline) | Averages F1 across classes equally, so it *penalizes* ignoring the rare `analysis` class. This is the number I'll judge success on. |
| **Per-class precision / recall / F1** | The spectrum's value is in the rare-but-important end. |
| **`analysis` recall specifically** | A "surface the good takes" tool fails if it misses real analysis — recall on `analysis` is the cost-sensitive metric I care most about. |
| **`analysis` precision** | If the tool surfaces takes to humans, surfaced items should mostly *be* analysis, or it wastes attention. |
| **3×3 confusion matrix** | Reveals *which* boundary the model confuses — I expect `analysis`↔`hot_take` (the decorative-evidence boundary) and `reaction`↔`hot_take` (short verdicts). The structure of errors validates or refutes my edge-case design. |

Both the fine-tuned DistilBERT and the zero-shot Groq baseline are scored on the
**same held-out test set** with these same metrics.

---

## 6. Definition of success

Concrete, checkable thresholds on the test set:

- **Useful as an assistive filter (deployment bar):** macro-F1 **≥ 0.70**,
  **AND** `analysis` recall **≥ 0.60** (don't miss most good takes), **AND**
  `analysis` precision **≥ 0.55** (surfaced takes are more often right than not).
- **Fine-tuning actually helped (project bar):** fine-tuned macro-F1 beats the
  zero-shot Groq baseline by **≥ 10 points** of macro-F1 on the same test set.
- **Sanity ceiling:** if test accuracy is **> 0.95** on this subjective task, I
  treat it as a red flag for train/test leakage or labels that are too easy, and
  investigate before reporting.

**Deployment interpretation:** the realistic role is an **assistive filter** —
surfacing candidate `analysis` comments for a human (or for "best of thread"
highlights), *not* autonomous moderation. At macro-F1 ≈ 0.70 with the recall/
precision floors above, it's good enough to reduce a human's reading load while
keeping them in the loop; it is **not** good enough to auto-remove or auto-rank
content without review, and I'll say so explicitly in the README.

---

## 7. AI Tool Plan

There's no application code to generate here, so AI tools help in three specific
places.

### a. Label stress-testing — *done, before annotation*
I gave the label definitions + edge cases to the AI and asked it to generate
boundary posts and classify them. Result: 2 of 3 classified cleanly; one — a
**bare superlative verdict** ("this episode was a masterpiece") — exposed that
my `reaction`/`hot_take` rule didn't cover flat one-line verdicts. **Action
taken:** added the bare-superlative tiebreaker to Edge case B (§3) *before*
annotating. This is logged in the README's AI-usage section.

### b. Annotation assistance — *decision: yes, with disclosure*
I will use an LLM (Groq `llama-3.3-70b-versatile`, the same model as the
baseline) to **pre-label** batches; I then review and correct every pre-label
myself — the human label is authoritative. **Leakage guard:** because the
baseline is also Groq zero-shot, pre-labels are *only* used to speed human
review, never copied as gold without my confirmation, and the **test set is
human-labeled from scratch with no pre-labels** to keep the baseline comparison
honest. A `prelabeled` boolean column in the CSV tracks which rows were
machine-suggested, for disclosure.

### c. Failure analysis — *planned*
After evaluation I'll hand the AI the list of wrong predictions (text + true
label + predicted label + confidence) and ask it to propose **systematic error
patterns** (e.g. "misses analysis when the comment is short," "confuses
sarcastic hot_takes for reactions"). **Verification:** I won't take a claimed
pattern at face value — for each proposed pattern I'll pull the supporting
examples and check the pattern holds on held-out errors, reporting only patterns
I can confirm by hand.

---

## Roadmap (later milestones)

- **M3 — Fine-tune:** `distilbert-base-uncased` on Colab T4. Document base model,
  training approach, ≥1 hyperparameter decision.
- **M4 — Baseline:** zero-shot Groq `llama-3.3-70b-versatile` on the same test set.
- **M5 — Eval report:** the §5 metrics, ≥3 analyzed errors, learned-vs-intended
  reflection.

### Stretch (update this doc before starting each)
- Inter-annotator reliability (Cohen's κ on 30+ shared examples)
- Confidence calibration
- Systematic error-pattern analysis (builds on §7c)
- Deployed interface (input → label + confidence)
