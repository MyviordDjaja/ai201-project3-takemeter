# TakeMeter

A fine-tuned text classifier that scores **discourse quality** in r/anime by
sorting comments along a "reasons → asserts → feels" spectrum:
`analysis` / `hot_take` / `reaction`.

Full label design, edge-case rules, metrics rationale, and success criteria live
in [`planning.md`](planning.md). This README documents the **data, model,
evaluation, and honest limitations**.

---

## TL;DR results (held-out test set, n=48)

| Model | Accuracy | Macro-F1 | analysis F1 | hot_take F1 | reaction F1 |
|---|---|---|---|---|---|
| Fine-tuned (MiniLM, ours) | 0.625 | **0.626** | 0.67 | 0.56 | 0.65 |
| Groq `llama-3.3-70b` zero-shot | **0.750** | **0.711** | 0.60 | 0.69 | 0.84 |

**The zero-shot 70B baseline beat our fine-tuned model by ~8.5 macro-F1 points.**
---

## 1. The community and labels

**r/anime.** Discourse quality there varies along a recognizable axis — the same
Episode Discussion thread holds raw emotional outbursts, confident unsupported
verdicts, and evidence-backed arguments. The labels classify the *form* of a
take, not its topic:

- **`analysis`** — structured argument backed by specific, verifiable evidence (it reasons)
- **`hot_take`** — bold evaluative claim with no real support (it asserts)
- **`reaction`** — in-the-moment emotional response (it feels)

---

## 2. Data

### Where it came from
Public comments from **8 r/anime threads** — a deliberate mix so all three
classes were represented:
- 4 **Episode Discussion** threads (Wistoria S2, Agents of the Four Seasons,
  Mistress Kanan, Akane-banashi) — reaction/hot_take heavy
- 2 **Rewatch** threads (Lycoris Recoil, Familiar of Zero) — where `analysis` concentrates
- 2 **Discussion/opinion** threads ("I really like Daemons of the Shadow Realm",
  "the one anime you'll defend to the death")

### How it was collected (and why this way)
I had claude code create a parser to look through the raw json files that I sourced myself by looking through different threads in the r/anime. Using `.json?limit=500&raw_json=1`, I saved the raw data json files into `data/saved_threads/` and parsed locally with no network (`scripts/parse_saved_json.py`). The parser flattens the nested comment tree,
drops deleted/bot/quote-only/too-short comments, dedups, and caps 40 comments per
thread for topic diversity. **320 raw comments → 315 labeled.**

### Labeling process (read this — it has a real limitation)
1. **Pre-label:** Groq `llama-3.3-70b` suggested a label for each comment using
   the `planning.md` rules (`scripts/prelabel.py`).
2. **First human pass:** every comment reviewed in a CLI showing the suggestion
   (`scripts/review.py`). **Problem:** this pass agreed with Groq **99%** of the
   time — a rubber-stamp. On a subjective task that's a red flag, and it makes the
   gold labels nearly identical to a zero-shot LLM (which would invalidate the
   baseline comparison).
3. **Blind re-review:** to fix this, a second pass re-judged comments with the
   suggestion **hidden** (`scripts/review_blind.py`). On the `analysis` class
   (where the LLM was weakest), **40% of first-pass labels flipped** to
   `hot_take`/`reaction` — confirming the rubber-stamp.
4. **Limitation (disclosed honestly):** the blind re-review was **partial** —
   101 of a planned 157 rows (all 97 `analysis` + part of a 60-row sample of the
   rest) were re-judged before time ran out. So the `analysis` class is mostly
   cleaned; `hot_take`/`reaction` rows that weren't in the blind subset may still
   carry first-pass (anchored) labels. **The dataset is improved but imperfect,
   and some label noise remains.** This is the single biggest caveat on every
   result below.

### Label distribution (final, n=315)
| Label | Count | Share |
|---|---|---|
| reaction | 151 | 48% |
| hot_take | 89 | 28% |
| analysis | 75 | 24% |

No class exceeds 70%; every class clears the 20% floor. `train_eval.py` does a
stratified **70/15/15** split → train 220 / val 47 / test 48.

### Three genuinely hard-to-label examples
Documented with decisions in [`planning.md §3`](planning.md) — real cases pulled
from the data (e.g. the *"brought a tear to my eye… the pacing felt rushed to set
up the Zenza arc"* comment that mixes emotion with an adaptation observation).

---

## 3. Model & training

- **Started from:** `sentence-transformers/all-MiniLM-L6-v2` — a 6-layer,
  ~22M-param BERT-family encoder. *(The plan was `distilbert-base-uncased`, but
  this environment throttled HuggingFace downloads to ~1 MB/s and the 268 MB
  DistilBERT download kept failing across WSL restarts. MiniLM was already cached
  locally, is the same architecture family, trains faster on CPU, and the brief
  permits "another pre-trained model of your choice." Swap `MODEL_NAME` in
  `scripts/train_eval.py` to use DistilBERT if bandwidth allows.)*
- **Setup:** fine-tuned a fresh 3-way classification head on CPU (no GPU needed
  for 220 examples), stratified split, max sequence length 128 (covers ~all
  comments; median length ~137 chars).

### Key hyperparameter decision: class-weighted loss + more epochs
The **first** training run (4 epochs, lr 2e-5, unweighted loss) **collapsed to
predicting `reaction` for everything** — `analysis` and `hot_take` both scored
F1 = 0.00 (accuracy 0.48, macro-F1 0.22). Two compounding causes: too few epochs
(loss barely moved) and the 48% `reaction` majority making collapse the lazy
optimum.

**Fix (the decision):** (1) **inverse-frequency class weights** in the loss
(`analysis` ×1.41, `hot_take` ×1.18, `reaction` ×0.69) to penalize majority
collapse, and (2) **12 epochs at lr 3e-5**. Result: loss fell 1.10 → 0.06, the
collapse disappeared, and macro-F1 went **0.22 → 0.63**. (The 0.06 final train
loss also signals overfitting on 220 examples — expected, and part of why the
model doesn't beat the baseline.)

---

## 4. Evaluation

Both models scored on the **same 48-example test set**. Primary metric is
**macro-F1** (the classes are imbalanced; macro-F1 refuses to reward majority
bias — rationale in `planning.md §5`).

### Per-class (fine-tuned MiniLM)
| Label | Precision | Recall | F1 | n |
|---|---|---|---|---|
| analysis | 0.56 | 0.82 | 0.67 | 11 |
| hot_take | 0.50 | 0.64 | 0.56 | 14 |
| reaction | 0.86 | 0.52 | 0.65 | 23 |

### Confusion matrices
Saved as `confusion_matrix.png` (fine-tuned) and `confusion_matrix_baseline.png`
(Groq). Raw counts (rows = true, cols = pred; order analysis/hot_take/reaction):

```
Fine-tuned          Groq baseline
        a  h  r              a  h  r
   a  [ 9  2  0 ]       a  [ 6  2  3 ]
   h  [ 3  9  2 ]       h  [ 2  9  3 ]
   r  [ 4  7 12 ]       r  [ 1  1 21 ]
```

The fine-tuned model over-predicts `analysis`/`hot_take` on long `reaction`
comments (bottom row); the baseline is much cleaner on `reaction` (21/23).

### Three errors the fine-tuned model made, and why
1. **True `hot_take` → predicted `analysis` (conf 0.97):** *"I'm not a fan of how
   the MC is written. He has little agency… the daemons don't outclass him, that
   stunt pulls me out."* — It's **long and structured-sounding**, so the model
   called it analysis, but it's unsupported taste (no verifiable evidence) =
   `hot_take`. **The model learned "long + structured = analysis," not "has
   evidence = analysis."**
2. **True `reaction` → predicted `analysis` (conf 0.97):** *"…brought a tear to my
   eye… Season 1 has been solid. The pacing felt rushed to set up the Zenza
   arc…"* — genuinely borderline (emotion + an adaptation observation); we labeled
   it `reaction` because the purpose is emotional. The model latched onto the
   structural clause.
3. **True `hot_take` → predicted `reaction` (conf 0.96):** *"25 minutes of straight
   fight animation, we finally have some budget"* — a (sarcastic) claim about
   production, but short and casual, so the model read it as a feeling.

**Systematic error pattern (stretch goal):** the model's mistake is almost
entirely **length/structure ≠ substance**. It over-assigns `analysis`/`hot_take`
to longer comments and confuses long emotional reactions for reasoning (11 of 18
errors are `reaction` misclassified upward). It uses surface length as a proxy
for the kind of reasoning it can't actually verify.

**Confidence calibration (stretch goal):** the model is **overconfident and
poorly calibrated** — 40 of 48 predictions are made at >0.90 confidence, but only
**68%** of those are correct (and the 0.7–0.9 bin is *less* accurate at 33%). A
"0.95 confidence" prediction is not meaningfully more trustworthy than a 0.70 one
here. Don't use these scores as a reliability signal.

---

## Reflection: what the model learned vs. what I intended

**Intended:** a classifier that detects *whether a comment reasons from verifiable
evidence* — the actual thing separating `analysis` from a confident `hot_take`.

**Learned:** a proxy — *comment length and structural surface features*. The
clearest evidence is error #1: a long, well-formatted **opinion** gets called
`analysis` at 0.97 confidence because it *looks* like reasoning. The model never
learned to check for evidence (it can't verify claims), so it substituted the
correlated-but-wrong signal it *could* see: length and connective structure.

**Why the 70B baseline won:** (1) it has world knowledge to judge whether a claim
is actually supported, which a 22M model trained on 220 noisy examples cannot
acquire; (2) our **label noise** (the partial blind re-review) puts a ceiling on
what fine-tuning could learn; (3) 220 examples is tiny. The honest lesson:
**fine-tuning a small model is not automatically better than prompting a large
one** — it depends on data volume, label quality, and whether the task needs
knowledge the small model doesn't have. For a subjective, knowledge-dependent
task like "is this evidence real?", the big zero-shot model is hard to beat
without far more, cleaner data.

## AI usage disclosure

- **Label pre-suggestions:** Groq `llama-3.3-70b` suggested a label per comment
  (`prelabel.py`); a `prelabeled` column flags every machine-suggested row. **All
  315 were machine-suggested**, then human-reviewed. As disclosed in §2, the first
  human pass rubber-stamped ~99%; a partial blind re-review corrected the
  `analysis` class (40% flip rate). The baseline being the *same* Groq model is
  why this matters — and why the labeling honesty is load-bearing for the
  comparison's validity.
- **Label stress-testing:** AI generated boundary cases during label design,
  which surfaced the bare-superlative tiebreaker now in `planning.md §3`.
- **Failure analysis:** AI assisted in summarizing error patterns; every pattern
  reported here was verified against the actual misclassified examples.
- **Code:** the collection/training/eval scripts were written with AI assistance.

### Stretch features attempted
-  **Confidence calibration** — reported (model is overconfident/miscalibrated).
-  **Systematic error-pattern analysis** — "length/structure ≠ substance".
-  **Deployed interface** — `scripts/predict.py` takes a post and prints the
  predicted label + confidence + full class distribution:
  ```
  ./.venv/bin/python scripts/predict.py "your r/anime comment here"
  ./.venv/bin/python scripts/predict.py          # runs the built-in demo posts
  ```
- ❌ Inter-annotator reliability — not done (time).
