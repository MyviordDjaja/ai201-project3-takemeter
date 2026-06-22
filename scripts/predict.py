"""
predict.py — TakeMeter demo interface. Classify r/anime comments with the
fine-tuned model and show the predicted label + confidence (+ full distribution).

Usage:
  ./.venv/bin/python scripts/predict.py                 # runs the built-in demo posts
  ./.venv/bin/python scripts/predict.py "your comment"  # classify one comment
  echo "a comment" | ./.venv/bin/python scripts/predict.py -   # from stdin
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "finetuned"
LABELS = ["analysis", "hot_take", "reaction"]
C = {"analysis": "\033[36m", "hot_take": "\033[33m", "reaction": "\033[35m",
     "b": "\033[1m", "d": "\033[2m", "x": "\033[0m"}

# Built-in demo set: REAL r/anime test-set comments. True labels noted for
# narration. First 3 the model classifies CORRECTLY; the 4th it gets WRONG.
DEMO = [
    ("I swear... I had heart palpitations with that sudden tragedy 😭😭😭",
     "reaction (CORRECT)"),
    ("sao catches so much hate but for a lot of people it was the gateway anime, "
     "that nostalgia's earned not cringe",
     "hot_take (CORRECT)"),
    ("His kneejerk would be to remove the system but soften to ensuring the system "
     "is as ethical as possible. He would know the two are incredibly capable of "
     "handling the risks their job entails, considering they have done the job "
     "without him noticing... He would as an adult focus on what support they can "
     "give the Lycoris to mitigate risks, and try to improve the system, but I "
     "don't think he would want it torn down.",
     "analysis (CORRECT)"),
    ("I'm also not a huge fan of how the main character is being written. He has "
     "little to no agency in this story, and I am not emotionally engaged with his "
     "journey to find his parents at all. I know others feel he is unique, but I "
     "find his character rather bland and for some reason he is always "
     "participating in the fights.",
     "hot_take (WRONG — model says analysis)"),
]


def classify(model, tok, text):
    enc = tok([text], truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx]), {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


def show(text, label, conf, dist, note=""):
    bar = "  ".join(f"{C.get(l,'')}{l} {p:.0%}{C['x']}" for l, p in dist.items())
    print(f"\n{C['d']}post:{C['x']} {text[:160]}")
    if note:
        print(f"{C['d']}true:{C['x']} {note}")
    print(f"{C['b']}=> {C.get(label,'')}{label.upper()}{C['x']}  "
          f"{C['b']}confidence {conf:.0%}{C['x']}   {C['d']}[{bar}{C['d']}]{C['x']}")


def main() -> int:
    if not MODEL_DIR.exists():
        print("No fine-tuned model found. Run scripts/train_eval.py first.", file=sys.stderr)
        return 1
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    args = sys.argv[1:]
    if args == ["-"]:
        texts = [sys.stdin.read().strip()]
        notes = [""]
    elif args:
        texts = [" ".join(args)]
        notes = [""]
    else:
        texts = [t for t, _ in DEMO]
        notes = [n for _, n in DEMO]

    print(f"{C['b']}TakeMeter — fine-tuned r/anime discourse classifier{C['x']}")
    for text, note in zip(texts, notes):
        label, conf, dist = classify(model, tok, text)
        show(text, label, conf, dist, note)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
