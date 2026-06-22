"""
baseline_groq.py — Zero-shot baseline: classify the SAME test set with
llama-3.3-70b-versatile via Groq, no task-specific training.

Reads data/splits/test.csv (written by train_eval.py) so both models are scored
on identical examples. Merges metrics into evaluation_results.json and saves
per-example predictions for error analysis.

Run: ./.venv/bin/python scripts/baseline_groq.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
LABELS = ["analysis", "hot_take", "reaction"]
L2I = {l: i for i, l in enumerate(LABELS)}

# Zero-shot prompt: label definitions, but NO training examples from our set.
SYSTEM = """You classify r/anime comments by discourse quality into exactly one label:

analysis  — structured argument about writing/themes/direction/pacing/animation backed by SPECIFIC, VERIFIABLE evidence (a named scene, directing/animation choice, source-material or production fact). It reasons.
hot_take  — a bold, confident OPINION or evaluative claim stated WITHOUT real supporting evidence. It asserts.
reaction  — an immediate EMOTIONAL response to a moment/episode, little or no argument. It feels.

Reply with ONLY one word: analysis, hot_take, or reaction."""


def classify(client, text):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile", temperature=0, max_tokens=4,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": text[:2000]}],
    )
    out = resp.choices[0].message.content.strip().lower()
    for l in LABELS:
        if l in out:
            return l
    return "reaction"  # safe fallback if it returns something odd


def main() -> int:
    key = os.getenv("GROQ_API_KEY")
    if not key or "your_groq" in key:
        print("ERROR: GROQ_API_KEY missing in .env", file=sys.stderr); return 1
    test_path = ROOT / "data" / "splits" / "test.csv"
    if not test_path.exists():
        print("ERROR: run train_eval.py first (need data/splits/test.csv).", file=sys.stderr); return 1

    test = pd.read_csv(test_path)
    test = test[test["label"].isin(LABELS)].reset_index(drop=True)
    client = Groq(api_key=key)
    preds = []
    for i, t in enumerate(test["text"].astype(str)):
        preds.append(classify(client, t))
        if (i + 1) % 10 == 0:
            print(f"  baseline {i+1}/{len(test)}")

    y_true = [L2I[l] for l in test["label"]]
    y_pred = [L2I[l] for l in preds]
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(len(LABELS)), zero_division=0)
    macro_f1 = f1.mean()
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    print(f"\nBASELINE  accuracy {acc:.3f} | macro-F1 {macro_f1:.3f}")
    for i, l in enumerate(LABELS):
        print(f"  {l:9s} P {p[i]:.2f} R {r[i]:.2f} F1 {f1[i]:.2f} (n={sup[i]})")

    test_out = test[["text", "label"]].copy()
    test_out["pred"] = preds
    test_out.to_csv(ROOT / "data" / "splits" / "test_predictions_baseline.csv", index=False)

    from train_eval import plot_cm  # reuse plotting
    plot_cm(cm, "Groq zero-shot baseline — test confusion", ROOT / "confusion_matrix_baseline.png")

    out = ROOT / "evaluation_results.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing["baseline"] = {
        "model": "llama-3.3-70b-versatile (zero-shot)",
        "accuracy": round(acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {LABELS[i]: {"precision": round(p[i], 4), "recall": round(r[i], 4),
                                  "f1": round(f1[i], 4), "support": int(sup[i])} for i in range(len(LABELS))},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "test_size": len(test),
    }
    if "finetuned" in existing:
        existing["comparison"] = {
            "finetuned_macro_f1": existing["finetuned"]["macro_f1"],
            "baseline_macro_f1": round(float(macro_f1), 4),
            "delta_macro_f1": round(existing["finetuned"]["macro_f1"] - float(macro_f1), 4),
        }
    out.write_text(json.dumps(existing, indent=2))
    print("\nSaved baseline -> evaluation_results.json + confusion_matrix_baseline.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
