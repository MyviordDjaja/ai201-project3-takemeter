"""
error_analysis.py — Pull the fine-tuned model's mistakes for the eval report,
plus a confidence-calibration check and an error-pattern summary (stretch goals).

Reads data/splits/test_predictions_finetuned.csv (from train_eval.py).
Run: ./.venv/bin/python scripts/error_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["analysis", "hot_take", "reaction"]


def main() -> int:
    p = ROOT / "data" / "splits" / "test_predictions_finetuned.csv"
    if not p.exists():
        print("Run train_eval.py first.")
        return 1
    d = pd.read_csv(p)
    wrong = d[d["label"] != d["pred"]].copy()
    print(f"Errors: {len(wrong)}/{len(d)} ({len(wrong)/len(d):.0%})\n")

    print("=== Misclassified examples (true -> pred, confidence) ===")
    for _, r in wrong.sort_values("confidence", ascending=False).head(8).iterrows():
        print(f"[{r['label']} -> {r['pred']}  conf {r['confidence']:.2f}] {str(r['text'])[:160]}")
    print()

    # error pattern: which confusions dominate
    print("=== Error pattern: confusion pairs ===")
    pair = wrong.groupby(["label", "pred"]).size().sort_values(ascending=False)
    for (t, pr), n in pair.items():
        print(f"  {t} mistaken for {pr}: {n}")
    avg_len_wrong = wrong["text"].str.len().mean()
    avg_len_right = d[d["label"] == d["pred"]]["text"].str.len().mean()
    print(f"\n  avg length  wrong: {avg_len_wrong:.0f} chars | correct: {avg_len_right:.0f} chars")

    # confidence calibration (stretch): are high-conf preds more accurate?
    print("\n=== Confidence calibration ===")
    d["correct"] = (d["label"] == d["pred"]).astype(int)
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    calib = []
    for lo, hi in bins:
        m = (d["confidence"] >= lo) & (d["confidence"] < hi)
        if m.sum():
            acc = d.loc[m, "correct"].mean()
            print(f"  conf [{lo:.1f},{hi:.1f}): {m.sum():3d} preds, {acc:.0%} accurate")
            calib.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "accuracy": round(float(acc), 3)})

    # persist into evaluation_results.json
    out = ROOT / "evaluation_results.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing["error_analysis"] = {
        "n_errors": int(len(wrong)),
        "confusion_pairs": {f"{t}->{pr}": int(n) for (t, pr), n in pair.items()},
        "avg_len_wrong": round(float(avg_len_wrong), 1),
        "avg_len_correct": round(float(avg_len_right), 1),
        "calibration": calib,
        "examples": [{"text": str(r["text"]), "true": r["label"], "pred": r["pred"],
                      "confidence": float(r["confidence"])}
                     for _, r in wrong.sort_values("confidence", ascending=False).head(6).iterrows()],
    }
    out.write_text(json.dumps(existing, indent=2))
    print("\nSaved error_analysis -> evaluation_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
