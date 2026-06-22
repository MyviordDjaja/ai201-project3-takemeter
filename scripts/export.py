"""
export.py — Produce the single committed dataset from reviewed rows.

Output: data/takemeter_labeled.csv  (columns: text,label,notes,prelabeled)
This is the ONE complete labeled file the Colab notebook ingests; the notebook
does the 70/15/15 split itself, so we do NOT pre-split here.

Usage: ./.venv/bin/python scripts/export.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "prelabeled.csv"
OUT = ROOT / "data" / "takemeter_labeled.csv"
LABELS = ["analysis", "hot_take", "reaction"]


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found. Run the collect -> prelabel -> review pipeline.", file=sys.stderr)
        return 1
    df = pd.read_csv(SRC).fillna({"notes": ""})
    # Blind re-judgments (review_blind.py) are authoritative where they exist.
    if "blind_reviewed" in df.columns:
        bl = df["blind_reviewed"].fillna(False).astype(bool)
        df.loc[bl, "label"] = df.loc[bl, "blind_label"]
        print(f"Applied {int(bl.sum())} blind-reviewed labels over the first pass.")
    if "reviewed" in df.columns:
        unrev = int((~df["reviewed"].astype(bool)).sum())
        if unrev:
            print(f"NOTE: {unrev} rows still unreviewed — they are excluded. "
                  f"Run review.py to finish them.")
        df = df[df["reviewed"].astype(bool)]
    df = df[df["label"].isin(LABELS)].copy()  # drops __EXCLUDE__

    final = df[["text", "label", "notes"]].copy()
    final["prelabeled"] = df.get("prelabeled", True)
    final.to_csv(OUT, index=False)

    n = len(final)
    counts = final["label"].value_counts()
    print(f"Wrote {n} labeled examples -> {OUT.relative_to(ROOT)}\n")
    print("Label distribution:")
    for lab in LABELS:
        c = int(counts.get(lab, 0))
        print(f"  {lab:9s} {c:4d}  ({c / n:.0%})" if n else f"  {lab}: 0")

    ok = True
    if n < 200:
        print(f"\n[FAIL] {n} < 200 required examples."); ok = False
    top = counts.max() / n if n else 1
    if top > 0.70:
        print(f"\n[FAIL] top class {top:.0%} > 70% — rebalance."); ok = False
    if any((counts.get(l, 0) / n if n else 0) < 0.20 for l in LABELS):
        print("\n[WARN] a class is <20% (planning.md target). Consider oversampling it.")
    if ok:
        print("\n[OK] >=200 examples, no class over 70%. Ready for the notebook.")
        n_pre = int(final["prelabeled"].sum())
        print(f"     ({n_pre}/{n} were machine-pre-labeled then human-reviewed — disclose in README.)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
