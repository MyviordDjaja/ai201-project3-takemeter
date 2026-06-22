"""
blind_report.py — Did blind re-judging disagree with Groq / the first pass?

Reads data/prelabeled.csv after review_blind.py. Reports agreement on the
blind-reviewed subset and recommends whether a FULL blind re-label is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "prelabeled.csv"


def main() -> int:
    df = pd.read_csv(CSV)
    if "blind_reviewed" not in df.columns:
        print("No blind review yet — run review_blind.py first.", file=sys.stderr)
        return 1
    sub = df[df["blind_reviewed"].fillna(False).astype(bool)].copy()
    if sub.empty:
        print("No blind-reviewed rows yet.", file=sys.stderr)
        return 1

    # exclude rows the blind pass marked junk from agreement math
    judged = sub[sub["blind_label"] != "__EXCLUDE__"]
    vs_groq = (judged["blind_label"] == judged["suggested_label"]).mean()
    vs_first = (judged["blind_label"] == judged["label"]).mean()

    print(f"Blind-reviewed: {len(sub)}  (excluded {int((sub['blind_label']=='__EXCLUDE__').sum())})")
    print(f"\nBlind vs Groq suggestion : {vs_groq:.0%} agree  ({1-vs_groq:.0%} disagree)")
    print(f"Blind vs your first pass : {vs_first:.0%} agree  ({1-vs_first:.0%} disagree)")

    # focus on the analysis class (Groq's weakest)
    an = sub[sub["label"] == "analysis"]
    if len(an):
        stayed = (an["blind_label"] == "analysis").mean()
        print(f"\nOf {len(an)} first-pass 'analysis' rows, blind kept {stayed:.0%} as analysis "
              f"({1-stayed:.0%} were really hot_take/reaction).")
        moved = an[an["blind_label"] != "analysis"]
        for _, r in moved.head(5).iterrows():
            print(f"  analysis -> {r['blind_label']}: {str(r['text'])[:130]}")

    disagree = 1 - vs_first
    print("\n" + "=" * 60)
    if disagree >= 0.15:
        print(f"VERDICT: {disagree:.0%} disagreement on the sample is HIGH.")
        print("Your first-pass labels are unreliable across the board — the rubber-")
        print("stamp affected all classes, not just analysis. RECOMMEND: full blind")
        print("re-label (raise SAMPLE to all rows in review_blind.py, or relabel the rest).")
    elif disagree >= 0.07:
        print(f"VERDICT: {disagree:.0%} disagreement is MODERATE.")
        print("The analysis-class fixes likely cover most of the damage. Defensible to")
        print("proceed with blind labels applied, but note the limitation in the README.")
    else:
        print(f"VERDICT: {disagree:.0%} disagreement is LOW.")
        print("First-pass labels hold up under blind scrutiny. Apply blind corrections")
        print("and proceed.")
    print("Next: ./.venv/bin/python scripts/export.py  (prefers blind_label where present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
