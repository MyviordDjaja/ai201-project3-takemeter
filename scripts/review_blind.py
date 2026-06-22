"""
review_blind.py — Re-judge a subset of comments with Groq's suggestion HIDDEN.

Why: the first pass agreed with Groq ~99% of the time, which makes the labels
~identical to a zero-shot LLM and invalidates the baseline comparison. This pass
removes the anchor: you see ONLY the comment text and decide cold.

Subset (per your choice): every row currently labeled `analysis` (Groq's weakest
class) + a deterministic random 60 from the rest. Your blind calls go in a new
`blind_label` column; nothing else is touched. Afterward, scripts/blind_report.py
compares blind vs. Groq to tell us whether a full re-label is required.

Run in a real terminal:  ./.venv/bin/python scripts/review_blind.py
Resumable + autosaves. Store: data/prelabeled.csv
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "prelabeled.csv"
SAMPLE_NON_ANALYSIS = 60
SEED = 42

RULES = """
  Judge COLD — no suggestion shown. Pick the single best label.
  analysis  reasons — structured argument + SPECIFIC verifiable evidence (named scene/choice/source fact)
  hot_take  asserts — bold evaluative claim, NO real support
  reaction  feels   — in-the-moment emotional response to a moment/episode

  decorative single fact propping an unsupported verdict -> hot_take
  bare superlative on a specific episode w/ emotion -> reaction ; flat generalizing -> hot_take
  feeling that merely mentions a detail (not arguing) -> reaction
"""
KEYMAP = {"a": "analysis", "h": "hot_take", "r": "reaction"}
DIM, BOLD, OFF, RED = "\033[2m", "\033[1m", "\033[0m", "\033[31m"


def build_subset(df: pd.DataFrame) -> pd.Series:
    """Stable boolean mask: all current-analysis rows + random N of the rest."""
    is_analysis = df["label"] == "analysis"
    others = df.index[~is_analysis].tolist()
    rng = random.Random(SEED)
    pick = set(rng.sample(others, min(SAMPLE_NON_ANALYSIS, len(others))))
    return df.index.to_series().apply(lambda i: bool(is_analysis[i]) or i in pick)


def main() -> int:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found.", file=sys.stderr)
        return 1
    df = pd.read_csv(CSV)
    if "blind_label" not in df.columns:
        df["blind_label"] = ""
        df["blind_reviewed"] = False
        df["in_blind_subset"] = build_subset(df)
        df.to_csv(CSV, index=False)
    df["blind_reviewed"] = df["blind_reviewed"].fillna(False).astype(bool)
    df["in_blind_subset"] = df["in_blind_subset"].fillna(False).astype(bool)
    df["blind_label"] = df["blind_label"].fillna("")

    todo = df.index[df["in_blind_subset"] & ~df["blind_reviewed"]].tolist()
    total = int(df["in_blind_subset"].sum())
    if not todo:
        print(f"All {total} subset rows already blind-reviewed. "
              f"Run: ./.venv/bin/python scripts/blind_report.py")
        return 0

    print(RULES)
    print(f"{BOLD}{len(todo)} of {total} subset comments left to judge.{OFF}\n")
    done_count = total - len(todo)
    for i in todo:
        row = df.loc[i]
        print(f"\n{BOLD}[{done_count}/{total}]{OFF}  {DIM}score={row['score']} · {str(row['thread'])[:48]}{OFF}")
        print(f"  {row['text']}")
        while True:
            try:
                cmd = input(f"  {DIM}a=analysis h=hot_take r=reaction · x=exclude · s=save&quit >{OFF} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                df.to_csv(CSV, index=False)
                print(f"\nSaved. {done_count}/{total} done.")
                return 0
            if cmd in KEYMAP:
                df.at[i, "blind_label"] = KEYMAP[cmd]
                df.at[i, "blind_reviewed"] = True
                done_count += 1
                break
            if cmd == "x":
                df.at[i, "blind_label"] = "__EXCLUDE__"
                df.at[i, "blind_reviewed"] = True
                done_count += 1
                break
            if cmd == "s":
                df.to_csv(CSV, index=False)
                print(f"\nSaved. {done_count}/{total} done. Re-run to continue.")
                return 0
            print(f"    {RED}use a / h / r / x / s{OFF}")
        if done_count % 10 == 0:
            df.to_csv(CSV, index=False)

    df.to_csv(CSV, index=False)
    print(f"\n{BOLD}Done — all {total} blind-reviewed.{OFF} "
          f"Run: ./.venv/bin/python scripts/blind_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
