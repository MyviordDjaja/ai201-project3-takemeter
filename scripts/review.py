"""
review.py — Read EVERY comment and confirm/correct its label. (The actual work.)

This is the human-in-the-loop step the project requires. Pre-labels are only a
starting point; you read each comment and make the call. Resumable — re-run any
time and it picks up where you left off.

Keys per comment:
    [Enter]  accept the suggested label
    a / h / r   set analysis / hot_take / reaction
    n        add a note (for difficult cases — these feed planning.md §3)
    x        exclude this comment (junk / off-topic / not a take)
    b        go back to the previous comment
    ?        reprint the decision rules
    s        save and quit

Input/Output: data/prelabeled.csv (updated in place; `reviewed` flips to True)
When done:    ./.venv/bin/python scripts/export.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "prelabeled.csv"

RULES = """
  analysis  reasons — structured argument + SPECIFIC verifiable evidence
  hot_take  asserts — bold evaluative claim, NO real support
  reaction  feels   — in-the-moment emotional response to a moment/episode

  order: evidence->analysis ; else flat claim->hot_take ; else feeling->reaction
  decorative single fact propping an unsupported verdict -> hot_take
  bare superlative on a specific episode w/ emotion -> reaction ; flat generalizing -> hot_take
"""
KEYMAP = {"a": "analysis", "h": "hot_take", "r": "reaction"}
C = {"analysis": "\033[36m", "hot_take": "\033[33m", "reaction": "\033[35m",
     "dim": "\033[2m", "bold": "\033[1m", "off": "\033[0m", "red": "\033[31m"}


def dist(df: pd.DataFrame) -> dict:
    done = df[df["reviewed"] & (df["label"] != "__EXCLUDE__")]
    return done["label"].value_counts().to_dict()


def main() -> int:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found. Run collect.py then prelabel.py first.", file=sys.stderr)
        return 1
    df = pd.read_csv(CSV).fillna({"notes": "", "suggested_label": ""})
    for col, default in (("reviewed", False), ("notes", ""), ("label", "")):
        if col not in df.columns:
            df[col] = default
    df["reviewed"] = df["reviewed"].astype(bool)

    print(RULES)
    i = 0
    n = len(df)
    # jump to first unreviewed
    unrev = df.index[~df["reviewed"]].tolist()
    if unrev:
        i = unrev[0]

    while 0 <= i < n:
        if df.at[i, "reviewed"]:
            i += 1
            continue
        row = df.loc[i]
        n_done = int(df["reviewed"].sum())
        sug = row["suggested_label"]
        sug_c = C.get(sug, "")
        print(f"\n{C['bold']}[{n_done}/{n} reviewed]{C['off']}  {C['dim']}score={row['score']} · {str(row['thread'])[:50]}{C['off']}")
        print(f"  {row['text']}")
        print(f"  suggested: {sug_c}{sug}{C['off']}   {C['dim']}(Enter=accept · a/h/r · n=note · x=exclude · b=back · ?=rules · s=save&quit){C['off']}")
        try:
            cmd = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving…")
            break

        if cmd == "" and sug in KEYMAP.values():
            df.at[i, "label"] = sug
            df.at[i, "reviewed"] = True
            i += 1
        elif cmd.lower() in KEYMAP:
            df.at[i, "label"] = KEYMAP[cmd.lower()]
            df.at[i, "reviewed"] = True
            i += 1
        elif cmd.lower() == "x":
            df.at[i, "label"] = "__EXCLUDE__"
            df.at[i, "reviewed"] = True
            i += 1
        elif cmd.lower() == "n":
            note = input("    note: ").strip()
            df.at[i, "notes"] = (str(df.at[i, "notes"]) + " " + note).strip()
            print(f"    {C['dim']}noted — now set the label (Enter/a/h/r/x){C['off']}")
            # don't advance; let them set the label next loop
        elif cmd.lower() == "b":
            j = i - 1
            while j >= 0 and not df.at[j, "reviewed"]:
                j -= 1
            if j >= 0:
                df.at[j, "reviewed"] = False
                i = j
        elif cmd == "?":
            print(RULES)
        elif cmd.lower() == "s":
            break
        else:
            print(f"    {C['red']}unrecognized — Enter/a/h/r/n/x/b/?/s{C['off']}")
        # autosave every few
        if n_done % 10 == 0:
            df.to_csv(CSV, index=False)

    df.to_csv(CSV, index=False)
    d = dist(df)
    total = sum(d.values())
    print(f"\nSaved. Reviewed {int(df['reviewed'].sum())}/{n}.  Labeled (excl. excluded): {total}")
    print("Distribution:", d)
    if total:
        top = max(d.values()) / total
        if top > 0.70:
            print(f"{C['red']}WARNING: top class is {top:.0%} (>70%). Collect more of the minority classes.{C['off']}")
        if total < 200:
            print(f"{C['red']}Below 200 labeled — keep going.{C['off']}")
        if total >= 200 and top <= 0.70:
            print(f"{C['bold']}On track. When fully reviewed: ./.venv/bin/python scripts/export.py{C['off']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
