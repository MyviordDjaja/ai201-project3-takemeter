"""
intake.py — Turn manually-pasted comments (data/collected_raw.txt) into the
pipeline's unlabeled CSV. Use this when the Reddit API isn't available.

Comments in collected_raw.txt are separated by a line containing only @@@.
Reuses collect.py's cleaning/junk filter so manual and API paths behave the same.

Usage: ./.venv/bin/python scripts/intake.py
Output: data/raw_unlabeled.csv  -> then prelabel.py -> review.py -> export.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# reuse the exact cleaning rules from the API path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import clean, is_junk  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "collected_raw.txt"
OUT = ROOT / "data" / "raw_unlabeled.csv"
DELIM = "@@@"


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found.", file=sys.stderr)
        return 1
    lines = SRC.read_text(encoding="utf-8").splitlines()
    # A delimiter is a line that is EXACTLY @@@ (so prose mentioning @@@ is safe).
    # Everything before the first delimiter line is preamble and is discarded.
    delim_idxs = [i for i, ln in enumerate(lines) if ln.strip() == DELIM]
    blocks: list[str] = []
    if delim_idxs:
        for a, b in zip(delim_idxs, delim_idxs[1:] + [len(lines)]):
            blocks.append("\n".join(lines[a + 1:b]).strip())
    blocks = [b for b in blocks]

    rows: list[dict] = []
    seen: set[str] = set()
    dropped_junk = dropped_dup = dropped_placeholder = 0
    for b in blocks:
        if not b:
            continue
        if b.upper().startswith("PASTE "):  # leftover template line
            dropped_placeholder += 1
            continue
        body = clean(b)
        if is_junk(body, None):
            dropped_junk += 1
            continue
        body = body[:1500]
        key = body.lower()[:120]
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        rows.append({"id": f"m{len(rows):04d}", "text": body, "score": "",
                     "permalink": "", "thread": "", "flair": "manual"})

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "score", "permalink", "thread", "flair"])
        w.writeheader()
        w.writerows(rows)

    print(f"Parsed {len(rows)} usable comments -> {OUT.relative_to(ROOT)}")
    print(f"  dropped: {dropped_junk} junk/too-short, {dropped_dup} duplicates, "
          f"{dropped_placeholder} template placeholders")
    if dropped_placeholder:
        print("  NOTE: template placeholder lines detected — make sure you replaced them.")
    if len(rows) < 230:
        print(f"  Only {len(rows)} so far — gather more (aim ~230-250) to clear 200 after labeling.")
    else:
        print("  Good headroom. NEXT: ./.venv/bin/python scripts/prelabel.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
