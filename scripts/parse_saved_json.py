"""
parse_saved_json.py — Turn browser-saved Reddit thread .json into the pipeline CSV.

Why this exists: automated clients (curl/requests/PRAW-without-creds) get a 403
from Reddit on this network, but your BROWSER loads Reddit fine. So the browser
does the fetch; this script only reads local files — no network at all.

WORKFLOW
  1. In your browser (logged in), open each thread with .json appended, e.g.
       https://www.reddit.com/r/anime/comments/abc123/some_title/.json
     You should see raw JSON, not the "blocked" page. Add ?limit=500&raw_json=1
     to the URL to get more comments and un-escaped text:
       .../some_title/.json?limit=500&raw_json=1
  2. Save each page (Ctrl+S) into  data/saved_threads/  with a .json name.
     Grab a MIX of thread types (planning.md §4): Episode Discussion threads
     (reaction-heavy) AND Rewatch/Discussion posts (where `analysis` lives).
  3. ./.venv/bin/python scripts/parse_saved_json.py

Output: data/raw_unlabeled.csv  ->  prelabel.py  ->  review.py  ->  export.py
Reuses collect.py's clean()/is_junk() so exclusions match planning.md exactly.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import clean, is_junk  # noqa: E402  (shared cleaning rules)

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "saved_threads"
OUT = ROOT / "data" / "raw_unlabeled.csv"
PER_THREAD = 40          # cap per thread -> spread across more threads for topic diversity
MAX_CHARS = 1500


def walk(children: list, out: list) -> None:
    """Flatten Reddit's nested comment tree; skip 'more' stubs and non-comments."""
    for child in children:
        if not isinstance(child, dict) or child.get("kind") != "t1":
            continue
        d = child.get("data", {})
        out.append(d)
        replies = d.get("replies")
        if isinstance(replies, dict):  # "" when a comment has no replies
            walk(replies.get("data", {}).get("children", []), out)


def main() -> int:
    files = sorted(glob.glob(str(IN_DIR / "*.json")))
    if not files:
        print(f"No .json files in {IN_DIR.relative_to(ROOT)}/ — save thread JSON there first.\n"
              f"(See the workflow comment at the top of this script.)", file=sys.stderr)
        return 1

    seen: set[str] = set()
    rows: list[dict] = []
    for path in files:
        name = Path(path).name
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            title = payload[0]["data"]["children"][0]["data"]["title"]
            flat: list[dict] = []
            walk(payload[1]["data"]["children"], flat)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"  SKIPPED {name} (not a valid thread .json: {e})")
            continue

        flat.sort(key=lambda d: d.get("score", 0), reverse=True)  # effort floats up
        kept = 0
        for d in flat:
            if kept >= PER_THREAD:
                break
            text = clean(d.get("body", ""))
            if is_junk(text, d.get("author")):
                continue
            text = text[:MAX_CHARS]
            key = text.lower()[:120]
            if key in seen:
                continue
            seen.add(key)
            permalink = d.get("permalink", "")
            rows.append({
                "id": d.get("id", f"s{len(rows):04d}"),
                "text": text,
                "score": d.get("score", 0),
                "permalink": f"https://reddit.com{permalink}" if permalink else "",
                "thread": title[:120],
                "flair": "saved_json",
            })
            kept += 1
        print(f"  {kept:3d} kept from: {title[:60]}")

    if not rows:
        print("No usable comments found in the saved files.", file=sys.stderr)
        return 1

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "score", "permalink", "thread", "flair"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} comments from {len(files)} file(s) -> {OUT.relative_to(ROOT)}")
    if len(rows) < 240:
        print(f"Only {len(rows)} so far — save a few more threads (aim >240) and re-run.")
    else:
        print("Good headroom. NEXT: ./.venv/bin/python scripts/prelabel.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
