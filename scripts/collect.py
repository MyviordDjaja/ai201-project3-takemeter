"""
collect.py — Pull real public comments from r/anime into an unlabeled CSV.

Sources (per planning.md §4): Episode Discussion threads + Discussion/Rewatch
posts. NOT recommendation-request threads. Read-only PRAW; no Reddit password
needed. Applies the planning.md "catch-all exclusion" policy at collection time
so the 3 labels stay >=90% exhaustive over what remains.

Usage:
    ./.venv/bin/python scripts/collect.py            # default targets
    ./.venv/bin/python scripts/collect.py --pool 450 # larger raw pool

Output: data/raw_unlabeled.csv  (columns: id,text,score,permalink,thread,flair)
This file is gitignored (intermediate). The committed dataset is the reviewed one.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import praw

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- collection knobs ---------------------------------------------------------
# We pull from a few thread types because each concentrates different labels:
#   Episode discussion -> mostly reaction + some hot_take
#   Discussion / Rewatch -> where analysis + hot_take concentrate (oversample!)
SEARCHES = [
    ('flair:"Episode"', "episode"),
    ('flair:"Discussion"', "discussion"),
    ('flair:"Rewatch"', "rewatch"),
]
MIN_CHARS = 20         # drop one-word/emoji junk; keeps "this episode was garbage"
MAX_CHARS = 1500       # truncate essays so they fit DistilBERT's window comfortably
BOT_AUTHORS = {"automoderator", "anime_irl", "sneakpeekbot", "[deleted]"}
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")
QUOTE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)  # drop Reddit blockquote lines


def clean(text: str) -> str:
    text = QUOTE_RE.sub("", text or "")
    text = URL_RE.sub("", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def is_junk(text: str, author: str | None) -> bool:
    if not text or text in ("[deleted]", "[removed]"):
        return True
    if author and (author.lower() in BOT_AUTHORS or author.lower().endswith("bot")):
        return True
    if len(text) < MIN_CHARS:
        return True
    # mostly-link or quote-only or single-emoji-ish: too few real word chars
    if len(re.sub(r"[^A-Za-z0-9]", "", text)) < 12:
        return True
    # pure question with no take (planning.md exclusion): single short sentence ending in ?
    if text.endswith("?") and len(text.split()) < 12 and text.count(".") == 0:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=400,
                    help="target size of the raw unlabeled pool (default 400)")
    ap.add_argument("--threads-per-search", type=int, default=8)
    ap.add_argument("--comments-per-thread", type=int, default=25)
    args = ap.parse_args()

    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT")
    if not all([cid, csec, ua]) or "your_client_id" in (cid or ""):
        print("ERROR: Reddit creds missing. Copy .env.example -> .env and fill it in.",
              file=sys.stderr)
        return 1

    reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)
    reddit.read_only = True
    sub = reddit.subreddit("anime")

    seen: set[str] = set()
    rows: list[dict] = []

    for query, tag in SEARCHES:
        try:
            threads = list(sub.search(query, sort="new", limit=args.threads_per_search))
        except Exception as e:  # noqa: BLE001
            print(f"  search {query!r} failed: {e}", file=sys.stderr)
            continue
        print(f"[{tag}] {len(threads)} threads")
        for t in threads:
            try:
                t.comment_sort = "top"  # bias toward higher-effort comments
                t.comments.replace_more(limit=0)
                comments = t.comments.list()[: args.comments_per_thread]
            except Exception as e:  # noqa: BLE001
                print(f"    thread {t.id} failed: {e}", file=sys.stderr)
                continue
            kept = 0
            for c in comments:
                body = clean(getattr(c, "body", ""))
                author = str(getattr(c, "author", "") or "")
                if is_junk(body, author):
                    continue
                body = body[:MAX_CHARS]
                key = body.lower()[:120]
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "id": c.id,
                    "text": body,
                    "score": getattr(c, "score", 0),
                    "permalink": f"https://reddit.com{c.permalink}",
                    "thread": t.title,
                    "flair": tag,
                })
                kept += 1
            print(f"    {t.id}  +{kept}  (pool={len(rows)})")
            if len(rows) >= args.pool:
                break
        if len(rows) >= args.pool:
            break

    out = ROOT / "data" / "raw_unlabeled.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "score", "permalink", "thread", "flair"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} unlabeled comments -> {out.relative_to(ROOT)}")
    by_flair: dict[str, int] = {}
    for r in rows:
        by_flair[r["flair"]] = by_flair.get(r["flair"], 0) + 1
    print("By source:", by_flair)
    if len(rows) < 280:
        print("WARNING: pool < 280. Re-run with a larger --pool or more threads;",
              "you want headroom above 240 after labeling/exclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
