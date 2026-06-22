"""
prelabel.py — Use Groq (llama-3.3-70b-versatile) to SUGGEST a label per comment.

These are suggestions ONLY. review.py forces you to read and confirm/correct each
one; the human label is authoritative (planning.md §7b). The `prelabeled` column
records that a row was machine-suggested, for disclosure in the README.

LEAKAGE GUARD: do NOT pre-label rows you intend to put in the test set with the
same model you baseline against. Practically: pre-label everything to speed
review, but during review keep a clean, human-only pass for the test split, OR
just remember the baseline is zero-shot on text it never saw labels for (it never
sees gold labels regardless). The real guard is that gold labels are YOUR calls.

Usage:
    ./.venv/bin/python scripts/prelabel.py
Input:  data/raw_unlabeled.csv
Output: data/prelabeled.csv
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LABELS = ["analysis", "hot_take", "reaction"]
BATCH = 12

# Condensed from planning.md §2-§3 so the suggester applies the SAME rules.
SYSTEM = """You label r/anime comments on a discourse-quality spectrum. Exactly one label per comment.

analysis  — structured argument about writing/themes/direction/pacing/animation, backed by SPECIFIC, VERIFIABLE evidence (named scene, directing/animation choice, source-material/production fact). It reasons.
hot_take  — a bold, confident OPINION or evaluative claim stated WITHOUT real supporting evidence. It asserts.
reaction  — an immediate EMOTIONAL response to a specific moment/episode, little/no argument. It feels.

Decision order: (1) has specific verifiable evidence supporting a claim -> analysis; (2) else, a flat evaluative claim about the work/genre/creator -> hot_take; (3) else, an in-the-moment feeling -> reaction.
Decorative evidence (one fact propping up an unsupported verdict) -> hot_take, not analysis.
Bare superlative on a specific episode with emotional markers (caps/emoji/!/"I"/slang) -> reaction; flat generalizing verdict -> hot_take.

Return ONLY a JSON array like [{"i":0,"label":"reaction"},...] — one object per input comment, same order. No prose."""


def suggest(client: Groq, batch: list[str]) -> list[str]:
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(batch))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Label these {len(batch)} comments:\n\n{numbered}"},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    # model may wrap the array in an object; handle both
    data = json.loads(raw)
    if isinstance(data, dict):
        # find the first list value
        data = next((v for v in data.values() if isinstance(v, list)), [])
    out = ["reaction"] * len(batch)
    for obj in data:
        i = obj.get("i")
        lab = str(obj.get("label", "")).strip().lower()
        if isinstance(i, int) and 0 <= i < len(batch) and lab in LABELS:
            out[i] = lab
    return out


def main() -> int:
    key = os.getenv("GROQ_API_KEY")
    if not key or "your_groq" in key:
        print("ERROR: GROQ_API_KEY missing in .env", file=sys.stderr)
        return 1
    src = ROOT / "data" / "raw_unlabeled.csv"
    if not src.exists():
        print(f"ERROR: {src} not found. Run collect.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(src)
    client = Groq(api_key=key)
    suggestions: list[str] = []
    texts = df["text"].astype(str).tolist()
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        try:
            labs = suggest(client, batch)
        except Exception as e:  # noqa: BLE001
            print(f"  batch {start} failed ({e}); defaulting to reaction", file=sys.stderr)
            labs = ["reaction"] * len(batch)
        suggestions.extend(labs)
        print(f"  pre-labeled {min(start + BATCH, len(texts))}/{len(texts)}")

    df["suggested_label"] = suggestions
    df["label"] = suggestions          # starting point; review.py overwrites with human calls
    df["prelabeled"] = True
    df["reviewed"] = False
    df["notes"] = ""
    out = ROOT / "data" / "prelabeled.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} suggestions -> {out.relative_to(ROOT)}")
    print("Suggested distribution:", df["suggested_label"].value_counts().to_dict())
    print("\nNEXT: review every one ->  ./.venv/bin/python scripts/review.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
