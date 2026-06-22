"""
train_eval.py — Fine-tune distilbert-base-uncased on the labeled data and
evaluate on a held-out test set. Runs on CPU (315 examples, a few minutes).

Pipeline: stratified 70/15/15 split -> tokenize -> fine-tune -> evaluate
(accuracy, per-class P/R/F1, macro-F1, confusion matrix). Saves the test split
so the Groq baseline scores the EXACT same examples.

Outputs:
  data/splits/{train,val,test}.csv
  data/splits/test_predictions_finetuned.csv   (text,true,pred,confidence)
  confusion_matrix.png                          (fine-tuned model)
  evaluation_results.json                       (finetuned section)
  models/finetuned/                             (saved model + tokenizer)

Run: ./.venv/bin/python scripts/train_eval.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["analysis", "hot_take", "reaction"]
L2I = {l: i for i, l in enumerate(LABELS)}
# all-MiniLM-L6-v2: BERT-family, already cached locally, ~22M params -> fast CPU
# training, no 268MB download. (brief allows "another pre-trained model of your choice")
SEED, MODEL_NAME = 42, "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS, LR, BATCH, MAXLEN = 12, 3e-5, 16, 128  # 12 ep + class-weighted loss: small data needs more passes & imbalance correction to avoid majority-class collapse; len128 covers ~all comments


def seed_all(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class TextDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(list(texts), truncation=True, padding=True,
                       max_length=MAXLEN, return_tensors="pt")
        self.labels = torch.tensor([L2I[l] for l in labels])

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return ({k: v[i] for k, v in self.enc.items()}, self.labels[i])


def main() -> int:
    seed_all(SEED)
    df = pd.read_csv(ROOT / "data" / "takemeter_labeled.csv")
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)

    # stratified 70/15/15
    train_df, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)
    splits = ROOT / "data" / "splits"; splits.mkdir(exist_ok=True)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d.to_csv(splits / f"{name}.csv", index=False)
    print(f"split: train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
    model.train()
    train_dl = DataLoader(TextDS(train_df["text"], train_df["label"], tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # class-weighted loss (inverse frequency) to counter reaction-majority collapse
    from collections import Counter
    cnt = Counter(train_df["label"])
    cw = torch.tensor([len(train_df) / (len(LABELS) * cnt[l]) for l in LABELS], dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw)
    print(f"class weights: {dict(zip(LABELS, [round(w, 2) for w in cw.tolist()]))}")

    for ep in range(EPOCHS):
        tot = 0.0
        for enc, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(**enc).logits, y)
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1}/{EPOCHS}  loss {tot/len(train_dl):.4f}")

    # evaluate on test
    model.eval()
    test_dl = DataLoader(TextDS(test_df["text"], test_df["label"], tok), batch_size=BATCH)
    preds, confs = [], []
    with torch.no_grad():
        for enc, _ in test_dl:
            probs = torch.softmax(model(**enc).logits, dim=-1)
            p = probs.argmax(-1)
            preds.extend(p.tolist())
            confs.extend(probs.max(-1).values.tolist())
    y_true = [L2I[l] for l in test_df["label"]]

    acc = accuracy_score(y_true, preds)
    p, r, f1, sup = precision_recall_fscore_support(y_true, preds, labels=range(len(LABELS)), zero_division=0)
    macro_f1 = f1.mean()
    cm = confusion_matrix(y_true, preds, labels=range(len(LABELS)))
    print(f"\nFINE-TUNED  accuracy {acc:.3f} | macro-F1 {macro_f1:.3f}")
    for i, l in enumerate(LABELS):
        print(f"  {l:9s} P {p[i]:.2f} R {r[i]:.2f} F1 {f1[i]:.2f} (n={sup[i]})")

    # save per-example predictions for error analysis + the baseline reuse
    pred_df = test_df[["text", "label"]].copy()
    pred_df["pred"] = [LABELS[i] for i in preds]
    pred_df["confidence"] = [round(c, 4) for c in confs]
    pred_df.to_csv(splits / "test_predictions_finetuned.csv", index=False)

    # confusion matrix png
    plot_cm(cm, "Fine-tuned DistilBERT — test confusion", ROOT / "confusion_matrix.png")

    results = {
        "finetuned": {
            "model": MODEL_NAME,
            "hyperparams": {"epochs": EPOCHS, "lr": LR, "batch": BATCH, "max_len": MAXLEN},
            "accuracy": round(acc, 4),
            "macro_f1": round(float(macro_f1), 4),
            "per_class": {LABELS[i]: {"precision": round(p[i], 4), "recall": round(r[i], 4),
                                      "f1": round(f1[i], 4), "support": int(sup[i])} for i in range(len(LABELS))},
            "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
            "test_size": len(test_df),
        }
    }
    out = ROOT / "evaluation_results.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=2))

    model.save_pretrained(ROOT / "models" / "finetuned")
    tok.save_pretrained(ROOT / "models" / "finetuned")
    print(f"\nSaved model, confusion_matrix.png, evaluation_results.json")
    return 0


def plot_cm(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right"); ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
