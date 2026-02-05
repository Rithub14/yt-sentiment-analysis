from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def _normalize_label(label: str) -> str:
    upper = label.upper()
    if "NEG" in upper:
        return "NEGATIVE"
    if "POS" in upper:
        return "POSITIVE"
    if "NEU" in upper:
        return "NEUTRAL"
    return "NEUTRAL"


def _load_rows(csv_path: Path) -> List[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    rows: List[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("No labels found in CSV.")
    return rows


def _confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str]) -> List[List[int]]:
    index = {label: i for i, label in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for truth, pred in zip(y_true, y_pred):
        matrix[index[truth]][index[pred]] += 1
    return matrix


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, float]:
    accuracy = float((y_true == y_pred).mean())
    f1_scores = []
    for cls in range(len(classes)):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {"accuracy": accuracy, "f1": macro_f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation report.")
    parser.add_argument("--input", default="data/processed/comments_cleaned.csv")
    parser.add_argument("--output", default="reports/eval_report.json")
    parser.add_argument("--model-dir", default="models/sentiment_model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_dir = Path(args.model_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_path)
    texts = [str(row.get("text", "")) for row in rows]
    y_true_labels = [_normalize_label(str(row.get("label", ""))) for row in rows]

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing trained model directory: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    preds = clf(texts, batch_size=16)
    y_pred_labels = [_normalize_label(str(p.get("label", ""))) for p in preds]

    classes = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    y_true = np.array([classes.index(lbl) for lbl in y_true_labels])
    y_pred = np.array([classes.index(lbl) for lbl in y_pred_labels])

    report = {
        "metrics": _metrics(y_true, y_pred, classes),
        "confusion_matrix": _confusion_matrix(y_true_labels, y_pred_labels, classes),
        "classes": classes,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
