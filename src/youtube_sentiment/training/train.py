from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
EXPERIMENT_NAME = "youtube-sentiment"


@dataclass
class Example:
    text: str
    label: str


class CommentDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer: Any, label2id: dict[str, int]):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.label2id[ex.label], dtype=torch.long)
        return item


def load_examples(path: Path) -> list[Example]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    examples: list[Example] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'text' and 'label' columns.")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                continue
            examples.append(Example(text=text, label=label))
    if not examples:
        raise ValueError("No valid rows found in the CSV.")
    return examples


def split_examples(examples: list[Example], train_ratio: float = 0.8) -> tuple[list[Example], list[Example]]:
    rng = np.random.default_rng(42)
    indices = np.arange(len(examples))
    rng.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train = [examples[i] for i in train_idx]
    val = [examples[i] for i in val_idx]
    return train, val


def build_label_map(examples: list[Example]) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted({ex.label for ex in examples})
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = float((preds == labels).mean())

    num_classes = int(max(labels.max(), preds.max()) + 1) if len(labels) else 0
    f1_scores = []
    for class_id in range(num_classes):
        tp = int(((preds == class_id) & (labels == class_id)).sum())
        fp = int(((preds == class_id) & (labels != class_id)).sum())
        fn = int(((preds != class_id) & (labels == class_id)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {"accuracy": accuracy, "f1": macro_f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune sentiment model.")
    parser.add_argument(
        "--data",
        default="data/processed/comments_cleaned.csv",
        help="Path to cleaned comments CSV with text/label columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    examples = load_examples(data_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label2id, id2label = build_label_map(examples)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    train_examples, val_examples = split_examples(examples)
    train_dataset = CommentDataset(train_examples, tokenizer, label2id)
    val_dataset = CommentDataset(val_examples, tokenizer, label2id)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
    )

    with mlflow.start_run(run_name="train") as run:
        mlflow.log_param("base_model", MODEL_NAME)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("train_size", len(train_examples))
        mlflow.log_param("val_size", len(val_examples))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()

        mlflow.log_metric("accuracy", float(eval_metrics.get("eval_accuracy", 0.0)))
        mlflow.log_metric("f1", float(eval_metrics.get("eval_f1", 0.0)))

        model_dir = output_dir / "sentiment_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        # Register trained model to MLflow Model Registry (Staging).
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
        )
        mlflow.transformers.log_model(
            transformers_model=sentiment_pipeline,
            artifact_path="model",
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        client = MlflowClient()
        latest_versions = client.get_latest_versions("yt_sentiment_model")
        if not latest_versions:
            raise RuntimeError(
                "Baseline (v1) not found in MLflow registry. "
                "Register the pretrained baseline first."
            )

        result = mlflow.register_model(model_uri, "yt_sentiment_model")
        notes = (
            "Fine-tuned on labeled YouTube comments. "
            "Expected to improve domain relevance versus pretrained baseline."
        )
        client.update_model_version(
            name="yt_sentiment_model",
            version=result.version,
            description=notes,
        )
        client.transition_model_version_stage(
            name="yt_sentiment_model",
            version=result.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        mlflow.log_param("registered_model_version", result.version)

        if str(result.version) == "1":
            raise RuntimeError(
                "Registered version is v1, but baseline already exists. "
                "Check MLflow tracking URI and registry."
            )

    print(f"Model saved to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
