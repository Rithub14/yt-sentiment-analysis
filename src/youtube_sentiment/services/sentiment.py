from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

_DEFAULT_CACHE_DIR = Path(".hf-cache")
_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", _DEFAULT_CACHE_DIR)).resolve()

_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=_CACHE_DIR)
_MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, cache_dir=_CACHE_DIR
)
_MODEL.eval()

_LOG = logging.getLogger("uvicorn.error")

_MLFLOW_EXPERIMENT_NAME = "youtube-sentiment"
_MLFLOW_RUN_ID: str | None = None


def _ensure_mlflow_tracking_uri() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        _LOG.info("MLflow tracking URI set to %s", tracking_uri)
    else:
        _LOG.info("MLflow tracking URI not set; using default local store.")
    _LOG.info("MLflow client tracking URI: %s", mlflow.get_tracking_uri())


def _get_or_create_mlflow_run_id() -> str:
    global _MLFLOW_RUN_ID

    if _MLFLOW_RUN_ID is None:
        _ensure_mlflow_tracking_uri()
        experiment = mlflow.set_experiment(_MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(
            run_name="inference",
            experiment_id=experiment.experiment_id,
        ) as run:
            _MLFLOW_RUN_ID = run.info.run_id
            mlflow.log_param("model_name", MODEL_NAME)

    return _MLFLOW_RUN_ID


def analyze_sentiments(texts: list[str]) -> list[dict[str, Any]]:
    """Analyze sentiment for a batch of texts.

    Returns a list of dicts with keys: "label" and "score".
    """
    _LOG.info("analyze_sentiments called with %d texts", len(texts))
    _ensure_mlflow_tracking_uri()

    if not texts:
        _log_mlflow_metrics(0, 0.0)
        return []

    results: list[dict[str, Any]] = [{} for _ in texts]
    batch_texts: list[str] = []
    batch_indices: list[int] = []

    for idx, text in enumerate(texts):
        if text is None:
            results[idx] = {"label": "NEUTRAL", "score": 0.0}
            continue

        cleaned = str(text).strip()
        if len(cleaned) < 2:
            results[idx] = {"label": "NEUTRAL", "score": 0.0}
            continue

        batch_texts.append(cleaned)
        batch_indices.append(idx)

    if batch_texts:
        with torch.no_grad():
            inputs = _TOKENIZER(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            outputs = _MODEL(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores, preds = probs.max(dim=-1)

        id2label = _MODEL.config.id2label
        for batch_pos, (score, pred) in enumerate(zip(scores, preds)):
            idx = batch_indices[batch_pos]
            label = id2label.get(int(pred), str(int(pred)))
            results[idx] = {"label": label, "score": float(score)}

    average_confidence = 0.0
    if results:
        total_confidence = sum(float(item.get("score", 0.0)) for item in results)
        average_confidence = total_confidence / len(results)

    _log_mlflow_metrics(len(batch_texts), average_confidence)

    return results


def _log_mlflow_metrics(num_texts: int, avg_confidence: float) -> None:
    try:
        run_id = _get_or_create_mlflow_run_id()
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("num_texts_analyzed", num_texts)
            mlflow.log_metric("avg_confidence", avg_confidence)
        _LOG.info(
            "MLflow metrics logged: num_texts_analyzed=%d avg_confidence=%.4f",
            num_texts,
            avg_confidence,
        )
    except Exception:
        _LOG.exception("MLflow logging failed; continuing without tracking.")
