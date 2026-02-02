from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


def analyze_sentiments(texts: list[str]) -> list[dict[str, Any]]:
    """Analyze sentiment for a batch of texts.

    Returns a list of dicts with keys: "label" and "score".
    """
    if not texts:
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

    return results
