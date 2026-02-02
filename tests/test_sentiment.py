from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from youtube_sentiment.services import sentiment


class DummyTokenizer:
    def __init__(self) -> None:
        self.last_texts: list[str] = []

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        self.last_texts = list(texts)
        batch_size = len(texts)
        return {
            "input_ids": torch.zeros((batch_size, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
        }


class DummyModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits
        self.config = SimpleNamespace(id2label={0: "NEGATIVE", 1: "POSITIVE"})

    def __call__(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self._logits)


def test_analyze_sentiments_handles_empty_input() -> None:
    assert sentiment.analyze_sentiments([]) == []


def test_analyze_sentiments_handles_short_texts() -> None:
    results = sentiment.analyze_sentiments(["", " ", None])
    assert results == [
        {"label": "NEUTRAL", "score": 0.0},
        {"label": "NEUTRAL", "score": 0.0},
        {"label": "NEUTRAL", "score": 0.0},
    ]


def test_analyze_sentiments_batches_and_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = DummyTokenizer()
    logits = torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float32)
    model = DummyModel(logits)

    monkeypatch.setattr(sentiment, "_TOKENIZER", tokenizer)
    monkeypatch.setattr(sentiment, "_MODEL", model)

    results = sentiment.analyze_sentiments(["good", "", "bad"])

    assert tokenizer.last_texts == ["good", "bad"]
    assert results[0]["label"] == "POSITIVE"
    assert results[1] == {"label": "NEUTRAL", "score": 0.0}
    assert results[2]["label"] == "NEGATIVE"
    assert 0.0 <= results[0]["score"] <= 1.0
    assert 0.0 <= results[2]["score"] <= 1.0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Set RUN_INTEGRATION_TESTS=1 to enable integration tests.",
)
def test_analyze_sentiments_integration() -> None:
    texts = ["I love this!", "This is terrible."]
    results = sentiment.analyze_sentiments(texts)

    assert len(results) == 2
    for result in results:
        assert isinstance(result["label"], str)
        assert result["label"]
        assert 0.0 <= result["score"] <= 1.0
