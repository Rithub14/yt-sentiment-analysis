from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from youtube_sentiment import main


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Set RUN_INTEGRATION_TESTS=1 to enable integration tests.",
)
def test_analyze_endpoint_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(video_id: str, max_comments: int) -> list[str]:
        assert video_id == "abc123"
        assert max_comments == 3
        return ["great", "bad", "meh"]

    def fake_analyze(texts: list[str]) -> list[dict]:
        assert texts == ["great", "bad", "meh"]
        return [
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEGATIVE", "score": 0.8},
            {"label": "NEUTRAL", "score": 0.1},
        ]

    monkeypatch.setattr(main, "fetch_top_level_comments", fake_fetch)
    monkeypatch.setattr(main, "analyze_sentiments", fake_analyze)

    client = TestClient(main.app)
    response = client.post(
        "/analyze", json={"video_id": "abc123", "max_comments": 3}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_comments"] == 3
    assert payload["sentiment_distribution"] == {
        "POSITIVE": 1,
        "NEGATIVE": 1,
        "NEUTRAL": 1,
    }
    assert payload["average_confidence"] == pytest.approx(0.6)
