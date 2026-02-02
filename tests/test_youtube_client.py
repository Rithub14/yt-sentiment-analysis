from __future__ import annotations

import logging
import os

import httpx
import pytest

from youtube_sentiment.services import youtube_client

_LOG = logging.getLogger(__name__)


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.com")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> dict:
        return self._payload


def _comment(text: str) -> dict:
    return {
        "snippet": {
            "topLevelComment": {"snippet": {"textDisplay": text}},
        }
    }


def test_fetch_top_level_comments_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")

    payloads = [
        {"items": [_comment("c1"), _comment("c2")], "nextPageToken": "next"},
        {"items": [_comment("c3"), _comment("c4")]},
    ]

    def fake_get(url: str, params: dict, timeout: float) -> DummyResponse:
        return DummyResponse(payloads.pop(0))

    monkeypatch.setattr(youtube_client.httpx, "get", fake_get)

    comments = youtube_client.fetch_top_level_comments("abc123", 3)

    assert comments == ["c1", "c2", "c3"]
    assert payloads == []


def test_fetch_top_level_comments_handles_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")

    def fake_get(url: str, params: dict, timeout: float) -> DummyResponse:
        request = httpx.Request("GET", "https://example.com")
        raise httpx.RequestError("boom", request=request)

    monkeypatch.setattr(youtube_client.httpx, "get", fake_get)

    comments = youtube_client.fetch_top_level_comments("abc123", 10)

    assert comments == []


def test_fetch_top_level_comments_requires_video_id() -> None:
    with pytest.raises(ValueError, match="video_id"):
        youtube_client.fetch_top_level_comments("", 10)


def test_fetch_top_level_comments_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="YOUTUBE_API_KEY"):
        youtube_client.fetch_top_level_comments("abc123", 10)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Set RUN_INTEGRATION_TESTS=1 to enable integration tests.",
)
def test_fetch_top_level_comments_integration() -> None:
    api_key = os.getenv("YOUTUBE_API_KEY")
    video_id = os.getenv("YOUTUBE_VIDEO_ID")
    if not api_key or not video_id:
        pytest.skip("Set YOUTUBE_API_KEY and YOUTUBE_VIDEO_ID to run this test.")

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 1,
        "textFormat": "plainText",
        "key": api_key,
    }
    response = httpx.get(
        "https://www.googleapis.com/youtube/v3/commentThreads",
        params=params,
        timeout=15.0,
    )
    response.raise_for_status()

    comments = youtube_client.fetch_top_level_comments(video_id, 5)

    if comments:
        preview = comments[:5]
        _LOG.info("Fetched %d comments. Preview: %s", len(comments), preview)
    else:
        _LOG.info("Fetched 0 comments.")

    assert isinstance(comments, list)
    assert len(comments) <= 5
    if comments:
        assert isinstance(comments[0], str)
