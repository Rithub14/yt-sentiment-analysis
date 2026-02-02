from __future__ import annotations

import logging
import os
from typing import List

import httpx

_LOG = logging.getLogger(__name__)

_YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
_MAX_RESULTS_PER_PAGE = 100


def fetch_top_level_comments(video_id: str, max_comments: int) -> List[str]:
    """Fetch top-level comment text for a YouTube video.

    Args:
        video_id: The YouTube video ID.
        max_comments: Maximum number of comments to return.

    Returns:
        A list of comment strings. Returns an empty list if an API error occurs.
    """
    if not video_id:
        raise ValueError("video_id must be a non-empty string.")
    if max_comments <= 0:
        return []

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY environment variable is not set.")

    comments: List[str] = []
    page_token: str | None = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(_MAX_RESULTS_PER_PAGE, max_comments - len(comments)),
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            response = httpx.get(
                f"{_YOUTUBE_API_BASE}/commentThreads", params=params, timeout=15.0
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            _LOG.warning("YouTube API request failed: %s", exc)
            break
        except ValueError as exc:
            _LOG.warning("YouTube API returned invalid JSON: %s", exc)
            break

        items = payload.get("items", [])
        for item in items:
            snippet = (
                item.get("snippet", {})
                .get("topLevelComment", {})
                .get("snippet", {})
            )
            text = snippet.get("textDisplay")
            if text:
                comments.append(text)
            if len(comments) >= max_comments:
                break

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return comments
