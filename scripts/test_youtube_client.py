from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        os.environ.setdefault(key, value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch top-level comments from a YouTube video."
    )
    parser.add_argument("video_id", nargs="?", help="YouTube video ID")
    parser.add_argument(
        "--max-comments",
        type=int,
        default=None,
        help="Maximum number of comments to fetch",
    )
    return parser.parse_args()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / ".env")

    sys.path.insert(0, str(root / "src"))

    from youtube_sentiment.services.youtube_client import (  # noqa: PLC0415
        fetch_top_level_comments,
    )

    args = _parse_args()
    video_id = args.video_id or os.getenv("YOUTUBE_VIDEO_ID")
    if not video_id:
        print(
            "Missing video_id. Provide it as an argument or set YOUTUBE_VIDEO_ID in .env."
        )
        return 1

    max_comments = args.max_comments
    if max_comments is None:
        max_comments_env = os.getenv("YOUTUBE_MAX_COMMENTS")
        max_comments = int(max_comments_env) if max_comments_env else 20

    comments = fetch_top_level_comments(video_id, max_comments)
    print(f"Fetched {len(comments)} comments.")

    preview_count = min(5, len(comments))
    for idx, text in enumerate(comments[:preview_count], start=1):
        print(f"{idx}. {text}")

    if len(comments) > preview_count:
        print(f"... ({len(comments) - preview_count} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
