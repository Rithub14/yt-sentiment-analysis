from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentiment analysis on input text.")
    parser.add_argument("text", help="Text to analyze")
    return parser.parse_args()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from youtube_sentiment.services.sentiment import analyze_sentiments  # noqa: PLC0415

    args = _parse_args()
    results = analyze_sentiments([args.text])
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
