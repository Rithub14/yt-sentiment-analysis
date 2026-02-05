from __future__ import annotations

import argparse
import json
from pathlib import Path

from youtube_sentiment.services.sentiment import analyze_sentiments


def _normalize_label(label: str) -> str:
    upper = label.upper()
    if "NEG" in upper:
        return "NEGATIVE"
    if "POS" in upper:
        return "POSITIVE"
    if "NEU" in upper:
        return "NEUTRAL"
    return "NEUTRAL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw comments into CSV.")
    parser.add_argument(
        "--input",
        default="data/raw/comments.json",
        help="Path to raw comments JSON list.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/comments_cleaned.csv",
        help="Path to output CSV with text,label,score columns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    comments: list[str]
    if isinstance(raw, list):
        comments = [str(item) for item in raw]
    elif isinstance(raw, dict) and isinstance(raw.get("comments"), list):
        comments = [str(item) for item in raw.get("comments", [])]
    else:
        raise ValueError("Raw comments must be a list or a dict with 'comments' list.")

    sentiments = analyze_sentiments(comments) if comments else []

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("text,label,score\n")
        for comment, result in zip(comments, sentiments):
            label = _normalize_label(str(result.get("label", "")))
            score = float(result.get("score", 0.0))
            safe_text = str(comment).replace("\"", "\"\"")
            handle.write(f"\"{safe_text}\",{label},{score:.6f}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
