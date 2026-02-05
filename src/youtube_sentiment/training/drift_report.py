from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict


def _normalize_label(label: str) -> str:
    upper = label.upper()
    if "NEG" in upper:
        return "NEGATIVE"
    if "POS" in upper:
        return "POSITIVE"
    if "NEU" in upper:
        return "NEUTRAL"
    return "NEUTRAL"


def _load_distribution(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    total = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = _normalize_label(str(row.get("label", "")))
            counts[label] += 1
            total += 1

    if total == 0:
        return {k: 0.0 for k in counts}

    return {k: v / total for k, v in counts.items()}


def _l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in {"POSITIVE", "NEGATIVE", "NEUTRAL"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sentiment drift report.")
    parser.add_argument("--current", default="data/processed/comments_cleaned.csv")
    parser.add_argument("--baseline", default="reports/baseline_distribution.json")
    parser.add_argument("--output", default="reports/drift_report.json")
    parser.add_argument("--threshold", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    current_path = Path(args.current)
    baseline_path = Path(args.baseline)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Missing baseline file: {baseline_path}. Generate it once and commit it."
        )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = _load_distribution(current_path)
    distance = _l1_distance(baseline, current)

    report = {
        "baseline": baseline,
        "current": current,
        "l1_distance": distance,
        "threshold": args.threshold,
        "drift_detected": distance > args.threshold,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
