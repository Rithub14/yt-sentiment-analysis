from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    report_path = Path("reports/drift_report.json")
    if not report_path.exists():
        print("Missing reports/drift_report.json")
        return 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    drift = bool(report.get("drift_detected"))
    distance = report.get("l1_distance")
    threshold = report.get("threshold")

    print(f"drift_detected={drift} l1_distance={distance} threshold={threshold}")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
