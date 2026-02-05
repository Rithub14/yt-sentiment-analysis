from __future__ import annotations

import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "yt_sentiment_model"


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("MLFLOW_TRACKING_URI is not set.")
        return 1

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        print("No Production model found. Deployment blocked.")
        return 1

    version = versions[0].version
    print(f"Production model found: {MODEL_NAME} v{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
