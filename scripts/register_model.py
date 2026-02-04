from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
REGISTERED_MODEL_NAME = "yt_sentiment_model"
EXPERIMENT_NAME = "youtube-sentiment"
MODEL_DESCRIPTION = "Pretrained baseline model."

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", PROJECT_ROOT / ".hf-cache")).resolve()


def _model_already_registered(client: MlflowClient) -> bool:
    try:
        model = client.get_registered_model(REGISTERED_MODEL_NAME)
        return bool(model.latest_versions)
    except Exception:
        return False


def main() -> int:
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()
    if _model_already_registered(client):
        print(
            "Model already has a registered version. "
            "Refusing to create a new version to keep v1 reserved."
        )
        return 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, cache_dir=HF_CACHE_DIR
    )
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
    )

    with mlflow.start_run(
        run_name="register-model",
        experiment_id=experiment.experiment_id,
    ) as run:
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.transformers.log_model(
            transformers_model=sentiment_pipeline,
            artifact_path="model",
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    version = str(result.version)
    if version != "1":
        print(
            f"Warning: registered model version is {version}, expected v1."
        )

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=MODEL_DESCRIPTION,
    )
    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description=MODEL_DESCRIPTION,
    )
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=False,
    )

    print(
        f"Registered {REGISTERED_MODEL_NAME} v{version} and set stage to Production."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
