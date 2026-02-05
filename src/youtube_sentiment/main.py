from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from typing import Literal

import csv
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from youtube_sentiment.services.sentiment import analyze_sentiments
from youtube_sentiment.services.youtube_client import fetch_top_level_comments

app = FastAPI(title="YouTube Sentiment Backend")

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status_code=response.status_code
    ).inc()
    return response


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class AnalyzeRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    max_comments: int = Field(100, gt=0, le=500)
    save_raw: bool = Field(False, description="Save comments to data/raw/comments.json")
    save_processed: bool = Field(
        False, description="Save labeled comments to data/processed/comments_cleaned.csv"
    )


class AnalyzeResponse(BaseModel):
    total_comments: int
    sentiment_distribution: dict[Literal["POSITIVE", "NEGATIVE", "NEUTRAL"], int]
    average_confidence: float


def _normalize_label(label: str) -> Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]:
    upper = label.upper()
    if "NEG" in upper:
        return "NEGATIVE"
    if "POS" in upper:
        return "POSITIVE"
    if "NEU" in upper:
        return "NEUTRAL"
    return "NEUTRAL"


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        comments = fetch_top_level_comments(request.video_id, request.max_comments)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.save_raw:
        raw_path = Path("data/raw/comments.json")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        existing_comments: list[str] = []
        if raw_path.exists():
            try:
                raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
                if isinstance(raw_data, list):
                    existing_comments = [str(item) for item in raw_data]
                elif isinstance(raw_data, dict) and isinstance(raw_data.get("comments"), list):
                    existing_comments = [str(item) for item in raw_data.get("comments", [])]
            except json.JSONDecodeError:
                existing_comments = []

        combined = existing_comments + [str(comment) for comment in comments]
        max_rows = 500
        if len(combined) > max_rows:
            combined = combined[-max_rows:]

        raw_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2))

    sentiments = analyze_sentiments(comments)

    distribution: dict[Literal["POSITIVE", "NEGATIVE", "NEUTRAL"], int] = {
        "POSITIVE": 0,
        "NEGATIVE": 0,
        "NEUTRAL": 0,
    }
    total_confidence = 0.0

    for result in sentiments:
        label = _normalize_label(str(result.get("label", "NEUTRAL")))
        distribution[label] += 1
        total_confidence += float(result.get("score", 0.0))

    if request.save_processed and comments:
        processed_path = Path("data/processed/comments_cleaned.csv")
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        existing_rows: list[dict[str, str]] = []
        if processed_path.exists():
            with processed_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row.get("text") and row.get("label") is not None:
                        existing_rows.append(row)

        new_rows: list[dict[str, str]] = []
        for comment, result in zip(comments, sentiments):
            label = _normalize_label(str(result.get("label", "NEUTRAL")))
            score = float(result.get("score", 0.0))
            new_rows.append(
                {
                    "text": str(comment),
                    "label": label,
                    "score": f"{score:.6f}",
                }
            )

        max_rows = 500
        combined = existing_rows + new_rows
        if len(combined) > max_rows:
            combined = combined[-max_rows:]

        with processed_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "label", "score"])
            writer.writeheader()
            writer.writerows(combined)

    total_comments = len(sentiments)
    average_confidence = (
        total_confidence / total_comments if total_comments > 0 else 0.0
    )

    return AnalyzeResponse(
        total_comments=total_comments,
        sentiment_distribution=distribution,
        average_confidence=average_confidence,
    )
