from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from youtube_sentiment.services.sentiment import analyze_sentiments
from youtube_sentiment.services.youtube_client import fetch_top_level_comments

app = FastAPI(title="YouTube Sentiment Backend")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


class AnalyzeRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    max_comments: int = Field(100, gt=0, le=500)


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

    total_comments = len(sentiments)
    average_confidence = (
        total_confidence / total_comments if total_comments > 0 else 0.0
    )

    return AnalyzeResponse(
        total_comments=total_comments,
        sentiment_distribution=distribution,
        average_confidence=average_confidence,
    )
