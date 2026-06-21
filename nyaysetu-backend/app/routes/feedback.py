"""/feedback — capture a thumbs up/down on an answer.

A trust-focused legal tool needs to know when it gets things wrong. This records the
user's verdict (with the question + cited section for context) so you can catch bad
answers and see what people actually ask. v1 logs a single greppable line — pipe it to
a durable store (HF dataset / DB) later without changing the contract.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

router = APIRouter(tags=["feedback"])
logger = get_logger(__name__)


class FeedbackRequest(BaseModel):
    verdict: Literal["up", "down"]
    query: str = Field("", max_length=2000)
    law_reference: str = Field("", max_length=200)
    language: str = Field("en", max_length=8)
    comment: str = Field("", max_length=1000)


@router.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    logger.info(
        "FEEDBACK verdict=%s lang=%s ref=%r query=%r comment=%r",
        req.verdict,
        req.language,
        req.law_reference[:120],
        req.query[:300],
        req.comment[:300],
    )
    return {"status": "ok"}
