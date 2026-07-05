from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db  # ← عدّل الاسم هنا لو مختلف عندك
from app.schemas import (
    ReelInteractionIn,
    ReelInteractionOut,
    ReelRerankIn,
    ReelRerankOut,
)
from app.services.reel_service import record_reel_interaction
from app.services.reel_rerank_service import rerank_reel_ids

router = APIRouter(prefix="/reels", tags=["reels"])


@router.post("/interaction", response_model=ReelInteractionOut)
async def post_reel_interaction(
    payload: ReelInteractionIn,
    db: AsyncSession = Depends(get_db),
):
    await record_reel_interaction(db, payload)
    return ReelInteractionOut(status="ok")


@router.post("/rerank", response_model=ReelRerankOut)
async def post_reel_rerank(
    payload: ReelRerankIn,
    db: AsyncSession = Depends(get_db),
):
    reordered = await rerank_reel_ids(db, payload.user_id, payload.reel_ids)
    return ReelRerankOut(reel_ids=reordered)
