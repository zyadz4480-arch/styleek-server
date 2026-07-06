# ─────────────────────────────────────────────────────────────────────────
# app/routers/reels.py
# [مُعدَّل] rerank الآن يستقبل سياقًا خفيفًا لكل ريل (RerankIn.reels: list[
# ReelContextIn]) بدل قائمة reel_id نصية فقط — ضروري لأن cold_start
# يحتاج outfit_style/dominant_color لبناء embedding فوري لريل جديد لا
# يملك واحدًا بعد. راجع app/schemas.py (يحتاج إضافة ReelContextIn وتعديل
# RerankIn — انظر ملف schemas_reel_context_addition.py المرفق منفصلاً).
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db  # ← عدّل الاسم هنا لو مختلف عندك
from app.schemas import (
    ReelInteractionIn,
    ReelInteractionOut,
    RerankIn,
    RerankOut,
)
from app.services.reel_service import (
    record_reel_interaction,
    rerank_reels_by_embedding,
)

router = APIRouter(prefix="/reels", tags=["reels"])


@router.post("/interaction", response_model=ReelInteractionOut)
async def post_reel_interaction(
    payload: ReelInteractionIn,
    db: AsyncSession = Depends(get_db),
):
    await record_reel_interaction(db, payload)
    return ReelInteractionOut(status="ok")


@router.post("/rerank", response_model=RerankOut)
async def rerank_reels(
    payload: RerankIn,
    db: AsyncSession = Depends(get_db),
):
    """
    يستقبل قائمة ريلز (id + سياق خفيف: outfit_style/dominant_color) كما
    رجّعتها Pexels لـ Flutter بالفعل، ويرجعها مُعاد ترتيبها حسب القرب من
    UserEmbedding المدرَّب (Two-Tower، المرحلة 3)، أو حسب embedding
    ابتدائي فوري (cold-start) لأي ريل جديد لا يملك واحدًا مدرَّبًا بعد.

    [مُعدَّل] لم يعد هناك تراجع صامت لريلز بلا embedding — راجع
    rerank_reels_by_embedding في reel_service.py للتفاصيل الكاملة.
    """
    reordered = await rerank_reels_by_embedding(db, payload.user_id, payload.reels)
    return RerankOut(reel_ids=reordered)
