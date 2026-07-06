# ─────────────────────────────────────────────────────────────────────────
# ملف جديد كامل — اسمه المقترح: app/routers/reels.py
#
# ⚠️ ملاحظة مهمة: راوتر style.py عندك أكيد فيه dependency لجلب الـ
#    AsyncSession (شي زي get_db). ما شفت محتوى style.py عشان أطابق اسمها
#    بالضبط — لو الاسم عندك مختلف عن "get_db" بـ app/database.py،
#    غيّر سطر الاستيراد والاستخدام أدناه فقط (سطرين).
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
    يستقبل قائمة reel_id كما رجّعتها Pexels لـ Flutter بالفعل، ويرجعها
    مُعاد ترتيبها حسب القرب من UserEmbedding المدرَّب (Two-Tower، المرحلة 3).
    تراجع آمن تلقائي للترتيب الأصلي عند غياب أي embedding ذي صلة —
    راجع rerank_reels_by_embedding في reel_service.py للتفاصيل الكاملة.
    """
    reordered = await rerank_reels_by_embedding(db, payload.user_id, payload.reel_ids)
    return RerankOut(reel_ids=reordered)
