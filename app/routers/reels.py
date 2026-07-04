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
from app.schemas import ReelInteractionIn, ReelInteractionOut
from app.services.reel_service import record_reel_interaction

router = APIRouter(prefix="/reels", tags=["reels"])


@router.post("/interaction", response_model=ReelInteractionOut)
async def post_reel_interaction(
    payload: ReelInteractionIn,
    db: AsyncSession = Depends(get_db),
):
    await record_reel_interaction(db, payload)
    return ReelInteractionOut(status="ok")
