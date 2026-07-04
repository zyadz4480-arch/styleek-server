"""
app/routers/debug.py  [مؤقت]

⚠️ endpoint تشخيصي مؤقت فقط — لعرض آخر صفوف interactions_v2 من المتصفح
مباشرة، بديل عن Render Shell غير المتاح في الخطة المجانية.

يُحذف هذا الملف بعد التأكد من أن الكتابة المزدوجة تعمل (لا يبقى في الإنتاج
كـ endpoint دائم بدون حماية/مصادقة).
"""
from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models import InteractionV2

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/interactions-v2")
async def latest_interactions_v2(limit: int = 5):
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(InteractionV2)
                .order_by(InteractionV2.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()

        return [
            {
                "interaction_id": r.interaction_id,
                "user_id": r.user_id,
                "event_type": r.event_type,
                "weight": r.weight,
                "reel_id": r.reel_id,
                "outfit_id": r.outfit_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
