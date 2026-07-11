# ─────────────────────────────────────────────────────────────────────────
# app/routers/automata_internal.py  [جديد]
# يُستدعى فقط من Render Cron Job (خارجي، مرة يوميًا) — لا يواجه Flutter
# إطلاقًا. محمي بـ cron_secret (منفصل عن api_key العام). راجع
# integration_guide.md §5.
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.ml.automata import service as automata_service

logger = logging.getLogger("automata_internal")

router = APIRouter(prefix="/internal/automata", tags=["automata-internal"])


@router.post("/nightly")
async def run_nightly(
    x_cron_secret: str = Header(..., alias="X-Cron-Secret"),
    db: AsyncSession = Depends(get_db),
):
    if x_cron_secret != settings.cron_secret:
        raise HTTPException(status_code=403, detail="forbidden")

    user_ids = await automata_service.users_with_interaction_last_24h(db)
    results = {"ok": 0, "failed": 0, "total": len(user_ids)}

    for uid in user_ids:
        try:
            await automata_service.nightly_cycle_for_user(db, uid)
            results["ok"] += 1
        except Exception:
            # عزل الفشل — مستخدم واحد معطوب لا يوقف الدفعة (integration_guide.md §6)
            logger.exception(f"[automata] nightly failed for user_id={uid}")
            results["failed"] += 1

    return results
