# ─────────────────────────────────────────────────────────────────────────
# ملف جديد كامل — اسمه المقترح: app/services/reel_service.py
# (لو عندك ملف الخدمة الحالي اسمه غير "services/" — مثلًا app/service.py
#  بالمفرد — انقل الدالة أدناه له بدل ملف منفصل، المهم بس المسار يطابق
#  استيراد المسار بملف الراوتر (reels.py) اللي بالأسفل)
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelInteraction
from app.schemas import ReelInteractionIn


async def record_reel_interaction(
    db: AsyncSession,
    payload: ReelInteractionIn,
) -> ReelInteraction:
    """
    تسجيل تفاعل ريل واحد. كتابة فقط — بدون أي تدريب أو معالجة ثقيلة هنا،
    بنفس فلسفة record_interaction() الخاصة بالخزانة (استجابة سريعة للواجهة).

    لا تُستدعى maybe_autotrain من هنا إطلاقًا — التدريب للريلز نموذج عالمي
    (يجمع كل المستخدمين)، فمنطقيًا يُفحص شرط التدريب بمهمة دورية منفصلة
    (cron / scheduled task) لاحقًا بالمرحلة ٢، مو بعد كل تفاعل فردي.
    """
    interaction = ReelInteraction(
        user_id=payload.user_id,
        reel_id=payload.reel_id,
        signal_type=payload.signal_type,
        outfit_style=payload.outfit_style,
        dominant_color=payload.dominant_color,
        watch_seconds=payload.watch_seconds,
        total_seconds=payload.total_seconds,
        content_type=payload.content_type,
        opened_profile_after=payload.opened_profile_after,
        position_in_session=payload.position_in_session,
    )
    db.add(interaction)
    await db.commit()

    return interaction
