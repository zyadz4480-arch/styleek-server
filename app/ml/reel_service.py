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
from app.ml import taste_profile


async def record_reel_interaction(
    db: AsyncSession,
    payload: ReelInteractionIn,
) -> ReelInteraction:
    """
    تسجيل تفاعل ريل واحد + تحديث ملف التفضيلات الموحّد (UserStyleProfile).

    التسجيل نفسه يبقى كتابة سريعة زي قبل. تحديث الملف الشخصي عملية خفيفة
    (EWMA على قائمة أرقام، لا تدريب) فمفيش داعي نأجّلها لمهمة دورية —
    لكنها منفصلة منطقيًا عن maybe_autotrain (تدريب الشبكة العصبية الثقيل)
    اللي يبقى مؤجَّل زي ما كان.
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

    await taste_profile.update_from_reel(db, interaction)

    return interaction
