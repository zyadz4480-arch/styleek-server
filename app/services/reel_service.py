# ─────────────────────────────────────────────────────────────────────────
# app/services/reel_service.py
# [مُعدَّل — Phase 0 من خطة الترحيل] إضافة وحيدة: استدعاء log_interaction_v2
# بجانب المسار القديم. لا تغيير على أي سلوك أو قيمة راجعة سابقة.
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelInteraction
from app.schemas import ReelInteractionIn
from app.ml import taste_profile
from app.services.interactions_v2_log import log_interaction_v2  # [جديد]


# نفس منطق الوزن المستخدَم أصلًا في taste_profile._reel_signal_weight —
# نعيد حسابه هنا فقط لتغذية عمود weight في interactions_v2 (Dual-Write)،
# بدون أي تغيير على taste_profile.py نفسه أو على قرار "متى نُحدّث الملف
# الشخصي" (ذلك القرار يبقى بالكامل في taste_profile.py كما هو).
def _signal_weight_for_log(payload: ReelInteractionIn) -> float | None:
    if payload.signal_type == "like":
        return 1.0
    if payload.signal_type == "save":
        return 1.2
    if payload.signal_type == "share":
        return 1.3
    if payload.signal_type == "watch":
        if not payload.watch_seconds or not payload.total_seconds:
            return 0.0
        ratio = payload.watch_seconds / payload.total_seconds
        if ratio >= 0.7:
            return 0.6 * ratio
        if ratio >= 0.3:
            return 0.2 * ratio
        return 0.0
    return 0.0  # skip وأي إشارة غير معروفة


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

    # [جديد — Phase 0 Dual-Write] نفس المعاملة/الـ commit أدناه، لا commit
    # إضافي منفصل. watch_ratio يُحسَب هنا فقط لو توفّر كلا الرقمين.
    watch_ratio = None
    if payload.watch_seconds and payload.total_seconds:
        watch_ratio = payload.watch_seconds / payload.total_seconds

    log_interaction_v2(
        db,
        user_id=payload.user_id,
        event_type=payload.signal_type,
        weight=_signal_weight_for_log(payload),
        reel_id=payload.reel_id,
        watch_time=payload.watch_seconds,
        watch_ratio=watch_ratio,
        occasion=None,  # ReelInteractionIn الحالي لا يحمل مناسبة صريحة بعد
    )

    await db.commit()

    await taste_profile.update_from_reel(db, interaction)

    return interaction
