# ─────────────────────────────────────────────────────────────────────────
# app/services/reel_service.py
# [مُعدَّل — إصلاح] أُزيلت الدالة المكرَّرة _signal_weight_for_log التي كانت
# تعيد صياغة نفس منطق الوزن محليًا (وتحمل نفس الخطأ: ratio بلا تقييد،
# سبَّب weight=511.2 في interactions_v2). الآن نستورد الدالة المُصلَحة
# الوحيدة من taste_profile.py بدل تكرارها — أي إصلاح مستقبلي على صيغة
# الوزن يحدث في مكان واحد فقط.
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelInteraction
from app.schemas import ReelInteractionIn
from app.ml import taste_profile
from app.services.interactions_v2_log import log_interaction_v2


def _clamped_watch_ratio(payload: ReelInteractionIn) -> float | None:
    """نفس منطق التقييد المستخدَم في taste_profile._reel_signal_weight —
    يُحسَب هنا فقط لتعبئة عمود watch_ratio في interactions_v2 (وصفي/تشخيصي)،
    وليس لإعادة اشتقاق weight (ذلك يبقى حصريًا من taste_profile)."""
    if not payload.watch_seconds or not payload.total_seconds:
        return None
    return min(max(payload.watch_seconds / payload.total_seconds, 0.0), 1.0)


async def record_reel_interaction(
    db: AsyncSession,
    payload: ReelInteractionIn,
) -> ReelInteraction:
    """
    تسجيل تفاعل ريل واحد + تحديث ملف التفضيلات الموحّد (UserStyleProfile).
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

    # [Phase 0 Dual-Write] نفس المعاملة/الـ commit أدناه، لا commit إضافي
    # منفصل. weight الآن يأتي من taste_profile._reel_signal_weight نفسها —
    # نفس القيمة بالضبط اللي تُستخدَم لتحديث الملف الشخصي، لا نسخة موازية.
    log_interaction_v2(
        db,
        user_id=payload.user_id,
        event_type=payload.signal_type,
        weight=taste_profile._reel_signal_weight(interaction),
        reel_id=payload.reel_id,
        watch_time=payload.watch_seconds,
        watch_ratio=_clamped_watch_ratio(payload),
        occasion=None,  # ReelInteractionIn الحالي لا يحمل مناسبة صريحة بعد
    )

    await db.commit()

    await taste_profile.update_from_reel(db, interaction)

    return interaction
