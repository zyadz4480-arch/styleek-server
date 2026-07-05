# ─────────────────────────────────────────────────────────────────────────
# app/services/reel_service.py
# [مُعدَّل — إصلاح] أُزيلت الدالة المكرَّرة _signal_weight_for_log التي كانت
# تعيد صياغة نفس منطق الوزن محليًا (وتحمل نفس الخطأ: ratio بلا تقييد،
# سبَّب weight=511.2 في interactions_v2). الآن نستورد الدالة المُصلَحة
# الوحيدة من taste_profile.py بدل تكرارها — أي إصلاح مستقبلي على صيغة
# الوزن يحدث في مكان واحد فقط.
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelInteraction, ReelEmbedding, UserEmbedding
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


# ─────────────────────────────────────────────────────────────────────────
# [جديد] POST /reels/rerank
# ─────────────────────────────────────────────────────────────────────────
# القرار المعماري (HANDOFF_PHASE3_v3.md): الباك اند لا يرى أبدًا قائمة
# الريلز الفعلية (تأتي من Pexels مباشرة لـ Flutter) — لذا لا "مخزون" هنا
# يُختار منه، فقط إعادة ترتيب لقائمة جاهزة أرسلها العميل. أي ريل ليس له
# ReelEmbedding مخزَّن (الأغلبية، خصوصًا الريلز الجديدة كليًا) يبقى بترتيبه
# النسبي الأصلي، مُلحَقًا بعد الريلز المُرتَّبة فعليًا — لا كسر، لا استثناء
# غير متوقَّع يوصل لـ Flutter.

async def rerank_reels_for_user(
    db: AsyncSession,
    user_id: str,
    reel_ids: list[str],
) -> list[str]:
    """يعيد ترتيب reel_ids حسب قرب ReelEmbedding من UserEmbedding
    (cosine distance، الأصغر أولًا = الأقرب لذوق المستخدم).
    عند غياب UserEmbedding أو ReelEmbedding: تراجع آمن كامل للترتيب الأصلي.
    """
    if not reel_ids:
        return reel_ids

    user_embedding_row = await db.get(UserEmbedding, user_id)
    if user_embedding_row is None:
        return reel_ids  # لا نعرف ذوق هذا المستخدم بعد — لا تغيير

    user_vector = user_embedding_row.embedding

    # الريلز التي لها embedding فعلي من بين المُرسَلة فقط — مرتَّبة مباشرة
    # عبر ORDER BY في القاعدة (أسرع وأدق من الجلب الكامل ثم الحساب بايثون).
    stmt = (
        select(ReelEmbedding.reel_id)
        .where(ReelEmbedding.reel_id.in_(reel_ids))
        .order_by(ReelEmbedding.embedding.cosine_distance(user_vector))
    )
    result = await db.execute(stmt)
    ranked_ids = list(result.scalars().all())

    if not ranked_ids:
        return reel_ids  # ولا ريل واحد من هذه الدفعة له embedding بعد

    ranked_set = set(ranked_ids)
    # الباقي (بلا embedding) يحافظ على ترتيبه النسبي الأصلي، مُلحَقًا بالنهاية
    remaining = [rid for rid in reel_ids if rid not in ranked_set]

    return ranked_ids + remaining
