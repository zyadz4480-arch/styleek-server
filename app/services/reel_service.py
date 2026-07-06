# ─────────────────────────────────────────────────────────────────────────
# app/services/reel_service.py
# [مُعدَّل — إصلاح] أُزيلت الدالة المكرَّرة _signal_weight_for_log التي كانت
# تعيد صياغة نفس منطق الوزن محليًا (وتحمل نفس الخطأ: ratio بلا تقييد،
# سبَّب weight=511.2 في interactions_v2). الآن نستورد الدالة المُصلَحة
# الوحيدة من taste_profile.py بدل تكرارها — أي إصلاح مستقبلي على صيغة
# الوزن يحدث في مكان واحد فقط.
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelInteraction, UserEmbedding, ReelEmbedding
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
# [جديد] إعادة الترتيب حسب Two-Tower Embeddings — POST /reels/rerank
#
# لا يوجد مخزون ريلز محلي (Flutter يستدعي Pexels مباشرة)، لذلك لا "بحث"
# تقليدي هنا — فقط إعادة ترتيب لقائمة reel_id التي أرسلتها Flutter بالفعل
# (نتاج Pexels)، حسب قربها من UserEmbedding المدرَّب (app/ml/trainer.py +
# app/ml/export_embeddings.py).
# ─────────────────────────────────────────────────────────────────────────

async def rerank_reels_by_embedding(
    db: AsyncSession,
    user_id: str,
    reel_ids: list[str],
) -> list[str]:
    """
    يعيد ترتيب reel_ids حسب قربها من UserEmbedding الخاص بالمستخدم.

    كلا المتجهين (User وReel) مُطبَّعان L2 مسبقًا وقت التصدير (قرار معماري
    ثابت للمشروع — راجع cold_start.py وtwo_tower.py) — لذلك الجداء الداخلي
    (dot product) يكافئ تشابه جيب التمام (cosine similarity) مباشرة، بلا
    حاجة لحساب المقام يدويًا.

    تراجع آمن (Graceful degradation) — القائمة تُرجَع دون أي تغيير في حالتين:
      1. لا يوجد UserEmbedding لهذا المستخدم بعد (لم يُدرَّب/يُصدَّر له شيء).
      2. لا يوجد ولا ReelEmbedding واحد ضمن قائمة reel_ids المُرسَلة.

    الريلز التي لها embedding تُرتَّب تنازليًا حسب التشابه وتُوضَع أولاً؛
    الريلز التي لا تاريخ تفاعل لها بعد (لا embedding) تُلحَق بعدها بترتيبها
    النسبي الأصلي كما وصل — لا تُستبعَد، فقط أولوية أقل مؤقتًا لحين تكوين
    تاريخ لها.
    """
    if not reel_ids:
        return []

    user_row = await db.get(UserEmbedding, user_id)
    if user_row is None:
        return reel_ids

    stmt = select(ReelEmbedding).where(ReelEmbedding.reel_id.in_(reel_ids))
    result = await db.execute(stmt)
    reel_embeddings = {row.reel_id: row.embedding for row in result.scalars().all()}

    if not reel_embeddings:
        return reel_ids

    user_vec = np.asarray(user_row.embedding, dtype=np.float64)

    known: list[tuple[str, float]] = []
    unknown: list[str] = []
    for reel_id in reel_ids:
        emb = reel_embeddings.get(reel_id)
        if emb is None:
            unknown.append(reel_id)
            continue
        reel_vec = np.asarray(emb, dtype=np.float64)
        similarity = float(np.dot(user_vec, reel_vec))
        known.append((reel_id, similarity))

    known.sort(key=lambda pair: pair[1], reverse=True)
    return [reel_id for reel_id, _ in known] + unknown
