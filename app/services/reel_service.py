# ─────────────────────────────────────────────────────────────────────────
# app/services/reel_service.py
# [مُعدَّل] rerank_reels_by_embedding لم تعد مسؤولة عن إنشاء/حفظ
# الـembeddings — هذا انتقل بالكامل لـ app/ml/embedding_repository
# (get_or_create_reel_embedding)، والتي تستدعي بدورها
# app/ml/cold_start.cold_start_reel_embedding عند غياب embedding مخزَّن.
#
# السبب: قبل هذا التعديل، أي ريل جديد وصل لأول مرة من Pexels (لا يملك
# صفًا في ReelInteraction بعد) كان يُرجَع بلا أي إعادة ترتيب فعلية —
# raise عند أول ريل بلا embedding كان يُنهي الدالة بـ "return reel_ids"
# صامتة. الآن كل ريل يحصل على embedding فوري (cold_start) إن لم يوجد
# له واحد، فيصبح قابلاً للترتيب الفعلي من أول ظهور له، لا بعد أول تفاعل
# مستخدم معه فقط.
#
# ⚠️ يتطلب هذا التعديل تحديث RerankIn في app/schemas.py لتحمل سياقًا
# خفيفًا لكل ريل (reel_id + outfit_style + dominant_color)، بدل قائمة
# reel_id نصية فقط كما كانت — راجع ملف schemas_reel_context_addition.py
# المرفق منفصلاً، وrouter المعدَّل app/routers/reels.py.
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelInteraction, UserEmbedding, UserStyleProfile
from app.schemas import ReelInteractionIn, ReelContextIn
from app.ml import taste_profile
from app.ml import cold_start
from app.ml.embedding_repository import get_or_create_reel_embedding
from app.services.interactions_v2_log import log_interaction_v2
from app.ml.automata import service as automata_service
from app.ml.automata.config import Config as AutomataConfig

_AUTOMATA_CFG = AutomataConfig()


def _diversify_cold_start(reels: list[ReelContextIn]) -> list[str]:
    """[جديد] يُستخدَم فقط لصفر دليل ذوق تمامًا (لا UserEmbedding عصبي،
    ولا حتى إشارة واحدة في UserStyleProfile). بدل إرجاع الترتيب الوارد
    كما هو (كان السلوك السابق — فعليًا بلا أي منطق)، نوزّع round-robin
    حسب outfit_style الخاص بكل ريل، فتُعرَض أنماط متنوّعة من أول جلسة
    فعلية بدل تكرار عرضي لنمط واحد (لو دفعة Pexels القادمة متحيّزة
    لنمط معيّن بالصدفة). فائدة إضافية: يُسرّع تعلّم الذوق الحقيقي لأن
    المستخدم يتفاعل مع أنماط متعددة من أول دقيقة، فتمتلئ
    automata/taste_profile بإشارات متنوّعة بدل نمط واحد متكرر."""
    buckets: dict[str, list[ReelContextIn]] = {}
    order: list[str] = []
    for r in reels:
        key = (r.outfit_style or "other").strip().lower()
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(r)

    result: list[str] = []
    while any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                result.append(buckets[k].pop(0).reel_id)
    return result


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
    (بدون تغيير عن النسخة السابقة)
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
    weight = taste_profile._reel_signal_weight(interaction)
    log_interaction_v2(
        db,
        user_id=payload.user_id,
        event_type=payload.signal_type,
        weight=weight,
        reel_id=payload.reel_id,
        watch_time=payload.watch_seconds,
        watch_ratio=_clamped_watch_ratio(payload),
        occasion=None,  # ReelInteractionIn الحالي لا يحمل مناسبة صريحة بعد
    )

    await db.commit()

    await taste_profile.update_from_reel(db, interaction)

    # [جديد — دمج Graph Cellular Automata] Fire-and-forget، بنفس البوابة
    # اللي تستخدمها taste_profile (weight<=0 → "skip"/إشارة ضعيفة → لا
    # نُنشئ/نُغذّي خلايا اهتمام من إشارة غامضة). الـembedding هنا هو
    # ReelEmbedding (128 بُعد) — نفس الفضاء المستخدَم في rerank أدناه،
    # وليس UserStyleProfile.avg_features (22 بُعد، فضاء مختلف تمامًا).
    if weight > 0.0:
        reel_embedding = await get_or_create_reel_embedding(
            db, payload.reel_id, payload.outfit_style, payload.dominant_color
        )
        await automata_service.record_interaction(
            db, payload.user_id, np.asarray(reel_embedding, dtype=np.float64),
            outfit_style=payload.outfit_style, dominant_color=payload.dominant_color,
        )

    return interaction


# ─────────────────────────────────────────────────────────────────────────
# [مُعدَّل] إعادة الترتيب حسب Two-Tower Embeddings — POST /reels/rerank
#
# لا يوجد مخزون ريلز محلي (Flutter يستدعي Pexels مباشرة)، لذلك لا "بحث"
# تقليدي هنا — فقط إعادة ترتيب لقائمة ريلز (مع سياق خفيف لكل واحد) التي
# أرسلتها Flutter بالفعل (نتاج Pexels)، حسب قربها من UserEmbedding
# المدرَّب (app/ml/trainer.py + app/ml/export_embeddings.py) أو، عند غياب
# embedding مدرَّب، حسب قيمة cold-start فورية (app/ml/cold_start.py) تُحسب
# وتُخزَّن تلقائيًا عبر embedding_repository.get_or_create_reel_embedding.
# ─────────────────────────────────────────────────────────────────────────

async def rerank_reels_by_embedding(
    db: AsyncSession,
    user_id: str,
    reels: list[ReelContextIn],
) -> list[str]:
    """
    يعيد ترتيب الريلز حسب قربها من UserEmbedding الخاص بالمستخدم.

    [مُعدَّل] هذه الدالة الآن مسؤولة عن "الترتيب فقط" — ضمان وجود
    ReelEmbedding لكل ريل (قراءة أو cold-start + حفظ) انتقل بالكامل لـ
    get_or_create_reel_embedding. لا فرع "unknown يبقى بترتيبه الأصلي"
    بعد الآن، لأن كل ريل يحصل على embedding فعلي (مدرَّب أو cold-start)
    قبل حساب التشابه — هذا تغيير سلوكي مقصود، انظر ملاحظة أعلى الملف.

    كلا المتجهين (User وReel) مُطبَّعان L2 مسبقًا وقت الإنشاء (قرار معماري
    ثابت للمشروع — راجع cold_start.py وtwo_tower.py) — لذلك الجداء الداخلي
    (dot product) يكافئ تشابه جيب التمام (cosine similarity) مباشرة، بلا
    حاجة لحساب المقام يدويًا.

    [مُعدَّل — Cold-Start الحقيقي] لم يعد هناك فرع "بلا أي تغيير" بلا مبرر.
    UserEmbedding العصبي (128 بُعد) يُصدَّر فقط دفعة عبر export_embeddings.py
    كل retrain_every تفاعل (app/config.py) — بينما UserStyleProfile
    (22 بُعد، taste_profile.py) يتحدّث حيًا مع كل تفاعل واحد. فجوة زمنية
    حقيقية بين "أول تفاعل" و"أول تصدير embedding عصبي" كانت تعني: كل
    مستخدم في هذه الفجوة (وهي الأغلبية وقت الإطلاق) يحصل على ترتيب
    عشوائي فعليًا. الآن:
      - لو عنده أي إشارة ذوق فعلية (reel_signal_count/outfit_signal_count > 0)
        رغم غياب UserEmbedding العصبي: نُسقِط avg_features الحالي عبر نفس
        الإسقاط الخطي الثابت المستخدَم أصلًا لـcold_start_reel_embedding
        (نفس فضاء الـ128 بُعد، مقارنته بمتجهات الريلز العصبية صحيحة رياضيًا
        — راجع القرار المعماري في رأس cold_start.py)، ونكمل بنفس مسار
        automata/semantic أدناه بلا أي فرق.
      - لو صفر دليل تمامًا (أول دقيقة فعلية، لا UserStyleProfile حتى):
        لا يوجد شيء نرتّب عليه بالذوق — بدل ترتيب عشوائي فعليًا، نُنوِّع
        حسب outfit_style (_diversify_cold_start) بدل الاعتماد على ترتيب
        Pexels الوارد كما هو.
    """
    if not reels:
        return []

    user_row = await db.get(UserEmbedding, user_id)

    if user_row is not None:
        user_vec = np.asarray(user_row.embedding, dtype=np.float64)
    else:
        profile = await db.get(UserStyleProfile, user_id)
        has_real_signal = profile is not None and (
            profile.reel_signal_count > 0 or profile.outfit_signal_count > 0
        )
        if not has_real_signal:
            return _diversify_cold_start(reels)
        user_vec = np.asarray(
            cold_start.cold_start_user_embedding(profile.avg_features), dtype=np.float64
        )

    reel_ids: list[str] = []
    candidate_embeddings: list[np.ndarray] = []
    candidate_tags: list[dict[str, float]] = []
    for r in reels:
        embedding = await get_or_create_reel_embedding(
            db, r.reel_id, r.outfit_style, r.dominant_color
        )
        reel_ids.append(r.reel_id)
        candidate_embeddings.append(np.asarray(embedding, dtype=np.float64))
        # [جديد] وسوم الريل نفسه (formal/occasion_work...) — لا تحتاج
        # أي تفاعل سابق للمستخدم مع هذا الريل بالذات، فقط outfit_style/
        # dominant_color الخاصة به.
        candidate_tags.append(
            automata_service.extract_semantic_tags(r.outfit_style, r.dominant_color)
        )

    candidate_matrix = np.stack(candidate_embeddings)  # (n_reels, 128)
    baseline_scores = candidate_matrix @ user_vec  # نفس dot product السابق تمامًا، محسوب دفعة واحدة

    # [جديد — دمج Graph Cellular Automata] راجع integration_guide.md §6.
    # get_automata_boost يرجع None صراحة (لا مصفوفة أصفار) لو المستخدم
    # بلا خلايا حية بعد (cold start) أو فشل تحميل حالته.
    automata_scores = await automata_service.get_automata_boost(db, user_id, candidate_matrix)
    automata_component = (
        automata_scores if automata_scores is not None else np.zeros_like(baseline_scores)
    )

    # [جديد — الطبقة الدلالية] راجع app/ml/automata/service.py
    # get_semantic_boost لشرح الفرق عن "المشي على edges موجودة فعليًا".
    semantic_boost = await automata_service.get_semantic_boost(db, user_id, candidate_tags)

    # ملاحظة: لو المستخدم بلا خلايا إطلاقًا (مستخدم جديد تمامًا)،
    # automata_component وsemantic_boost كلاهما صفر لكل الترشيحات بالتساوي
    # — final_scores تصبح ببساطة baseline_scores × ثابت، والترتيب
    # النسبي لا يتغيّر، فهذا لا يُضعف تجربة المستخدم الجديد إطلاقًا.
    final_scores = (
        _AUTOMATA_CFG.hybrid_embedding_weight * baseline_scores
        + _AUTOMATA_CFG.hybrid_automata_weight * automata_component
        + _AUTOMATA_CFG.hybrid_semantic_weight * semantic_boost
    )

    scored = list(zip(reel_ids, final_scores.tolist()))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [reel_id for reel_id, _ in scored]
