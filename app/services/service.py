"""
طبقة الخدمة — نسخة محسّنة:
✅ قفل التدريب (training_lock) — يمنع تدريبين متزامنين لنفس المستخدم
✅ حماية النموذج القديم — save_model لا يُستدعى إلا بعد نجاح التدريب
✅ منع التدريب إذا لم تتغير البيانات (موجود مسبقاً في maybe_autotrain)

[مُعدَّل — Phase 0 من خطة الترحيل]: إضافة وحيدة داخل record_interaction()
هي استدعاء log_interaction_v2 (Dual-Write إلى interactions_v2). كل شيء
آخر في هذا الملف مطابق للنسخة السابقة حرفيًا — بلا أي تغيير سلوكي.
"""

from __future__ import annotations

import asyncio
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Interaction, UserModelState
from app.ml.features import extract_features, FEATURE_DIM, IDX_SEASON_MATCH, IDX_WARM_COLOR, IDX_SATURATION
from app.ml.neural import StyleNeuralNet, EXPERT_NAMES
from app.ml.store import load_model, save_model
from app.ml import inspiration
from app.ml import taste_profile
from app.schemas import ItemContext, PredictionOut, MatchOut
from app.services.interactions_v2_log import log_interaction_v2  # [جديد]

# ── قفل التدريب ──────────────────────────────────────────────────────────────
# مفتاح: user_id — قيمة: asyncio.Lock
# يمنع بدء تدريبين متزامنين لنفس المستخدم (نفس المشكلة التي تحدث
# عند استدعاء /train مرتين بسرعة أو تزامن maybe_autotrain مع /train اليدوي)
_training_locks: dict[str, asyncio.Lock] = {}


def _get_lock(user_id: str) -> asyncio.Lock:
    """يُعيد Lock خاص بالمستخدم — يُنشئه إذا لم يكن موجوداً."""
    if user_id not in _training_locks:
        _training_locks[user_id] = asyncio.Lock()
    return _training_locks[user_id]


def _needs_training(state: UserModelState, total: int) -> bool:
    """
    [SINGLE SOURCE OF TRUTH] قرار "هل يحتاج هذا المستخدم تدريباً الآن؟"
    مُستخرَج في دالة واحدة يستخدمها كل من maybe_autotrain() (يُشغِّل
    التدريب فعلياً) و get_performance_summary() (يُبلِّغ العميل عبر
    needs_training بدون تشغيل شيء). أي تعديل مستقبلي على الشرط (رفع/خفض
    25، إضافة قيود جديدة) يكفي أن يحدث هنا مرة واحدة — لا حاجة لتكرار
    المنطق في Flutter أو أي عميل آخر (ويب/iOS) يُبنى لاحقاً.
    """
    return (
        (total - state.sample_count_at_last_train) >= settings.retrain_every
        and total >= 20
    )


# ── دوال مساعدة ──────────────────────────────────────────────────────────────

async def _get_or_create_state(db: AsyncSession, user_id: str) -> UserModelState:
    state = await db.get(UserModelState, user_id)
    if state is None:
        state = UserModelState(
            user_id=user_id,
            expert_logits=[0.0] * len(EXPERT_NAMES),
        )
        db.add(state)
        await db.flush()
    return state


# ── تسجيل تفاعل ──────────────────────────────────────────────────────────────

async def record_interaction(
    db: AsyncSession,
    user_id: str,
    item: ItemContext,
    accepted: bool,
    rating: float | None,
    item_id: str | None,
) -> Interaction:
    """يطابق StyleMLOrchestrator.record() في main.dart."""
    features = extract_features(
        category_name=item.category_name,
        colors=item.colors,
        season_name=item.season_name,
        current_season=item.current_season,
        occasion_name=item.occasion_name,
        temperature=item.temperature,
        wear_count=item.wear_count,
        is_favorite=item.is_favorite,
        last_worn_at=item.last_worn_at,
        brand=item.brand,
        is_layerable=item.is_layerable,
        dna_formal=item.dna_formal,
        dna_casual=item.dna_casual,
    )

    final_rating = rating if rating is not None else (1.0 if accepted else 0.3)
    actual = 1.0 if accepted else 0.0

    await _get_or_create_state(db, user_id)

    interaction = Interaction(
        user_id=user_id,
        features=features,
        label=actual,
        rating=final_rating,
        item_id=item_id,
        category=item.category_name,
        occasion=item.occasion_name,
    )
    db.add(interaction)

    # [جديد — Phase 0 Dual-Write] نفس المعاملة/الـ commit أدناه، لا commit
    # إضافي منفصل. weight هنا = final_rating (نفس القيمة المستخدَمة أصلًا
    # كهدف تدريب rating_head في neural.py) بدل إعادة اختراع صيغة وزن جديدة.
    log_interaction_v2(
        db,
        user_id=user_id,
        event_type="outfit_accepted" if accepted else "outfit_rejected",
        label=actual,
        weight=final_rating,
        item_id=item_id,
        occasion=item.occasion_name,
        season=item.season_name,
        temperature=float(item.temperature) if item.temperature is not None else None,
    )

    await db.commit()

    # يغذّي ملف التفضيلات الموحّد — نفس المتجه اللي تقرأه get_inspiration()
    # وأي منطق اختيار ريلز مستقبلي، بحيث تفضيل قطعة هنا يؤثر على الريلز
    # المقترحة، تمامًا زي ما يؤثر تفاعل ريل على اقتراح الإطلالات.
    await taste_profile.update_from_outfit(db, user_id, features, accepted, final_rating)

    return interaction


# ── التدريب التلقائي ──────────────────────────────────────────────────────────

async def maybe_autotrain(db: AsyncSession, user_id: str) -> bool:
    """
    يُدرِّب تلقائيًا إن تراكم settings.retrain_every عيّنة جديدة منذ آخر تدريب.
    يتجاهل الطلب بصمت إذا كان التدريب جارياً (القفل مشغول).
    """
    lock = _get_lock(user_id)

    # إذا كان التدريب جارياً → تخطّ بدون انتظار
    if lock.locked():
        return False

    state = await _get_or_create_state(db, user_id)
    count_result = await db.execute(
        select(func.count()).select_from(Interaction).where(Interaction.user_id == user_id)
    )
    total = count_result.scalar_one()

    if _needs_training(state, total):
        await train_user_model(db, user_id)
        return True

    return False


# ── التدريب الكامل ────────────────────────────────────────────────────────────

async def train_user_model(db: AsyncSession, user_id: str) -> dict:
    """
    يطابق StyleMLOrchestrator.trainAll() في main.dart.

    التحسينات:
    ① القفل يمنع تدريبين متزامنين — الطلب الثاني يُرجع فوراً بدون تدريب.
    ② save_model لا يُستدعى إلا بعد نجاح التدريب الكامل
       (النموذج القديم يبقى سليماً إذا فشل التدريب الجديد).

    [ملاحظة Phase 0]: هذه الدالة (تدريب نموذج منفصل لكل مستخدم) ستُستبدَل
    بالكامل بـ train_global_model() في مرحلة لاحقة من خطة الترحيل
    (Architecture Freeze v1.0، §6 و§8) — لم تُعدَّل هنا عمدًا، البقاء عليها
    حاليًا هو ما يبقي التطبيق يعمل أثناء بناء المسار الجديد بالتوازي.
    """
    lock = _get_lock(user_id)

    # ① إذا كان التدريب جارياً → ارجع فوراً بدون انتظار
    if lock.locked():
        return {
            "trained": False,
            "reason": "training_already_in_progress",
            "sample_count": 0,
        }

    async with lock:
        # ② جلب البيانات
        rows = (
            await db.execute(
                select(Interaction)
                .where(Interaction.user_id == user_id)
                .order_by(Interaction.created_at)
            )
        ).scalars().all()

        if len(rows) < 20:
            return {
                "trained": False,
                "reason": "insufficient_data",
                "sample_count": len(rows),
            }

        X = np.array([r.features for r in rows], dtype=float)
        y = np.array([r.label for r in rows], dtype=float)
        ratings = np.array([r.rating for r in rows], dtype=float)

        rng = np.random.default_rng()
        idx = rng.permutation(len(rows))
        split = int(len(rows) * 0.8)
        train_idx, test_idx = idx[:split], idx[split:]

        # ③ Warm-start من آخر أوزان محفوظة
        state_check = await _get_or_create_state(db, user_id)
        model = load_model(user_id) if state_check.is_trained else StyleNeuralNet()

        # ④ التدريب — في حالة الفشل، النموذج القديم لا يتأثر (لم نحفظ بعد)
        try:
            model.train(X[train_idx], y[train_idx], ratings[train_idx])
        except Exception as e:
            return {
                "trained": False,
                "reason": f"training_failed: {e}",
                "sample_count": len(rows),
            }

        # ⑤ التقييم
        test_metrics = {}
        if model.is_trained and len(test_idx) > 0:
            try:
                test_metrics = model.evaluate(X[test_idx], y[test_idx])
            except Exception:
                test_metrics = {}

        # ⑥ الحفظ — فقط بعد نجاح التدريب والتقييم
        try:
            save_model(user_id, model)
        except Exception as e:
            return {
                "trained": False,
                "reason": f"save_failed: {e}",
                "sample_count": len(rows),
            }

        # ⑦ تحديث الحالة في قاعدة البيانات
        state = await _get_or_create_state(db, user_id)
        state.is_trained = model.is_trained
        state.sample_count_at_last_train = len(rows)
        state.test_accuracy = test_metrics
        state.last_trained_at = datetime.now(timezone.utc)
        await db.commit()

        return {
            "trained": model.is_trained,
            "sample_count": len(rows),
            "test_metrics": test_metrics,
            "architecture": "neural_net_end_to_end",
        }


# ── التنبؤ ────────────────────────────────────────────────────────────────────

async def predict_for_item(
    db: AsyncSession,
    user_id: str,
    item: ItemContext,
) -> PredictionOut:
    """يطابق StyleMLOrchestrator.predict() في main.dart."""
    model = load_model(user_id)

    features = extract_features(
        category_name=item.category_name,
        colors=item.colors,
        season_name=item.season_name,
        current_season=item.current_season,
        occasion_name=item.occasion_name,
        temperature=item.temperature,
        wear_count=item.wear_count,
        is_favorite=item.is_favorite,
        last_worn_at=item.last_worn_at,
        brand=item.brand,
        is_layerable=item.is_layerable,
        dna_formal=item.dna_formal,
        dna_casual=item.dna_casual,
    )

    result = model.predict_one(np.array(features))

    if result.final_score >= 0.75:
        label, emoji = "مثالية", "🔥"
    elif result.final_score >= 0.55:
        label, emoji = "جيدة", "✅"
    elif result.final_score >= 0.35:
        label, emoji = "مقبولة", "🤔"
    else:
        label, emoji = "غير مناسبة", "❌"

    return PredictionOut(
        logistic_proba=result.logistic_proba,
        linear_rating=result.linear_rating,
        svm_confidence=result.svm_confidence,
        tree_proba=result.tree_proba,
        forest_proba=result.forest_proba,
        forest_agreement=result.forest_agreement,
        final_score=result.final_score,
        is_ml_prediction=result.is_trained,
        label_ar=label,
        emoji=emoji,
    )


# ── هل تناسبني؟ — مطابقة قطعة خارجية مع الخزانة كاملة ────────────────────────

def _match_reason(features: list[float], compatibility: float) -> str:
    """سبب مختصر بالعربية — rule-based بسيط فوق نفس متجه الميزات المُستخرَج
    أصلًا لـ extract_features (لا نموذج إضافي، فقط قراءة فهارس معروفة).
    راجع تعليق فهارس الميزات أعلى app/ml/features.py."""
    season_match = features[IDX_SEASON_MATCH] >= 1.0
    is_warm = features[IDX_WARM_COLOR] >= 1.0
    saturation = features[IDX_SATURATION]

    if compatibility >= 0.75:
        base = "قريبة جدًا من ذوقك المسجَّل في النظام"
    elif compatibility >= 0.55:
        base = "قريبة من ذوقك بشكل عام"
    elif compatibility >= 0.35:
        base = "توافقها مع ذوقك متوسط"
    else:
        base = "بعيدة نوعًا ما عن ذوقك الحالي"

    extra = ""
    if season_match and saturation >= 0.4:
        extra = "، وتتناسق ألوانها مع الموسم الحالي"
    elif season_match:
        extra = "، وتناسب الموسم الحالي"
    elif is_warm:
        extra = "، بدرجة لونية دافئة"

    return base + extra + "."


async def match_item_with_wardrobe(
    db: AsyncSession,
    user_id: str,
    external_item: ItemContext,
    wardrobe: list,  # list[WardrobeItemIn] — تفادي استيراد دائري مع schemas
) -> MatchOut:
    """يطابق matchRemote() في main.dart: القطعة الخارجية + الخزانة كاملة في
    نداء forward() واحد فقط لِـ StyleNeuralNet.predict_batch — بدل N نداء
    HTTP منفصل (predictItemRemote لكل قطعة) كما كان في النسخة المؤقتة.

    الترتيب والاختيار (أعلى 6) يتمّان هنا بالكامل على السيرفر بناءً على
    final_score من نفس النموذج المستخدَم في /style/predict — بلا أي منطق
    قرار في Flutter.
    """

    model = load_model(user_id)

    ext_features = extract_features(
        category_name=external_item.category_name,
        colors=external_item.colors,
        season_name=external_item.season_name,
        current_season=external_item.current_season,
        occasion_name=external_item.occasion_name,
        temperature=external_item.temperature,
        wear_count=external_item.wear_count,
        is_favorite=external_item.is_favorite,
        last_worn_at=external_item.last_worn_at,
        brand=external_item.brand,
        is_layerable=external_item.is_layerable,
        dna_formal=external_item.dna_formal,
        dna_casual=external_item.dna_casual,
    )

    all_features = [ext_features]
    for w in wardrobe:
        all_features.append(extract_features(
            category_name=w.category_name,
            colors=w.colors,
            season_name=w.season_name,
            current_season=w.current_season,
            occasion_name=w.occasion_name,
            temperature=w.temperature,
            wear_count=w.wear_count,
            is_favorite=w.is_favorite,
            last_worn_at=w.last_worn_at,
            brand=w.brand,
            is_layerable=w.is_layerable,
            dna_formal=w.dna_formal,
            dna_casual=w.dna_casual,
        ))

    X = np.array(all_features, dtype=np.float32)
    scores = model.predict_batch(X)  # نداء forward() واحد للجميع معًا

    compatibility = float(scores[0])

    top_ids: list[str] = []
    if wardrobe:
        wardrobe_scores = list(zip([w.id for w in wardrobe], scores[1:]))
        wardrobe_scores.sort(key=lambda t: t[1], reverse=True)
        top_ids = [wid for wid, _ in wardrobe_scores[:6]]

    return MatchOut(
        compatibility=round(max(0.0, min(1.0, compatibility)), 4),
        reason=_match_reason(ext_features, compatibility),
        matches=top_ids,
    )


# ── ملخص الأداء ───────────────────────────────────────────────────────────────

async def get_performance_summary(db: AsyncSession, user_id: str) -> dict:
    state = await _get_or_create_state(db, user_id)

    count_result = await db.execute(
        select(func.count()).select_from(Interaction).where(Interaction.user_id == user_id)
    )
    total = count_result.scalar_one()

    accept_result = await db.execute(
        select(func.count())
        .select_from(Interaction)
        .where(
            Interaction.user_id == user_id,
            Interaction.label == 1.0,
        )
    )
    accepted = accept_result.scalar_one()

    # هل التدريب جارٍ الآن؟
    lock = _get_lock(user_id)

    # [SINGLE SOURCE OF TRUTH] نفس شرط maybe_autotrain بالضبط — العميل
    # (Flutter أو أي واجهة أخرى مستقبلاً) لا يحتاج معرفة الأرقام (20، 25)،
    # فقط يقرأ needs_training وينفّذ. لا تدريب يبدأ من هنا — فقط إبلاغ.
    needs_training = (not lock.locked()) and _needs_training(state, total)

    return {
        "user_id": user_id,
        "sample_count": total,
        "sample_count_at_last_train": state.sample_count_at_last_train,
        "accept_ratio": (accepted / total) if total else 0.5,
        "is_trained": state.is_trained,
        "training_in_progress": lock.locked(),
        "needs_training": needs_training,
        "last_trained_at": state.last_trained_at,
        "architecture": "neural_net_end_to_end",
        "test_metrics": state.test_accuracy or {},
    }


# ── الإلهام البصري ────────────────────────────────────────────────────────────

async def get_inspiration(db: AsyncSession, user_id: str) -> dict:
    """يبني كلمات بحث بصرية من ملف التفضيلات الموحّد — يشمل الآن إشارات
    الريلز (like/save/share/watch) وليس بس القطع المقبولة محليًا، وهذا
    بالضبط ما يخلي تفاعل ريل يؤثر على اقتراح الإطلالات."""
    profile = await taste_profile.get_or_create_profile(db, user_id)
    total_signals = profile.reel_signal_count + profile.outfit_signal_count

    if total_signals < 3:
        return {
            "ready": False,
            "reason": "insufficient_accepted_interactions",
            "sample_count": total_signals,
        }

    avg_features = list(profile.avg_features)

    result = inspiration.build_inspiration_query(avg_features)
    result["ready"] = True
    result["sample_count"] = total_signals

    return result
