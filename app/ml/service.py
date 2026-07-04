"""
طبقة الخدمة — نسخة محسّنة:
✅ قفل التدريب (training_lock) — يمنع تدريبين متزامنين لنفس المستخدم
✅ حماية النموذج القديم — save_model لا يُستدعى إلا بعد نجاح التدريب
✅ منع التدريب إذا لم تتغير البيانات (موجود مسبقاً في maybe_autotrain)
"""

from __future__ import annotations

import asyncio
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Interaction, UserModelState
from app.ml.features import extract_features, FEATURE_DIM
from app.ml.neural import StyleNeuralNet, EXPERT_NAMES
from app.ml.store import load_model, save_model
from app.ml import inspiration
from app.ml import taste_profile
from app.schemas import ItemContext, PredictionOut

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
