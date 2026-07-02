"""
طبقة الخدمة: تجمع بين قاعدة البيانات (Interaction/UserModelState) ومحرك StyleNeuralNet.
يعادل تجميع StyleMLOrchestrator.record() + trainAll() + predict() في مكان واحد،
لكن بشكل غير متزامن (async) ومع تخزين دائم في PostgreSQL.

تغيير مهم عن النسخة القديمة:
  - كان الدمج بين "الخبراء" الخمسة يتم بأوزان يدوية (expert_logits) تُحدَّث
    بقاعدة رياضية مكتوبة يدويًا بعد كل تفاعل (update_expert_logits).
  - الآن الموديل شبكة عصبية واحدة (StyleNeuralNet) تتعلم كل شيء بنفسها
    عبر إعادة تدريب كاملة (backpropagation) — فلا يوجد تحديث وزن يدوي
    لكل تفاعل بمفرده؛ التعلّم يحصل دفعة واحدة في train_user_model عندما
    تتراكم عيّنات كافية (نفس آلية maybe_autotrain القديمة).
"""
from __future__ import annotations
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
from app.schemas import ItemContext, PredictionOut


async def _get_or_create_state(db: AsyncSession, user_id: str) -> UserModelState:
    state = await db.get(UserModelState, user_id)
    if state is None:
        # expert_logits لم يعد يُستخدَم فعليًا في القرار (الشبكة تتعلم الدمج
        # داخليًا)، لكن نُبقي العمود بقيمة افتراضية محايدة للتوافق مع الجدول
        # الحالي في قاعدة البيانات من غير الحاجة لِمigration.
        state = UserModelState(user_id=user_id, expert_logits=[0.0] * len(EXPERT_NAMES))
        db.add(state)
        await db.flush()
    return state


async def record_interaction(
    db: AsyncSession,
    user_id: str,
    item: ItemContext,
    accepted: bool,
    rating: float | None,
    item_id: str | None,
) -> Interaction:
    """يطابق StyleMLOrchestrator.record() في main.dart سطر 4283-4321."""
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

    # ملاحظة: لم يعد هناك تحديث وزن يدوي لكل تفاعل — الشبكة العصبية تتعلم
    # فقط عبر إعادة تدريب كاملة (train_user_model) تُستدعى تلقائيًا من
    # maybe_autotrain عند تراكم عيّنات كافية. هذا أصح إحصائيًا لشبكة عصبية
    # من محاولة تحديثها بعيّنة واحدة في كل مرة.
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

    return interaction


async def maybe_autotrain(db: AsyncSession, user_id: str) -> bool:
    """
    يُدرِّب تلقائيًا إن تراكم settings.retrain_every عيّنة جديدة منذ آخر تدريب.
    يطابق فكرة StyleTrainingScheduler في main.dart (سطر 289-347).
    """
    state = await _get_or_create_state(db, user_id)
    count_result = await db.execute(
        select(func.count()).select_from(Interaction).where(Interaction.user_id == user_id)
    )
    total = count_result.scalar_one()

    if total - state.sample_count_at_last_train >= settings.retrain_every and total >= 20:
        await train_user_model(db, user_id)
        return True
    return False


async def train_user_model(db: AsyncSession, user_id: str) -> dict:
    """يطابق StyleMLOrchestrator.trainAll() في main.dart سطر 4325-4347."""
    rows = (
        await db.execute(
            select(Interaction).where(Interaction.user_id == user_id).order_by(Interaction.created_at)
        )
    ).scalars().all()

    if len(rows) < 20:
        return {"trained": False, "reason": "insufficient_data", "sample_count": len(rows)}

    X = np.array([r.features for r in rows], dtype=float)
    y = np.array([r.label for r in rows], dtype=float)
    ratings = np.array([r.rating for r in rows], dtype=float)

    # Shuffle مختلف في كل تدريب (بدل seed ثابت) — يمنع نفس التقسيم يتكرر دائمًا
    rng = np.random.default_rng()
    idx = rng.permutation(len(rows))
    split = int(len(rows) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    # Warm-start: نكمل من آخر أوزان محفوظة بدل إعادة البدء من الصفر في كل
    # مرة (بدلاً من عشوائية جديدة كاملة) — أسرع وأكثر استقرارًا مع الوقت.
    state_check = await _get_or_create_state(db, user_id)
    model = load_model(user_id) if state_check.is_trained else StyleNeuralNet()
    model.train(X[train_idx], y[train_idx], ratings[train_idx])

    test_metrics = {}
    if model.is_trained and len(test_idx) > 0:
        test_metrics = model.evaluate(X[test_idx], y[test_idx])

    save_model(user_id, model)

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
        "architecture": "neural_net_end_to_end",  # الدمج بين المسارات متعلَّم داخليًا، لا أوزان يدوية
    }


async def predict_for_item(db: AsyncSession, user_id: str, item: ItemContext) -> PredictionOut:
    """يطابق StyleMLOrchestrator.predict() في main.dart سطر 4351-4392."""
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

    # يطابق عتبات main.dart AIConstants (سطر 118-122) و MLPrediction.labelAr/emoji
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


async def get_performance_summary(db: AsyncSession, user_id: str) -> dict:
    state = await _get_or_create_state(db, user_id)
    count_result = await db.execute(
        select(func.count()).select_from(Interaction).where(Interaction.user_id == user_id)
    )
    total = count_result.scalar_one()

    accept_result = await db.execute(
        select(func.count()).select_from(Interaction)
        .where(Interaction.user_id == user_id, Interaction.label == 1.0)
    )
    accepted = accept_result.scalar_one()

    return {
        "user_id": user_id,
        "sample_count": total,
        "accept_ratio": (accepted / total) if total else 0.5,
        "is_trained": state.is_trained,
        "last_trained_at": state.last_trained_at,
        "architecture": "neural_net_end_to_end",
        "test_metrics": state.test_accuracy or {},
    }


# ============================================================
#   جديد: إلهام بصري (كلمات بحث لـ Pexels بناءً على ذوق المستخدم المتعلَّم)
# ============================================================
async def get_inspiration(db: AsyncSession, user_id: str) -> dict:
    """يبني كلمات بحث بصرية بناءً على متوسط ميزات القطع اللي المستخدم قبلها فعلاً."""
    rows = (
        await db.execute(
            select(Interaction).where(Interaction.user_id == user_id, Interaction.label == 1.0)
        )
    ).scalars().all()

    if len(rows) < 3:
        return {"ready": False, "reason": "insufficient_accepted_interactions", "sample_count": len(rows)}

    X = np.array([r.features for r in rows], dtype=float)
    avg_features = X.mean(axis=0).tolist()

    result = inspiration.build_inspiration_query(avg_features)
    result["ready"] = True
    result["sample_count"] = len(rows)
    return result
