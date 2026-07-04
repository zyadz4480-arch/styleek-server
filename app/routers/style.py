"""
نقاط الـ API الرئيسية — المعادل السيرفري لواجهة StyleMLOrchestrator في main.dart.

POST /style/interactions       → يعادل orchestrator.record(...)
POST /style/predict            → يعادل orchestrator.predict(...)
POST /style/train/{user}       → يعادل orchestrator.trainAll() (تدريب يدوي فوري)
GET  /style/summary/{user}     → يعادل orchestrator.performanceSummary
GET  /style/inspiration/{user} → جديد: كلمات بحث بصرية جاهزة لـ Pexels
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

import numpy as np

from app.database import get_db
from app.security import verify_api_key
from app.schemas import (
    InteractionIn,
    PredictionOut,
    ItemContext,
    TrainResult,
    PerformanceSummary,
    BatchPredictIn,
    BatchPredictOut,
    StyleProfileOut,
)
from app.ml import service, store, taste_profile
from app.ml.features import FEATURE_DIM

router = APIRouter(
    prefix="/style",
    tags=["style-engine"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/interactions", status_code=201)
async def record_interaction(
    payload: InteractionIn,
    db: AsyncSession = Depends(get_db),
):
    await service.record_interaction(
        db,
        payload.user_id,
        payload.item,
        payload.accepted,
        payload.rating,
        payload.item_id,
    )

    auto_trained = await service.maybe_autotrain(db, payload.user_id)

    return {
        "status": "recorded",
        "auto_trained": auto_trained,
    }


@router.post("/predict", response_model=PredictionOut)
async def predict(
    user_id: str,
    item: ItemContext,
    db: AsyncSession = Depends(get_db),
):
    return await service.predict_for_item(db, user_id, item)


@router.post("/predict-batch", response_model=BatchPredictOut)
async def predict_batch(
    user_id: str,
    payload: BatchPredictIn,
):
    """يعادل predictBatchRemote() في main.dart — متجهات ميزات خام جاهزة
    (لا وصف قطعة)، نداء forward() واحد للدفعة كاملة بدل N نداء HTTP منفصل.

    ⚠️ user_id مطلوب هنا كـ query param (نفس نمط /predict)، لكن نسخة
    predictBatchRemote الحالية في Flutter لسه ما بترسله — يلزم تعديلها
    لإضافة queryParameters: {'user_id': userId} قبل ما هذا الـ endpoint يشتغل.
    """
    if not payload.items:
        return BatchPredictOut(scores=[])

    for i, vec in enumerate(payload.items):
        if len(vec) != FEATURE_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"items[{i}]: طول {len(vec)}، متوقَّع {FEATURE_DIM} (FEATURE_DIM)",
            )

    model = store.load_model(user_id)
    X = np.array(payload.items, dtype=np.float32)
    scores = model.predict_batch(X)

    return BatchPredictOut(scores=scores)


@router.post("/train/{user_id}", response_model=TrainResult)
async def train(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await service.train_user_model(db, user_id)

    # إذا كان التدريب جارياً → 409 Conflict
    if result.get("reason") == "training_already_in_progress":
        raise HTTPException(
            status_code=409,
            detail="التدريب جارٍ بالفعل لهذا المستخدم",
        )

    return TrainResult(
        user_id=user_id,
        sample_count=result.get("sample_count", 0),
        trained=result.get("trained", False),
        train_metrics=result.get("train_metrics", {}),
        test_metrics=result.get("test_metrics", {}),
        # expert_weights أُزيل من الشبكة العصبية الموحّدة — يُرجع {} للتوافق
        expert_weights={},
    )


@router.get("/summary/{user_id}", response_model=PerformanceSummary)
async def summary(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    return await service.get_performance_summary(db, user_id)


@router.get("/profile/{user_id}", response_model=StyleProfileOut)
async def get_profile(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """معاينة ملف التفضيلات الموحّد — مفيد للتحقق يدويًا إن تفاعل ريل
    فعلاً حرّك نفس المتجه اللي هيُستخدم لاحقًا في /style/inspiration."""
    profile = await taste_profile.get_or_create_profile(db, user_id)
    return StyleProfileOut(
        user_id=profile.user_id,
        avg_features=list(profile.avg_features),
        reel_signal_count=profile.reel_signal_count,
        outfit_signal_count=profile.outfit_signal_count,
        updated_at=profile.updated_at,
    )


@router.get("/inspiration/{user_id}")
async def inspiration(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """يرجع كلمات بحث إنجليزية جاهزة تُستخدم مباشرة مع Pexels API لجلب صور إلهام."""
    return await service.get_inspiration(db, user_id)
