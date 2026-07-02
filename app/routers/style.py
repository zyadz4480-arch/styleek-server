"""
نقاط الـ API الرئيسية — المعادل السيرفري لواجهة StyleMLOrchestrator في main.dart.

POST /style/interactions       → يعادل orchestrator.record(...)
POST /style/predict            → يعادل orchestrator.predict(...)
POST /style/train/{user}       → يعادل orchestrator.trainAll() (تدريب يدوي فوري)
GET  /style/summary/{user}     → يعادل orchestrator.performanceSummary
GET  /style/inspiration/{user} → جديد: كلمات بحث بصرية جاهزة لـ Pexels
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.security import verify_api_key
from app.schemas import InteractionIn, PredictionOut, ItemContext, TrainResult, PerformanceSummary
from app.ml import service

router = APIRouter(prefix="/style", tags=["style-engine"], dependencies=[Depends(verify_api_key)])


@router.post("/interactions", status_code=201)
async def record_interaction(payload: InteractionIn, db: AsyncSession = Depends(get_db)):
    await service.record_interaction(
        db, payload.user_id, payload.item, payload.accepted, payload.rating, payload.item_id
    )
    auto_trained = await service.maybe_autotrain(db, payload.user_id)
    return {"status": "recorded", "auto_trained": auto_trained}


@router.post("/predict", response_model=PredictionOut)
async def predict(user_id: str, item: ItemContext, db: AsyncSession = Depends(get_db)):
    return await service.predict_for_item(db, user_id, item)


@router.post("/train/{user_id}", response_model=TrainResult)
async def train(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await service.train_user_model(db, user_id)
    return TrainResult(
        user_id=user_id,
        sample_count=result.get("sample_count", 0),
        trained=result.get("trained", False),
        train_metrics=result.get("train_metrics", {}),
        test_metrics=result.get("test_metrics", {}),
        expert_weights=result.get("expert_weights", {}),
    )


@router.get("/summary/{user_id}", response_model=PerformanceSummary)
async def summary(user_id: str, db: AsyncSession = Depends(get_db)):
    return await service.get_performance_summary(db, user_id)


@router.get("/inspiration/{user_id}")
async def inspiration(user_id: str, db: AsyncSession = Depends(get_db)):
    """يرجع كلمات بحث إنجليزية جاهزة تُستخدم مباشرة مع Pexels API لجلب صور إلهام."""
    return await service.get_inspiration(db, user_id)
