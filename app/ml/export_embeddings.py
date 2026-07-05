"""
app/ml/export_embeddings.py

خطوة النشر (Deployment) — منفصلة تمامًا عن trainer.py كما أوصى
المستخدم. تُشغَّل يدويًا بعد التأكد من نجاح التدريب ووجود
best_model.pt سليم. لو فشلت هذه الخطوة (مثلاً خطأ اتصال بالقاعدة)،
تُعاد فقط هي — لا حاجة لإعادة تدريب النموذج من الصفر.

الخطوات:
  1. تحميل أفضل نموذج مدرَّب (trainer.load_best_model).
  2. جلب كل user_id وreel_id الظاهرين فعليًا في ReelInteraction
     (مصدر البيانات الحقيقي الوحيد حتى الآن — لا item_embeddings هنا
     إطلاقًا، لعدم وجود تفاعلات قطع حقيقية بعد).
  3. بناء متجه ميزات كل مستخدم/ريل بنفس المنطق المستخدم وقت التدريب
     تمامًا (get_avg_features لكل مستخدم، _reel_feature_vector لكل ريل)
     — أي تناقض هنا يعني embedding مُصدَّر لا يطابق ما تعلّمه النموذج فعليًا.
  4. الترميز عبر model.encode_user/encode_reel (النموذج في eval()، بدون تدرّج).
  5. Upsert في user_embeddings/reel_embeddings (get-or-create، نفس نمط
     get_or_create_profile في taste_profile.py).

⚠️ ملاحظة عن الريلز: نفس reel_id قد يظهر في أكثر من صف ReelInteraction
(مستخدمون مختلفون تفاعلوا مع نفس الريل). لا يوجد جدول "Reel" مستقل
يخزّن خصائصه مرة واحدة — outfit_style/dominant_color مخزَّنة داخل كل
صف تفاعل على حدة. لذلك: نبني متجه ميزات كل ريل من **أحدث** صف تفاعل
له (created_at الأكبر)، على افتراض أن آخر بيانات مسجَّلة هي الأدق/الأحدث.
لو تبيّن لاحقًا أن outfit_style/dominant_color لنفس الريل تتضارب بين
الصفوف (خلل بيانات)، هذا الافتراض يحتاج مراجعة عندها فقط.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelInteraction, UserEmbedding, ReelEmbedding
from app.ml.dataset import _reel_feature_vector
from app.ml.taste_profile import get_avg_features
from app.ml.trainer import load_best_model, DEFAULT_MODEL_PATH


@dataclass
class ExportStats:
    users_updated: int
    reels_updated: int
    duration_seconds: float


async def _latest_interaction_per_reel(db: AsyncSession) -> dict[str, ReelInteraction]:
    """يُرجع {reel_id: أحدث صف ReelInteraction له} — انظر الملاحظة أعلى
    الملف بخصوص سبب اعتماد "الأحدث" كمصدر وحيد لخصائص الريل."""
    stmt = select(ReelInteraction).order_by(ReelInteraction.created_at.asc())
    result = await db.execute(stmt)
    latest: dict[str, ReelInteraction] = {}
    for interaction in result.scalars().all():
        latest[interaction.reel_id] = interaction  # الأحدث يستبدل الأقدم دائمًا
    return latest


async def _distinct_user_ids(db: AsyncSession) -> list[str]:
    stmt = select(ReelInteraction.user_id).distinct()
    result = await db.execute(stmt)
    return [row[0] for row in result.all()]


async def _upsert_user_embedding(db: AsyncSession, user_id: str, embedding: list[float]) -> None:
    row = await db.get(UserEmbedding, user_id)
    if row is None:
        row = UserEmbedding(user_id=user_id, embedding=embedding)
        db.add(row)
    else:
        row.embedding = embedding
        row.embedding_version += 1


async def _upsert_reel_embedding(db: AsyncSession, reel_id: str, embedding: list[float]) -> None:
    row = await db.get(ReelEmbedding, reel_id)
    if row is None:
        row = ReelEmbedding(reel_id=reel_id, embedding=embedding)
        db.add(row)
    else:
        row.embedding = embedding
        row.embedding_version += 1


async def export_embeddings(
    db: AsyncSession,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> ExportStats:
    start = time.monotonic()

    model = load_best_model(model_path)  # eval() مُفعَّل مسبقًا داخل load_best_model

    # ── المستخدمون ──────────────────────────────────────────────────
    user_ids = await _distinct_user_ids(db)
    users_updated = 0
    for user_id in user_ids:
        raw_vector = await get_avg_features(db, user_id)
        features = torch.tensor(raw_vector, dtype=torch.float32).unsqueeze(0)  # (1, 22)
        with torch.no_grad():
            embedding = model.encode_user(features).squeeze(0)  # (128,)
        await _upsert_user_embedding(db, user_id, embedding.tolist())
        users_updated += 1

    # ── الريلز ───────────────────────────────────────────────────────
    latest_by_reel = await _latest_interaction_per_reel(db)
    reels_updated = 0
    for reel_id, interaction in latest_by_reel.items():
        raw_vector = _reel_feature_vector(interaction)
        features = torch.tensor(raw_vector, dtype=torch.float32).unsqueeze(0)  # (1, 22)
        with torch.no_grad():
            embedding = model.encode_reel(features).squeeze(0)  # (128,)
        await _upsert_reel_embedding(db, reel_id, embedding.tolist())
        reels_updated += 1

    await db.commit()

    duration = time.monotonic() - start
    return ExportStats(
        users_updated=users_updated,
        reels_updated=reels_updated,
        duration_seconds=duration,
    )
