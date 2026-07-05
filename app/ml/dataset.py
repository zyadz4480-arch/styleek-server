"""
app/ml/dataset.py

يبني عيّنات تدريب لنموذج Two-Tower (المرحلة 3) من ReelInteraction —
المصدر الحقيقي الوحيد المتاح حاليًا. جدول `interactions` القديم غير
مُستخدَم هنا إطلاقًا: تأكَّد أثناء المراجعة المعمارية أنه بيانات
اختبار/Seed بالكامل (كل الصفوف خلال 5.7 ثانية، user_id واحد اختباري
مختلف تمامًا عن أي مستخدم حقيقي في reel_interactions) — لا تفاعلات
مستخدمين فعلية.

كل عيّنة تدريب: (متجه المستخدم [22], متجه الريل [22], الوزن float)

  - متجه المستخدم = UserStyleProfile.avg_features، عبر
    taste_profile.get_avg_features(). هذا هو "ملف التفضيلات الموحّد"
    المُحدَّث فعليًا بـ EWMA من كل من الريلز والإطلالات — لا حاجة لإعادة
    حسابه هنا، هو جاهز تمامًا لدور "مدخل user tower".

  - متجه الريل = DEFAULT_FEATURE_VECTOR (من app.constants) + تراكب
    الأبعاد المعروفة فعليًا من taste_profile._reel_partial_features
    (بحماية فهرس 0 <= idx < FEATURE_DIM، بنفس نمط cold_start.py).
    مطابق تمامًا لمنطق cold_start_reel_embedding، فقط بدون خطوة
    الإسقاط/التطبيع النهائية — تلك تصير جزءًا من reel tower القابل
    للتدريب في neural.py الخاص بالمرحلة 3، لا مصفوفة ثابتة كما في
    cold_start.py.

  - الوزن = taste_profile._reel_signal_weight(interaction). يُعاد
    استخدامها كما هي حرفيًا، بدون تكرار المنطق — نفس مبدأ
    reel_service.py (الذي أزال نسخة مكررة من هذا المنطق سابقًا كانت
    تحمل نفس الخطأ في تقييد watch_ratio). أي إصلاح مستقبلي على صيغة
    الوزن يحدث في taste_profile.py فقط وينعكس هنا تلقائيًا.

فلسفة الاستبعاد (مؤكَّدة مع المستخدم): صفوف weight <= 0.0 (skip، أو
watch بنسبة مشاهدة ضعيفة جدًا) تُستبعد بالكامل من عيّنات التدريب —
نفس فلسفة taste_profile.py تمامًا ("لا دليل كافٍ لتحديث الملف
الشخصي"). هذه الصفوف لا تُستخدَم كأمثلة سلبية صريحة هنا؛ الأمثلة
السلبية تُبنى أثناء التدريب نفسه عبر in-batch negative sampling
(ريلز أخرى داخل نفس الدفعة تُستخدَم كسلبيات ضمنية لكل مستخدم).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelInteraction
from app.constants import DEFAULT_FEATURE_VECTOR
from app.ml.features import FEATURE_DIM
from app.ml.taste_profile import (
    _reel_partial_features,
    _reel_signal_weight,
    get_avg_features,
)


@dataclass(frozen=True)
class ReelTrainingSample:
    """عيّنة تدريب واحدة جاهزة للـ Two-Tower — قبل التحويل لـ torch.Tensor."""

    user_id: str
    reel_id: str
    user_features: np.ndarray  # shape (FEATURE_DIM,), dtype float32
    reel_features: np.ndarray  # shape (FEATURE_DIM,), dtype float32
    weight: float


def _reel_feature_vector(interaction: ReelInteraction) -> np.ndarray:
    """يبني متجه ميزات 22 بُعد لريل واحد — نفس منطق تركيب المتجه
    المستخدَم في cold_start_reel_embedding بالضبط (DEFAULT_FEATURE_VECTOR
    + تراكب الأبعاد المعروفة)، لكن بدون الإسقاط/التطبيع، لأن هذا المتجه
    الخام هو مدخل reel tower القابل للتدريب لا الناتج النهائي."""
    vector = list(DEFAULT_FEATURE_VECTOR)

    partial = _reel_partial_features(interaction)
    for idx, value in partial.items():
        if 0 <= idx < FEATURE_DIM:
            vector[idx] = value

    return np.asarray(vector, dtype=np.float32)


async def build_reel_training_samples(
    db: AsyncSession,
    *,
    limit: int | None = None,
) -> list[ReelTrainingSample]:
    """يقرأ كل ReelInteraction الحقيقية بترتيب زمني تصاعدي، يستبعد
    weight <= 0.0، ويبني عيّنة تدريب واحدة لكل صف متبقٍ.

    ملاحظة أداء: نجلب متجه كل مستخدم فريد مرة واحدة فقط عبر
    get_avg_features (بدل إعادة القراءة/الإنشاء لكل صف من نفس المستخدم)،
    لأن get_or_create_profile قد تُنشئ صفًا جديدًا لو غير موجود.
    """
    stmt = select(ReelInteraction).order_by(ReelInteraction.created_at.asc())
    if limit is not None:
        stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    interactions = list(result.scalars().all())

    user_ids = {i.user_id for i in interactions}
    user_vectors: dict[str, np.ndarray] = {}
    for uid in user_ids:
        vec = await get_avg_features(db, uid)
        user_vectors[uid] = np.asarray(vec, dtype=np.float32)

    samples: list[ReelTrainingSample] = []
    for interaction in interactions:
        weight = _reel_signal_weight(interaction)
        if weight <= 0.0:
            continue  # نفس فلسفة taste_profile.py — لا دليل كافٍ، نستبعد

        samples.append(
            ReelTrainingSample(
                user_id=interaction.user_id,
                reel_id=interaction.reel_id,
                user_features=user_vectors[interaction.user_id],
                reel_features=_reel_feature_vector(interaction),
                weight=float(weight),
            )
        )

    return samples


class ReelTwoTowerDataset(Dataset):
    """يغلّف قائمة ReelTrainingSample كـ torch.utils.data.Dataset عادي.

    القراءة من القاعدة (async) تحدث مرة واحدة مسبقًا عبر
    build_reel_training_samples() قبل إنشاء هذا الكلاس — PyTorch
    Dataset متزامن (sync) بطبيعته، ولا يدعم await داخل __getitem__.
    """

    def __init__(self, samples: list[ReelTrainingSample]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self._samples[idx]
        return (
            torch.from_numpy(s.user_features),
            torch.from_numpy(s.reel_features),
            torch.tensor(s.weight, dtype=torch.float32),
        )
