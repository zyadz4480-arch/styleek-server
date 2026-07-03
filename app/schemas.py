"""
Pydantic schemas — عقود البيانات بين Flutter و FastAPI.
يطابق هياكل البيانات في main.dart (ClothingItem, MLPrediction, إلخ).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# المدخلات
# ──────────────────────────────────────────────────────────────────────────────

class ItemContext(BaseModel):
    """وصف القطعة — يطابق ClothingItem في main.dart."""

    category_name: str
    colors: list[str] = Field(default_factory=list)

    season_name: str = "all"
    current_season: str = "all"
    occasion_name: str | None = None

    temperature: int | None = Field(default=None, ge=-50, le=70)

    wear_count: int = Field(default=0, ge=0)
    is_favorite: bool = False

    last_worn_at: datetime | None = None
    brand: str | None = None
    is_layerable: bool = False

    dna_formal: float = Field(default=0.5, ge=0.0, le=1.0)
    dna_casual: float = Field(default=0.5, ge=0.0, le=1.0)


class InteractionIn(BaseModel):
    """تفاعل مستخدم — يطابق StyleMLOrchestrator.record() في main.dart."""

    user_id: str
    item: ItemContext
    accepted: bool

    rating: float | None = Field(default=None, ge=0.0, le=1.0)
    item_id: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# المخرجات
# ──────────────────────────────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    """نتيجة التنبؤ — يطابق MLPrediction في main.dart."""

    logistic_proba: float
    linear_rating: float
    svm_confidence: float
    tree_proba: float
    forest_proba: float
    forest_agreement: float

    final_score: float
    is_ml_prediction: bool

    label_ar: str
    emoji: str


class TrainResult(BaseModel):
    """نتيجة التدريب — يُرجعها POST /style/train/{user_id}."""

    user_id: str
    sample_count: int
    trained: bool

    train_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)

    # أُبقي للتوافق مع الإصدارات القديمة من العميل (Flutter)
    expert_weights: dict[str, Any] = Field(default_factory=dict)


class PerformanceSummary(BaseModel):
    """
    ملخص أداء النموذج — يُرجعه GET /style/summary/{user_id}.
    """

    user_id: str
    sample_count: int
    # [AUTOTRAIN] عدد العيّنات وقت آخر تدريب ناجح — يُستخدم من جانب Flutter
    # لحساب "كم عيّنة جديدة تراكمت منذ آخر تدريب" وتقرير الحاجة لإعادة
    # التدريب (بدل الاعتماد فقط على is_trained). يُقرأ من UserModelState،
    # وليس من النموذج نفسه (النموذج يحتوي فقط على أوزانه).
    sample_count_at_last_train: int = 0
    accept_ratio: float

    is_trained: bool
    training_in_progress: bool = False
    # [SINGLE SOURCE OF TRUTH] السيرفر وحده يقرر — العميل (Flutter وأي
    # واجهة أخرى مستقبلاً) يقرأ هذا الحقل وينفّذ POST /train بدون معرفة
    # أرقام العتبات (20 عيّنة أولى، 25 عيّنة لإعادة التدريب...). أي تغيير
    # على هذه الأرقام يحدث في service.py فقط، بلا حاجة لتحديث أي عميل.
    needs_training: bool = False

    last_trained_at: datetime | None = None
    architecture: str

    test_metrics: dict[str, Any] = Field(default_factory=dict)
    from pydantic import BaseModel
from typing import Literal


class ReelInteractionIn(BaseModel):
    """مدخل POST /reels/interaction — يطابق حقول ReelInteraction بالضبط."""
    user_id: str
    reel_id: str
    signal_type: Literal["like", "skip", "watch", "share", "save"]

    outfit_style: str | None = None
    dominant_color: str | None = None
    watch_seconds: float | None = None
    total_seconds: float | None = None
    content_type: str | None = None            # image | video
    opened_profile_after: bool | None = None
    position_in_session: int | None = None


class ReelInteractionOut(BaseModel):
    status: str  # "ok"
