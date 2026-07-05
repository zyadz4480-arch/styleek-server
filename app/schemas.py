"""
Pydantic schemas — عقود البيانات بين Flutter و FastAPI.
يطابق هياكل البيانات في main.dart (ClothingItem, MLPrediction, إلخ).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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
    sample_count_at_last_train: int = 0
    accept_ratio: float

    is_trained: bool
    training_in_progress: bool = False
    needs_training: bool = False

    last_trained_at: datetime | None = None
    architecture: str

    test_metrics: dict[str, Any] = Field(default_factory=dict)


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


# ──────────────────────────────────────────────────────────────────────────────
# Reel Rerank — POST /reels/rerank
# ──────────────────────────────────────────────────────────────────────────────

class ReelRerankIn(BaseModel):
    """مدخل POST /reels/rerank — قائمة reel_id بترتيبها الحالي (بعد الترتيب
    المحلي في Flutter)، ليعيد السيرفر ترتيبها حسب قرب كل ريل من ذوق المستخدم
    (ReelEmbedding مقابل UserEmbedding)."""
    user_id: str
    reel_ids: list[str]


class ReelRerankOut(BaseModel):
    """نفس reel_ids المُدخلة، بترتيب جديد. الريلز التي لا تملك embedding
    مخزَّن تبقى في مكانها الأصلي بالضبط (تراجع آمن تلقائي)."""
    reel_ids: list[str]


# ──────────────────────────────────────────────────────────────────────────────
# Batch prediction — /style/predict-batch
# ──────────────────────────────────────────────────────────────────────────────

class BatchPredictIn(BaseModel):
    """مدخل POST /style/predict-batch — متجهات ميزات خام جاهزة (لا وصف قطعة).
    يطابق FeatureVector.values في main.dart: كل عنصر داخل items يجب أن يكون
    بطول FEATURE_DIM بالضبط (app/ml/features.py)."""
    items: list[list[float]]


class BatchPredictOut(BaseModel):
    scores: list[float]  # final_score لكل عنصر، بنفس ترتيب items المُدخلة


class StyleProfileOut(BaseModel):
    """مخرج GET /style/profile/{user_id} — لمعاينة/تشخيص ملف التفضيلات
    الموحّد مباشرة (هل الريلز فعلاً تحدّث نفس المتجه اللي تقرأه الإطلالات؟)."""
    user_id: str
    avg_features: list[float]
    reel_signal_count: int
    outfit_signal_count: int
    updated_at: datetime | None = None
