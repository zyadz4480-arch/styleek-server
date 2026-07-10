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
# Rerank — POST /reels/rerank
# ──────────────────────────────────────────────────────────────────────────────

class ReelContextIn(BaseModel):
    """[جديد] سياق خفيف لريل واحد ضمن قائمة RerankIn.reels.

    ضروري (لا اختياري) لحل مشكلة "الريل الجديد بلا Embedding": عندما
    يصل ريل لأول مرة من Pexels ولا يملك بعد صفًا في ReelInteraction ولا
    ReelEmbedding مُصدَّر من export_embeddings.py، cold_start_reel_embedding
    (app/ml/cold_start.py) تحتاج outfit_style/dominant_color لحساب قيمة
    ابتدائية ذات معنى بدل الرجوع للديفولت المحايد بالكامل.

    نفس البيانات بالضبط متوفرة أصلاً في FashionReel على جهة Flutter
    (outfitStyle, colors من PexelsReelService._parseVideo) — لا حاجة
    لأي حساب إضافي هناك، فقط تمريرها مع كل ريل بدل الـid وحده."""
    reel_id: str
    outfit_style: str | None = None
    dominant_color: str | None = None


class RerankIn(BaseModel):
    """مدخل POST /reels/rerank — قائمة ريلز (id + سياق خفيف) كما رجّعتها
    Pexels لـ Flutter بالفعل (الباك اند لا يخزّن هذه القائمة، Flutter
    يرسلها في كل مرة).

    [مُعدَّل] كانت reel_ids: list[str] — أصبحت الآن reels: list[ReelContextIn]
    لتحمل outfit_style/dominant_color اللازمين لـ cold-start embedding.
    يتطلب هذا تحديثًا مقابلًا في PexelsReelService._applyServerRerank
    على جهة Flutter (main.dart)."""
    user_id: str
    reels: list[ReelContextIn] = Field(default_factory=list)


class RerankOut(BaseModel):
    """مخرج POST /reels/rerank — قائمة reel_id فقط، مُعاد ترتيبها. عند
    عدم وجود UserEmbedding للمستخدم، تُرجَع القائمة بترتيبها الأصلي دون
    تغيير (تراجع آمن) — راجع rerank_reels_by_embedding في reel_service.py."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Match — POST /style/match
# ──────────────────────────────────────────────────────────────────────────────

class WardrobeItemIn(ItemContext):
    """قطعة خزانة ضمن طلب /style/match — نفس ItemContext + هوية القطعة
    (ClothingItem.id في main.dart) حتى نرجع للعميل أي القطع هي الأفضل."""
    id: str


class MatchIn(BaseModel):
    """مدخل POST /style/match — يطابق matchRemote() في main.dart: القطعة
    الخارجية المرشَّحة + خزانة المستخدم كاملة، في نداء واحد بدل N نداء
    لكل قطعة خزانة."""
    external_item: ItemContext
    wardrobe: list[WardrobeItemIn] = Field(default_factory=list)


class MatchOut(BaseModel):
    """مخرج POST /style/match — يطابق ServerMatchResult في main.dart."""
    compatibility: float = Field(ge=0.0, le=1.0)
    reason: str
    matches: list[str] = Field(default_factory=list)  # حتى 6 معرّفات، الأعلى توافقًا
