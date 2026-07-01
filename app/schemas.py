from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ItemContext(BaseModel):
    """بيانات القطعة + السياق، تُستخدم لبناء متجه الميزات على السيرفر
    (نفس مدخلات MLFeatureExtractor.extract في main.dart سطر 2948)."""
    category_name: str
    colors: list[str] = Field(default_factory=list)
    season_name: str = "all"           # summer | winter | all
    current_season: str = "all"
    occasion_name: Optional[str] = None  # work | university | outing | special | sport
    temperature: Optional[int] = None
    wear_count: int = 0
    is_favorite: bool = False
    last_worn_at: Optional[datetime] = None
    brand: Optional[str] = None
    is_layerable: bool = False
    dna_formal: float = 0.5
    dna_casual: float = 0.5


class InteractionIn(BaseModel):
    user_id: str
    item: ItemContext
    accepted: bool
    rating: Optional[float] = None     # إن لم تُرسَل تُحسب تلقائيًا (0.3 رفض / 1.0 قبول)
    item_id: Optional[str] = None


class PredictionOut(BaseModel):
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
    user_id: str
    sample_count: int
    trained: bool
    train_metrics: dict
    test_metrics: dict
    expert_weights: dict


class PerformanceSummary(BaseModel):
    user_id: str
    sample_count: int
    accept_ratio: float
    is_trained: bool
    last_trained_at: Optional[datetime]
    expert_weights: dict
    test_metrics: dict
