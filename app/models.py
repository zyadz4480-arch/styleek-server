"""
جداول قاعدة البيانات

Interaction  = المعادل السيرفري لِـ MLSample في الكود الأصلي (main.dart سطر 3048)
              لكن هنا نخزّن التفاعلات لكل المستخدمين بشكل دائم بدل آخر 200 عيّنة محليًا فقط.

UserModelState = يخزّن آخر وقت تدريب + أوزان الخبراء المتعلَّمة (softmax logits)
                لكل مستخدم، معادل expertLogits في StyleMLOrchestrator.
"""
from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, DateTime, JSON, Boolean, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Interaction(Base):
    """
    تفاعل واحد لمستخدم: قطعة/إطلالة + السياق + هل قُبلت أم لا.
    يعادل MLSample (features + label + rating) في main.dart لكن مع user_id
    ليصبح التدريب قابلاً للتجميع عبر كل المستخدمين لاحقًا (Collaborative).
    """
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)

    # متجه الميزات الـ 20 بُعدًا — نفس ترتيب FeatureVector في main.dart (سطر 2882-2901)
    features: Mapped[list] = mapped_column(JSON)

    label: Mapped[float] = mapped_column(Float)   # 0.0 رفض / 1.0 قبول
    rating: Mapped[float] = mapped_column(Float)   # 0..1 رضا مُقدَّر

    # بيانات وصفية اختيارية (مفيدة للتحليل والتصحيح لاحقًا)
    item_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    occasion: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("ix_interactions_user_created", "user_id", "created_at"),
    )


class UserModelState(Base):
    """
    حالة النموذج لكل مستخدم: آخر تدريب + أوزان الخبراء الخمسة (softmax).
    يعادل حقلي _lastTrainingTime و _expertLogits في StyleMLOrchestrator (main.dart سطر 4213/4236).
    """
    __tablename__ = "user_model_state"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    # logits خام للخبراء الخمسة: [logistic, linear, svm, tree, forest]
    # القيم الابتدائية تطابق main.dart سطر 4236-4242
    expert_logits: Mapped[list] = mapped_column(JSON, default=lambda: [1.00, 0.82, 0.69, 0.69, 1.20])

    sample_count_at_last_train: Mapped[int] = mapped_column(Integer, default=0)
    is_trained: Mapped[bool] = mapped_column(Boolean, default=False)

    train_accuracy: Mapped[dict] = mapped_column(JSON, default=dict)
    test_accuracy: Mapped[dict] = mapped_column(JSON, default=dict)

    last_trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
