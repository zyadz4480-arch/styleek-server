"""
جداول قاعدة البيانات
"""

from datetime import datetime, timezone
from sqlalchemy import (
    String,
    Float,
    Integer,
    DateTime,
    JSON,
    Boolean,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)

    features: Mapped[list] = mapped_column(JSON)

    label: Mapped[float] = mapped_column(Float)
    rating: Mapped[float] = mapped_column(Float)

    item_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    occasion: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
    )

    __table_args__ = (
        Index("ix_interactions_user_created", "user_id", "created_at"),
    )


class UserModelState(Base):
    __tablename__ = "user_model_state"

    user_id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
    )

    expert_logits: Mapped[list] = mapped_column(
        JSON,
        default=lambda: [1.00, 0.82, 0.69, 0.69, 1.20],
    )

    sample_count_at_last_train: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    is_trained: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    train_accuracy: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    test_accuracy: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    last_trained_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )


# ===========================
# Reel Interactions
# ===========================

class ReelInteraction(Base):
    __tablename__ = "reel_interactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_id: Mapped[str] = mapped_column(String(128), index=True)
    reel_id: Mapped[str] = mapped_column(String(128), index=True)

    outfit_style: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dominant_color: Mapped[str | None] = mapped_column(String(64), nullable=True)

    signal_type: Mapped[str] = mapped_column(String(16))

    watch_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    content_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    opened_profile_after: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    position_in_session: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
    )

    __table_args__ = (
        Index("ix_reel_user_created", "user_id", "created_at"),
    )


# ===========================
# ملف التفضيلات الموحّد (Unified Taste Profile)
# ===========================
# القلب التقني لفكرة "أي تفاعل بأي مكان يؤثر على كل الاقتراحات":
# الريلز والإطلالات كلاهما يقرأ/يكتب على نفس الصف هنا، بدل نظامين منفصلين.

class UserStyleProfile(Base):
    __tablename__ = "user_style_profile"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    # متوسط متحرك مرجّح (EWMA) لمتجه الميزات — بنفس فضاء 22 بُعد
    # (app/ml/features.py، FEATURE_DIM). القيم الافتراضية = نقطة انطلاق
    # محايدة (0.5 للميزات المستمرة، "other"/"بلا مناسبة" للفئة/المناسبة).
    avg_features: Mapped[list] = mapped_column(
        JSON,
        default=lambda: [0.5] * 20 + [9.0, 5.0],
    )

    # عدد الإشارات (من كل مصدر) اللي ساهمت فعليًا في تحديث المتجه أعلاه —
    # يُستخدم لحساب alpha تكيّفي في EWMA (إشارات أقل → تحديث أسرع/أقوى،
    # إشارات أكثر → استقرار أعلى وتذبذب أقل).
    reel_signal_count: Mapped[int] = mapped_column(Integer, default=0)
    outfit_signal_count: Mapped[int] = mapped_column(Integer, default=0)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )


# ===========================
# [جديد — Architecture Freeze v1.0، §2]
# جدول التفاعلات الموحّد — Training Dataset من اليوم الأول
# ===========================
# Phase 0 من خطة الترحيل: يُضاف بجانب الجداول القديمة (Interaction،
# ReelInteraction) دون حذفها أو تعديل سلوكها. الكتابة هنا تتم بشكل
# "مزدوج" (Dual-Write) من نفس نقاط التسجيل الحالية — انظر
# app/services/interactions_v2_log.py و service.py و reel_service.py.
#
# أعمدة معلَّمة بـ"⚠ V2" موجودة الآن فارغة/غير مُستخدَمة عمدًا (session_id،
# location_type) — إضافتها لاحقًا على جدول فيه ملايين الصفوف أصعب بكثير
# من تركها فارغة الآن. لا تُحذف هذه الأعمدة إن بدت غير مستخدَمة — هذا
# متعمَّد وموثَّق في وثيقة الـ Architecture Freeze.

class InteractionV2(Base):
    __tablename__ = "interactions_v2"

    interaction_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_id: Mapped[str] = mapped_column(String(128), index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)  # ⚠ V2

    reel_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    outfit_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    item_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    event_type: Mapped[str] = mapped_column(String(32))
    # label/weight: نفس منطق الحساب الحالي (accepted→1.0/0.0،
    # _reel_signal_weight) — مُمرَّرة من المتصل، لا تُعاد حسابها هنا.
    label: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)

    watch_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    watch_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    weather: Mapped[str | None] = mapped_column(String(32), nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    season: Mapped[str | None] = mapped_column(String(16), nullable=True)
    occasion: Mapped[str | None] = mapped_column(String(32), nullable=True)
    time_of_day: Mapped[str | None] = mapped_column(String(16), nullable=True)
    location_type: Mapped[str | None] = mapped_column(String(32), nullable=True)  # ⚠ V2 (خصوصية، NULL الآن)

    device: Mapped[str | None] = mapped_column(String(32), nullable=True)
    app_version: Mapped[str | None] = mapped_column(String(16), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        index=True,
    )

    __table_args__ = (
        Index("ix_interactions_v2_user_created", "user_id", "created_at"),
        Index("ix_interactions_v2_session", "session_id"),
        Index("ix_interactions_v2_reel_outfit", "reel_id", "outfit_id"),
    )
