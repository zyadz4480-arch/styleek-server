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
