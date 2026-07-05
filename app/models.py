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
from pgvector.sqlalchemy import Vector

from app.database import Base
from app.constants import DEFAULT_FEATURE_VECTOR


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
        # مصدرها الوحيد الآن app/constants.py — بدل رقمين ثابتين
        # منسوخين هنا ([0.5]*20 + [9.0, 5.0]، القيمتان الأخيرتان كانتا
        # مطابقتين لـ "other"/"بلا مناسبة" صدفةً، الآن مُشتقّتان فعليًا).
        # list(...) تحافظ على نفس السلوك السابق: نسخة جديدة لكل صف،
        # لا مرجع مشترك قابل للتعديل بالخطأ بين المستخدمين.
        default=lambda: list(DEFAULT_FEATURE_VECTOR),
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


# ===========================
# [جديد — Architecture Freeze v1.0، §2 و§9]
# Embeddings — صف واحد لكل كيان (pgvector، 128 بُعد)
# ===========================
# الجداول أُنشئت فعليًا في القاعدة عبر migration_0002 (نُفِّذ يدويًا
# قبل هذا الكود، بالترتيب الموثَّق في الـ HANDOFF — لا يُعاد ترتيبه).
#
# ⚠️ لا يوجد OutfitEmbedding — القرار المعماري #10: لا كيان "إطلالة"
# حقيقي له id يُرسَل من Flutter. تمثيل الإطلالة الكاملة (لو احتجناه)
# يُحسَب ديناميكيًا (متوسط متجهات القطع)، بدون جدول مستقل.
#
# embedding_version: يسمح بترقية النموذج/الإسقاط لاحقًا بلا كسر (§9) —
# صفوف بنسخة أقدم يمكن ترشيحها لإعادة الحساب دون التأثير على البقية.


class UserEmbedding(Base):
    __tablename__ = "user_embeddings"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    embedding: Mapped[list[float]] = mapped_column(Vector(128))

    embedding_version: Mapped[int] = mapped_column(Integer, default=1)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )


class ReelEmbedding(Base):
    __tablename__ = "reel_embeddings"

    reel_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    embedding: Mapped[list[float]] = mapped_column(Vector(128))

    embedding_version: Mapped[int] = mapped_column(Integer, default=1)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )


class ItemEmbedding(Base):
    __tablename__ = "item_embeddings"

    item_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    embedding: Mapped[list[float]] = mapped_column(Vector(128))

    embedding_version: Mapped[int] = mapped_column(Integer, default=1)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )


# ===========================
# [جديد — المرحلة الثالثة]
# Item — الخصائص الخام لكل قطعة ملابس
# ===========================
# مصدر البيانات الخام اللي يحتاجها dataset.py ليبني متجه الميزات (22 بُعد)
# عبر extract_features() في app/ml/features.py، وبالتالي يدرّب ItemEmbedding
# بالنهج Feature-based (بدل نهج IDs المجرّد).
#
# ⚠️ مُطابَق حرفيًا لكلاس ClothingItem الحقيقي في main.dart (سطر 441-529)
# بعد مراجعة الكود الفعلي — تم تصحيحه من نسخة سابقة كانت تحتوي حقولًا
# افتراضية غير موجودة فعليًا في التطبيق (occasion_name، is_layerable،
# dna_formal، dna_casual). القرار الآن: الجدول يطابق فقط ما يُستخدَم
# فعلًا حاليًا، وأي حقل جديد يُضاف لاحقًا فقط عندما يصبح مستخدَمًا حقًا.
#
# لا يوجد FK صريح على item_id هنا تجاه ItemEmbedding/InteractionV2 —
# بنفس نمط باقي الجداول في هذا الملف (ربط بالقيمة النصية فقط، بدون
# قيد قاعدة بيانات صريح؛ ItemEmbedding.item_id وInteractionV2.item_id
# يتبعان نفس الأسلوب).
#
# ثلاثة معاملات لا تُخزَّن هنا إطلاقًا لأنها ليست خصائص ذاتية للقطعة —
# dataset.py مسؤول عن جلبها من مصدرها الصحيح وقت استدعاء extract_features():
#
#   - is_layerable: محسوبة ديناميكيًا من category_name، وليست عمودًا.
#     القاعدة الفعلية من main.dart (مكررة بنفس الشكل في 6 مواضع):
#     is_layerable = category_name in {"jacket", "hoodie"}
#
#   - dna_formal / dna_casual: ليستا خاصية للقطعة أبدًا — main.dart يجلبهما
#     دائمًا من profile.styleDNA.formal/casual (ملف تفضيلات المستخدم
#     المتفاعل، لا القطعة). لنفس القطعة بالضبط، تختلف هذه القيم حسب
#     المستخدم. يجب أن يجلبهما dataset.py من UserStyleProfile (أو الجدول
#     المعادل لـ StyleDNA) الخاص بالمستخدم صاحب التفاعل وقت البناء.
#
#   - occasion_name: ClothingItem الحقيقية لا تحمل حقل occasion إطلاقًا
#     (تأكيد من التعريف الكامل في main.dart سطر 441-529). المناسبة سياقية
#     بحتة، تأتي من التفاعل نفسه (ReelInteraction.outfit_style أو
#     InteractionV2.occasion)، لا من القطعة.
#
# current_season وtemperature أيضًا غير مخزَّنين هنا لنفس السبب (سياقيان
# بحتان، لحظة الاستخدام/الطلب) — يُمرَّران لـ extract_features من الاستدعاء
# نفسه وقت التدريب/الاستدلال.

class Item(Base):
    __tablename__ = "items"

    item_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)  # مالك القطعة

    category_name: Mapped[str] = mapped_column(String(64))
    colors: Mapped[list] = mapped_column(JSON)  # ["#RRGGBB", ...]
    season_name: Mapped[str] = mapped_column(String(16))  # "all" أو اسم موسم محدد
    brand: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # سلوكي — يتراكم/يتغيّر مع الاستخدام، لكنه يبقى مخزَّنًا على نفس صف القطعة
    wear_count: Mapped[int] = mapped_column(Integer, default=0)
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    last_worn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )
