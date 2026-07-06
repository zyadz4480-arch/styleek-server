"""
app/ml/taste_profile.py

هذا هو "ملف التفضيلات الموحّد" اللي اتفقنا عليه: صف واحد لكل مستخدم
(UserStyleProfile.avg_features) يقرأه ويكتب فيه كل من:
  - الريلز  (عبر update_from_reel، يُستدعى من reel_service.py)
  - الإطلالات (عبر update_from_outfit، يُستدعى من service.py — انظر الملاحظة
    بالأسفل)

وبالتالي أي تفاعل بأي مكان في التطبيق يؤثر على المتجه نفسه، واللي يُستخدم
لاحقًا في:
  - GET /style/inspiration/{user_id}  (بدل حساب avg_features من جدول
    Interaction فقط — الآن يشمل ذوق الريلز أيضًا)
  - اختيار/ترتيب الريلز القادمة (نفس المتجه، اتجاه معاكس: نستخدمه لتفضيل
    محتوى قريب من الذوق مع تنويع — هذا الجزء يُنفَّذ في طبقة اختيار الريلز،
    مش هنا).

ملاحظة عن التصميم (متعمَّد، وليس نسيانًا):
  - تفاعلات الريلز "جزئية" (outfit_style/dominant_color نص حر، مش قطعة
    موصوفة بالكامل) → بنحدّث بس الأبعاد اللي عندنا فيها دليل فعلي
    (مثلاً formal/casual لو الكلمة موجودة، brightness/saturation/warm لو
    فيه dominant_color)، ونسيب باقي المتجه كما هو.
  - تفاعلات الإطلالات "كاملة" (متجه 22 بُعد جاهز من features.py) → بنحدّث
    كل الأبعاد.
  - "skip" على الريلز حاليًا وزنه = 0 (لا يُحدّث الملف الشخصي) — مش لأنه
    مُهمَل، لكن لأن "تخطّي" إشارة ضعيفة وغامضة (ممكن يكون بسبب الفيديو نفسه
    لا الستايل)؛ يُسجَّل في ReelInteraction للتحليل/التنويع، لكن ما يهزّ
    ذوق المستخدم المتعلَّم. أي تغيير لهذا القرار مستقبلاً يحتاج فقط تعديل
    _reel_signal_weight أدناه.

[مُعدَّل] استخراج الدليل الجزئي من تفاعل ريل انقسم الآن لطبقتين:
  - _partial_features_from_raw(outfit_style, dominant_color): المنطق
    الفعلي، يعمل على قيم خام (نص/hex) بدون أي اعتماد على ORM. هذا يسمح
    باستخدامه في سياقات لا يوجد فيها صف ReelInteraction بعد — تحديدًا
    cold_start_reel_embedding() وقت /reels/rerank لريل وصل لأول مرة ولم
    يتفاعل معه أحد بعد.
  - _reel_partial_features(interaction): تبقى بنفس التوقيع والسلوك
    المستخدَم من update_from_reel، لكنها الآن مجرد غلاف يُفوِّض
    للدالة الخام أعلاه — لا تكرار منطق، ولا تغيير سلوكي على أي استدعاء
    حالي لها.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserStyleProfile, ReelInteraction
from app.ml.features import (
    FEATURE_DIM,
    IDX_FORMAL, IDX_CASUAL, IDX_SPORTY,
    IDX_BRIGHTNESS, IDX_SATURATION, IDX_WARM_COLOR,
    IDX_OCC_WORK, IDX_OCC_UNI, IDX_OCC_OUTING, IDX_OCC_SPECIAL, IDX_OCC_SPORT,
    _hex_to_rgb,  # إعادة استخدام نفس منطق تحويل اللون المستخدم وقت التدريب
)

# ──────────────────────────────────────────────────────────────────────────
# استخراج دليل جزئي من تفاعل ريل
# ──────────────────────────────────────────────────────────────────────────

_STYLE_KEYWORDS: dict[str, int] = {
    "formal": IDX_FORMAL, "tailored": IDX_FORMAL,
    "casual": IDX_CASUAL, "everyday": IDX_CASUAL,
    "sporty": IDX_SPORTY, "athleisure": IDX_SPORTY,
}

_OCCASION_KEYWORDS: dict[str, int] = {
    "work": IDX_OCC_WORK, "office": IDX_OCC_WORK,
    "university": IDX_OCC_UNI, "campus": IDX_OCC_UNI,
    "outing": IDX_OCC_OUTING, "streetwear": IDX_OCC_OUTING,
    "special": IDX_OCC_SPECIAL, "evening": IDX_OCC_SPECIAL,
    "sport": IDX_OCC_SPORT, "activewear": IDX_OCC_SPORT,
}


def _partial_features_from_raw(
    outfit_style: Optional[str],
    dominant_color: Optional[str],
) -> dict[int, float]:
    """[جديد] نفس منطق استخراج الدليل الجزئي، لكن على قيم خام (نص/hex)
    بدل ReelInteraction ORM object. يرجع {index: value} بس للأبعاد اللي
    عندنا دليل فعلي عليها — لا نخمّن باقي المتجه.

    تُستخدَم من مكانين:
      1. _reel_partial_features أدناه (بعد وجود صف ReelInteraction فعلي).
      2. cold_start_reel_embedding في app/ml/cold_start.py (قبل وجود أي
         صف تفاعل — ريل جديد وصل لأول مرة من Pexels وقت /reels/rerank).
    """
    updates: dict[int, float] = {}

    style_text = (outfit_style or "").lower()
    for word, idx in _STYLE_KEYWORDS.items():
        if word in style_text:
            updates[idx] = 1.0
    for word, idx in _OCCASION_KEYWORDS.items():
        if word in style_text:
            updates[idx] = 1.0

    if dominant_color:
        rgb = _hex_to_rgb(dominant_color)
        if rgb is not None:
            r, g, b = rgb
            brightness = (r * 0.299 + g * 0.587 + b * 0.114) / 255
            mx, mn = max(r, g, b) / 255, min(r, g, b) / 255
            saturation = (mx - mn) / mx if mx > 0 else 0.0
            updates[IDX_BRIGHTNESS] = brightness
            updates[IDX_SATURATION] = saturation
            updates[IDX_WARM_COLOR] = 1.0 if r > b else 0.0

    return updates


def _reel_partial_features(interaction: ReelInteraction) -> dict[int, float]:
    """يرجع {index: value} بس للأبعاد اللي عندنا دليل فعلي عليها من هذا
    التفاعل تحديدًا — لا نخمّن باقي المتجه.

    [مُعدَّل] مجرد غلاف حول _partial_features_from_raw — نفس التوقيع
    والسلوك السابقين تمامًا، لا تغيير على أي مستدعٍ حالي (update_from_reel،
    export_embeddings.py، dataset.py)."""
    return _partial_features_from_raw(interaction.outfit_style, interaction.dominant_color)


def _reel_signal_weight(interaction: ReelInteraction) -> float:
    """0.0 = لا يُحدّث الملف الشخصي إطلاقًا (خطوة أمان لإشارات ضعيفة/غامضة)."""
    signal = interaction.signal_type

    if signal == "like":
        return 1.0
    if signal == "save":
        return 1.2
    if signal == "share":
        return 1.3

    if signal == "watch":
        if not interaction.watch_seconds or not interaction.total_seconds:
            return 0.0
        # [إصلاح] تقييد النسبة بين 0 و1 — قيمة total_seconds خاطئة/صغيرة جدًا
        # من العميل (أو watch_seconds أكبر من مدة الفيديو الفعلية بسبب تكرار/
        # buffering) كانت تُنتج ratio > 1 وبالتالي وزنًا غير منطقي (لوحظ فعليًا:
        # weight=511.2 في interactions_v2 بسبب هذا بالضبط). المشاهدة الفعلية
        # لا يمكن منطقيًا أن تتجاوز مدة الفيديو، فنقيّدها هنا كخط دفاع أخير
        # حتى لو أُصلح لاحقًا مصدر الخطأ في العميل.
        ratio = min(max(interaction.watch_seconds / interaction.total_seconds, 0.0), 1.0)
        if ratio >= 0.7:
            return 0.6 * ratio
        if ratio >= 0.3:
            return 0.2 * ratio
        return 0.0  # شاهد جزء بسيط جدًا — لا يكفي كدليل ذوق

    # "skip" وأي إشارة غير معروفة مستقبلاً → بلا تأثير (انظر ملاحظة أعلى الملف)
    return 0.0


# ──────────────────────────────────────────────────────────────────────────
# EWMA — نفس المتجه يتحدّث تدريجيًا من مصدرين مختلفين
# ──────────────────────────────────────────────────────────────────────────

def _alpha_for(signal_count: int, weight: float) -> float:
    """يبدأ سريع التعلّم (أول إشارات) ويستقر تدريجيًا — نفس فلسفة
    manual_signal_decay في neural.py لكن مطبَّقة هنا على متجه بسيط بدل شبكة."""
    base = 4.0 / (signal_count + 5)  # count=0 → 0.8, count=20 → 0.16, count=95 → 0.04
    return max(0.03, min(0.8, base * min(weight, 1.5)))


async def get_or_create_profile(db: AsyncSession, user_id: str) -> UserStyleProfile:
    profile = await db.get(UserStyleProfile, user_id)
    if profile is None:
        profile = UserStyleProfile(user_id=user_id)
        db.add(profile)
        await db.flush()
    return profile


async def update_from_reel(db: AsyncSession, interaction: ReelInteraction) -> None:
    weight = _reel_signal_weight(interaction)
    if weight <= 0.0:
        return  # لا دليل كافٍ — لا نلمس الملف الشخصي (نسجّل التفاعل بس، مسبقًا محفوظ)

    updates = _reel_partial_features(interaction)
    if not updates:
        return  # إشارة إيجابية لكن بلا أي بيانات ستايل/لون نستفيد منها

    profile = await get_or_create_profile(db, interaction.user_id)
    alpha = _alpha_for(profile.reel_signal_count, weight)

    vec = list(profile.avg_features)
    for idx, value in updates.items():
        vec[idx] = vec[idx] * (1 - alpha) + value * alpha

    profile.avg_features = vec
    profile.reel_signal_count += 1
    await db.commit()


async def update_from_outfit(
    db: AsyncSession,
    user_id: str,
    feature_vector: list[float],
    accepted: bool,
    rating: Optional[float] = None,
) -> None:
    """يُستدعى من app/ml/service.py داخل record_interaction() — القطعة
    الوصفية بتوفّر متجه 22 بُعد كامل، فبنحدّث كل الأبعاد (خلافًا للريلز)."""
    if len(feature_vector) != FEATURE_DIM:
        return  # حماية بسيطة — لا نكسر الملف الشخصي بمتجه بحجم غلط

    weight = rating if rating is not None else (1.0 if accepted else 0.2)
    if weight <= 0.0:
        return

    profile = await get_or_create_profile(db, user_id)
    alpha = _alpha_for(profile.outfit_signal_count, weight)

    vec = list(profile.avg_features)
    for i in range(FEATURE_DIM):
        vec[i] = vec[i] * (1 - alpha) + feature_vector[i] * alpha

    profile.avg_features = vec
    profile.outfit_signal_count += 1
    await db.commit()


async def get_avg_features(db: AsyncSession, user_id: str) -> list[float]:
    """نقطة القراءة الموحّدة — تستخدمها /style/inspiration وأي منطق اختيار
    ريلز مستقبلي، بدل ما كل جهة تحسب متوسطها الخاص من جدول مختلف."""
    profile = await get_or_create_profile(db, user_id)
    return list(profile.avg_features)
