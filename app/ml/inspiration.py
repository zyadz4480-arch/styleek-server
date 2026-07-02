"""
app/ml/inspiration.py
يبني كلمات بحث بصرية (إنجليزية) لـ Pexels بناءً على متوسط ميزات القطع
اللي المستخدم قبلها فعلاً (avg_features من service.get_inspiration).

بدل اختيار نمط/مناسبة واحدة بس (max())، بنختار كل الأنماط/المناسبات اللي
تعدّت عتبة معيّنة — عشان نلتقط تنوّع ذوق المستخدم مش بس أقوى إشارة واحدة.
"""
from __future__ import annotations

from app.ml.features import (
    IDX_FORMAL, IDX_CASUAL, IDX_SPORTY,
    IDX_BRIGHTNESS, IDX_SATURATION, IDX_WARM_COLOR,
    IDX_OCC_WORK, IDX_OCC_UNI, IDX_OCC_OUTING, IDX_OCC_SPECIAL, IDX_OCC_SPORT,
)

STYLE_THRESHOLD = 0.50
OCCASION_THRESHOLD = 0.25

_STYLE_WORDS = {
    IDX_FORMAL: ["formal", "tailored"],
    IDX_CASUAL: ["casual", "everyday"],
    IDX_SPORTY: ["sporty", "athleisure"],
}

_OCCASION_WORDS = {
    IDX_OCC_WORK: "office",
    IDX_OCC_UNI: "campus",
    IDX_OCC_OUTING: "streetwear",
    IDX_OCC_SPECIAL: "evening",
    IDX_OCC_SPORT: "activewear",
}

# كلمات فاشون حقيقية بدل أوصاف عامة (bright/dark) — مبنية على سطوع/تشبّع/دفء اللون
PALETTE_WORDS = {
    ("light", "warm"): ["beige", "cream", "camel"],
    ("light", "cool"): ["ivory", "pastel", "powder blue"],
    ("mid", "warm"): ["earth tones", "terracotta", "olive"],
    ("mid", "cool"): ["denim blue", "sage green", "slate"],
    ("dark", "warm"): ["chocolate brown", "rust"],
    ("dark", "cool"): ["charcoal", "monochrome", "navy"],
}


def _brightness_bucket(v: float) -> str:
    if v >= 0.65:
        return "light"
    if v >= 0.35:
        return "mid"
    return "dark"


def build_inspiration_query(avg_features: list[float], gender: str | None = None) -> dict:
    """avg_features: متوسط متجه الميزات (20+ بُعد) للقطع اللي المستخدم قبلها.
    يرجع dict فيه query (جملة بحث واحدة) وkeywords (كلمات منفصلة)، مرتبة
    حسب قوة الإشارة (الأعلى قيمة أولاً) بدل ترتيب ثابت حسب القواميس."""
    scored: list[tuple[float, str]] = []

    # 1) الأنماط — كل اللي فوق العتبة، مع درجة قوة الإشارة نفسها للترتيب
    for idx, words in _STYLE_WORDS.items():
        if avg_features[idx] >= STYLE_THRESHOLD:
            scored.append((avg_features[idx], words[0]))
            # الكلمة الثانوية (tailored/everyday/athleisure) تُستخدم فقط
            # عند إشارة قوية جدًا (>0.75) عشان تزيد التنوع من غير ضجيج
            if len(words) > 1 and avg_features[idx] >= 0.75:
                scored.append((avg_features[idx] - 0.01, words[1]))

    # 2) المناسبات
    for idx, word in _OCCASION_WORDS.items():
        if avg_features[idx] >= OCCASION_THRESHOLD:
            scored.append((avg_features[idx], word))

    # 3) لون النخبة: سطوع + دفء/برودة → كلمة فاشون حقيقية
    brightness = _brightness_bucket(avg_features[IDX_BRIGHTNESS])
    warmth = "warm" if avg_features[IDX_WARM_COLOR] >= 0.5 else "cool"
    palette = PALETTE_WORDS.get((brightness, warmth), [])
    if palette:
        # نستخدم أول كلمتين من نفس الحزمة بدل واحدة بس — تنوّع أعلى
        # لنفس قوة الإشارة (السطوع)، فبنعطيها نفس درجة الأولوية تقريبًا
        color_strength = avg_features[IDX_BRIGHTNESS]
        for i, word in enumerate(palette[:2]):
            scored.append((color_strength - i * 0.01, word))

    seen = set()
    unique_keywords = []
    for _, k in sorted(scored, key=lambda t: t[0], reverse=True):
        if k not in seen:
            seen.add(k)
            unique_keywords.append(k)

    # لو ملوش ألوان مميزة (تشبع منخفض جدًا) نضيف "neutral"
    if avg_features[IDX_SATURATION] < 0.2 and "neutral" not in seen:
        unique_keywords.append("neutral")

    if gender and gender not in seen:
        unique_keywords.append(gender)

    if not unique_keywords:
        unique_keywords = ["minimalist", "everyday"]

    unique_keywords += ["outfit", "fashion"]

    return {
        "query": " ".join(unique_keywords),
        "keywords": unique_keywords,
    }
