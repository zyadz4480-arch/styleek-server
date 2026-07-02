# -*- coding: utf-8 -*-
"""
app/ml/inspiration.py  (نسخة محسّنة)
============================================
يحوّل "بروفايل الذوق" اللي القلب فهمه (متوسط ميزات القطع اللي المستخدم قبلها)
لكلمات بحث إنجليزية جاهزة تُستخدم مباشرة مع Pexels API لجلب صور إلهام.

تحسينات النسخة دي:
  - اختيار أكتر من نمط/مناسبة (مش نمط واحد بس) عن طريق عتبة (threshold).
  - كلمات ألوان مأخوذة من عالم الأزياء (beige, monochrome, earth tones...) بدل
    كلمات عامة (bright/dark) أقل فايدة لمحرك بحث زي Pexels.
  - keywords كلمات مفردة منفصلة بدل عبارات طويلة.
  - دعم اختياري لتحديد الجندر (لو التطبيق بيوفّره لاحقاً) — مش إجباري.
"""
from __future__ import annotations

from app.ml.features import (
    IDX_FORMAL, IDX_CASUAL, IDX_SPORTY,
    IDX_BRIGHTNESS, IDX_WARM_COLOR,
    IDX_OCC_WORK, IDX_OCC_UNI, IDX_OCC_OUTING, IDX_OCC_SPECIAL, IDX_OCC_SPORT,
)

STYLE_THRESHOLD = 0.50       # لو متوسط قيمة النمط >= كده، يُعتبر جزء من ذوق المستخدم
OCCASION_THRESHOLD = 0.25    # لو 25%+ من القطع المقبولة كانت لمناسبة دي، تتضاف

STYLE_WORDS = {
    "formal": ["formal", "elegant"],
    "casual": ["casual", "streetwear"],
    "sporty": ["sporty", "athletic"],
}

OCCASION_WORDS = {
    "work": ["office", "outfit"],
    "university": ["campus", "style"],
    "outing": ["street", "style"],
    "special": ["evening", "outfit"],
    "sport": ["athletic", "outfit"],
}

# كلمات ألوان مأخوذة من عالم الأزياء (أدق لمحركات بحث الصور من bright/dark/warm/cool)
PALETTE_WORDS = {
    ("bright", "warm"): ["beige", "cream", "camel"],
    ("bright", "cool"): ["white", "light gray", "minimalist"],
    ("neutral", "warm"): ["beige", "taupe", "neutral tones"],
    ("neutral", "cool"): ["gray", "neutral tones"],
    ("dark", "warm"): ["brown", "chocolate", "earth tones"],
    ("dark", "cool"): ["black", "charcoal", "monochrome"],
}


def _tone_bucket(brightness: float) -> str:
    if brightness >= 0.6:
        return "bright"
    if brightness <= 0.35:
        return "dark"
    return "neutral"


def build_inspiration_query(avg_features: list[float], gender: str | None = None) -> dict:
    """
    avg_features: متوسط متجه الميزات (20 بُعد، نفس ترتيب features.py)
    للقطع اللي المستخدم قبلها فعلاً (label == 1).
    gender: اختياري ("men" / "women") — لو التطبيق عنده المعلومة دي لاحقاً.
    """
    style_scores = {
        "formal": avg_features[IDX_FORMAL],
        "casual": avg_features[IDX_CASUAL],
        "sporty": avg_features[IDX_SPORTY],
    }
    selected_styles = [k for k, v in style_scores.items() if v >= STYLE_THRESHOLD]
    if not selected_styles:
        selected_styles = [max(style_scores, key=style_scores.get)]

    occ_scores = {
        "work": avg_features[IDX_OCC_WORK],
        "university": avg_features[IDX_OCC_UNI],
        "outing": avg_features[IDX_OCC_OUTING],
        "special": avg_features[IDX_OCC_SPECIAL],
        "sport": avg_features[IDX_OCC_SPORT],
    }
    selected_occasions = [k for k, v in occ_scores.items() if v >= OCCASION_THRESHOLD]
    if not selected_occasions:
        selected_occasions = [max(occ_scores, key=occ_scores.get)]

    brightness = avg_features[IDX_BRIGHTNESS]
    is_warm = avg_features[IDX_WARM_COLOR] >= 0.5
    tone_key = (_tone_bucket(brightness), "warm" if is_warm else "cool")
    palette_words = PALETTE_WORDS[tone_key]

    # تجميع كل الكلمات (بدون تكرار، بالترتيب)
    words: list[str] = []
    if gender in ("men", "women"):
        words.append(gender)
    for s in selected_styles:
        words.extend(STYLE_WORDS[s])
    for o in selected_occasions:
        words.extend(OCCASION_WORDS[o])
    words.extend(palette_words)
    words.append("minimalist fashion")

    seen = set()
    keywords = []
    for w in words:
        if w not in seen:
            seen.add(w)
            keywords.append(w)

    query = " ".join(keywords)

    return {
        "query": query,
        "keywords": keywords,
        "dominant_styles": selected_styles,
        "dominant_occasions": selected_occasions,
    }
