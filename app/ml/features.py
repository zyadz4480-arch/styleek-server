"""
استخراج متجه الميزات — نسخة v2: 20 بُعد هندسي (كما كانت) + 2 مُعرّف فئوي خام
(category_id, occasion_id) يُستخدَمان في neural.py عبر طبقات Embedding متعلَّمة،
بدل الاعتماد فقط على جدول أوزان يدوي (_CAT_WEIGHTS) لتمثيل الفئة.

فهارس الميزات الأصلية (طابق main.dart سطر 2882-2901) — لم تتغيّر:
  0  formal        7  occWork       14 favorite
  1  casual        8  occUni        15 daysSinceWorn
  2  sporty        9  occOuting     16 hasBrand
  3  brightness   10  occSpecial    17 isLayerable
  4  saturation   11  occSport      18 dnaFormal
  5  isWarmColor  12  temperature   19 dnaCasual
  6  seasonMatch  13  wearCount

جديد (v2):
  20  category_id   (0..9  — فهرس فئة الملابس، لِـ nn.Embedding)
  21  occasion_id   (0..5  — فهرس المناسبة، 5 = بلا مناسبة، لِـ nn.Embedding)
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional

FEATURE_DIM = 22  # كانت 20 — أضفنا category_id و occasion_id

IDX_FORMAL, IDX_CASUAL, IDX_SPORTY = 0, 1, 2
IDX_BRIGHTNESS, IDX_SATURATION, IDX_WARM_COLOR = 3, 4, 5
IDX_SEASON_MATCH = 6
IDX_OCC_WORK, IDX_OCC_UNI, IDX_OCC_OUTING, IDX_OCC_SPECIAL, IDX_OCC_SPORT = 7, 8, 9, 10, 11
IDX_TEMPERATURE, IDX_WEAR_COUNT, IDX_FAVORITE, IDX_DAYS_SINCE = 12, 13, 14, 15
IDX_BRAND, IDX_LAYERABLE = 16, 17
IDX_DNA_FORMAL, IDX_DNA_CASUAL = 18, 19
IDX_CATEGORY_ID, IDX_OCCASION_ID = 20, 21

OCCASION_INDICES = [IDX_OCC_WORK, IDX_OCC_UNI, IDX_OCC_OUTING, IDX_OCC_SPECIAL, IDX_OCC_SPORT]
PRACTICAL_INDICES = [IDX_FORMAL, IDX_CASUAL, IDX_SPORTY, IDX_SEASON_MATCH, IDX_TEMPERATURE, IDX_LAYERABLE]
PERSONAL_TASTE_INDICES = [IDX_WEAR_COUNT, IDX_FAVORITE, IDX_DAYS_SINCE, IDX_DNA_FORMAL, IDX_DNA_CASUAL]

# نُبقي جدول الأوزان الهندسي كإشارة مساعدة إضافية (لا ضرر من وجوده)،
# لكن التمثيل "الحقيقي" لفئة الملابس أصبح متعلَّمًا عبر category_id + Embedding
# في neural.py، وليس معتمدًا عليه وحده كما كان سابقًا.
_CAT_WEIGHTS: dict[str, list[float]] = {
    "shirt":     [1.0, 0.3, 0.0, 0.0],
    "tshirt":    [0.1, 0.9, 0.4, 0.0],
    "hoodie":    [0.0, 0.7, 0.5, 1.0],
    "pants":     [0.6, 0.5, 0.2, 0.0],
    "jeans":     [0.2, 1.0, 0.3, 0.0],
    "jacket":    [0.7, 0.4, 0.2, 1.0],
    "shoes":     [0.4, 0.5, 0.6, 0.0],
    "boots":     [0.5, 0.4, 0.1, 0.1],
    "accessory": [0.6, 0.4, 0.1, 0.0],
    "other":     [0.3, 0.5, 0.3, 0.0],
}
_DEFAULT_CAT_WEIGHTS = [0.3, 0.5, 0.3, 0.0]

# ترتيب ثابت لبناء category_id / occasion_id (يجب ألا يتغيّر بعد أول تدريب)
CATEGORY_LIST = ["shirt", "tshirt", "hoodie", "pants", "jeans",
                  "jacket", "shoes", "boots", "accessory", "other"]
OCCASION_LIST = ["work", "university", "outing", "special", "sport"]  # + 5 = بلا مناسبة
NUM_CATEGORIES = len(CATEGORY_LIST)      # 10
NUM_OCCASIONS = len(OCCASION_LIST) + 1   # 6 (يشمل "بلا مناسبة")


def _category_id(category_name: str) -> int:
    try:
        return CATEGORY_LIST.index(category_name)
    except ValueError:
        return CATEGORY_LIST.index("other")


def _occasion_id(occasion_name: Optional[str]) -> int:
    if occasion_name in OCCASION_LIST:
        return OCCASION_LIST.index(occasion_name)
    return len(OCCASION_LIST)  # فهرس "بلا مناسبة"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _hex_to_rgb(hex_color: str) -> Optional[tuple[int, int, int]]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return None
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return None


def extract_features(
    *,
    category_name: str,
    colors: list[str],
    season_name: str,
    current_season: str,
    occasion_name: Optional[str],
    temperature: Optional[int],
    wear_count: int,
    is_favorite: bool,
    last_worn_at: Optional[datetime],
    brand: Optional[str],
    is_layerable: bool,
    dna_formal: float,
    dna_casual: float,
) -> list[float]:
    """يطابق MLFeatureExtractor.extract في main.dart سطر 2948-3026، مع بُعدين إضافيين للـ Embedding."""
    cat_w = _CAT_WEIGHTS.get(category_name, _DEFAULT_CAT_WEIGHTS)

    season_match = 1.0 if (season_name == "all" or season_name == current_season) else 0.0

    brightness, saturation, is_warm = 0.5, 0.5, False
    if colors:
        rgb = _hex_to_rgb(colors[0])
        if rgb is not None:
            r, g, b = rgb
            brightness = (r * 0.299 + g * 0.587 + b * 0.114) / 255
            mx, mn = max(r, g, b) / 255, min(r, g, b) / 255
            saturation = (mx - mn) / mx if mx > 0 else 0.0
            is_warm = r > b

    occ = occasion_name or ""
    occ_work = 1.0 if occ == "work" else 0.0
    occ_uni = 1.0 if occ == "university" else 0.0
    occ_out = 1.0 if occ == "outing" else 0.0
    occ_special = 1.0 if occ == "special" else 0.0
    occ_sport = 1.0 if occ == "sport" else 0.0

    temp_norm = _clamp((temperature - 10) / 35.0, 0.0, 1.0) if temperature is not None else 0.5

    wear_log = math.log(1 + wear_count) / 5.0

    if last_worn_at is not None:
        now = datetime.now(timezone.utc)
        if last_worn_at.tzinfo is None:
            last_worn_at = last_worn_at.replace(tzinfo=timezone.utc)
        days_since = (now - last_worn_at).days
    else:
        days_since = 30
    days_since = _clamp(float(days_since), 0, 365)
    day_log = math.log(1 + days_since) / math.log(366)

    return [
        cat_w[0], cat_w[1], cat_w[2],
        brightness, saturation, 1.0 if is_warm else 0.0,
        season_match,
        occ_work, occ_uni, occ_out, occ_special, occ_sport,
        temp_norm,
        _clamp(wear_log, 0.0, 1.0),
        1.0 if is_favorite else 0.0,
        day_log,
        1.0 if brand else 0.0,
        1.0 if is_layerable else 0.0,
        dna_formal, dna_casual,
        float(_category_id(category_name)),   # جديد: للـ Embedding
        float(_occasion_id(occasion_name)),    # جديد: للـ Embedding
    ]
