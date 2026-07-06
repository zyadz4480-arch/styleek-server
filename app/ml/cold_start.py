"""
Cold Start — تهيئة أول قيمة لكل Embedding (User / Reel / Item)
================================================================

المشكلة: مستخدم/قطعة/ريل جديد ما عنده تفاعلات كافية بعد يتعلّم منها
النموذج المشترك. بدون قيمة ابتدائية، ما نقدر نحسب أي توصية له من
اليوم الأول.

الحل: إسقاط خطي ثابت من فضاء الميزات الحالي (FEATURE_DIM=22، من
app/ml/features.py) إلى فضاء الـ Embedding المشترك (128 بُعد)،
ببذرة عشوائية ثابتة لا تتغيّر أبدًا بعد أول استخدام حقيقي.

[مُعدَّل — رُبطت الآن بمسار حيّ]: cold_start_reel_embedding() تُستدعى
الآن فعليًا من app/ml/embedding_repository.get_or_create_reel_embedding،
واللي بدورها تُستدعى من reel_service.rerank_reels_by_embedding عند
POST /reels/rerank لأي ريل وصل لأول مرة من Pexels ولا يملك بعد صفًا في
ReelInteraction (وبالتالي لا ReelEmbedding مُصدَّر له من export_embeddings.py).
هذا يحل مشكلة "الريل الجديد بلا embedding" (Cold-Start البنيوي) — قبل
هذا التعديل كانت الدالة موجودة لكن غير مربوطة بأي endpoint أو service.

مصادر حقيقية مربوطة هنا (لا افتراضات):
  - extract_features            → app/ml/features.py  (متجه القطعة الكامل)
  - taste_profile._partial_features_from_raw → app/ml/taste_profile.py
    (دليل جزئي عن الريل، من قيم خام — لا يتطلب صف ReelInteraction)
  - CATEGORY_LIST / OCCASION_LIST → app/ml/features.py  (لبناء الديفولت المحايد)

قرار معماري ثابت — L2-Normalization (لا يقتصر على هذا الملف):
  كل embedding يُخزَّن في user_embeddings/reel_embeddings/item_embeddings
  يجب أن يكون unit-normalized (طول = 1)، بما في ذلك مخرجات النموذج
  المشترك لاحقًا (الخطوة 3) وأي تحديث حيّ (الخطوة 4). هذا يضمن أن مقارنة
  أي كيانين (بغضّ النظر إن كان أحدهما لسه "بارد" من هذا الملف أو مُدرَّب
  فعليًا) صحيحة رياضيًا عبر أي عملية مسافة في pgvector تُستخدم لاحقًا —
  cosine `<=>` أو L2 `<->` يعطيان نفس ترتيب النتائج على متجهات وحدة الطول
  (inner product `<#>` وحده يبقى حسّاسًا لو أي مصدر مستقبلي كسر هذا الالتزام،
  فتجنّبه ما لم يُراجَع صراحة). لا تُبنى فهارس pgvector بعد (مؤجَّلة لحين
  تعبئة بيانات حقيقية)، لكن أي فهرس مستقبلي (ivfflat/HNSW) يجب اختيار
  opclass له بما يتوافق مع هذا الالتزام، لا العكس.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.ml.features import FEATURE_DIM, extract_features
from app.ml import taste_profile
from app.constants import DEFAULT_FEATURE_VECTOR

# =====================================================================
# ثوابت — لا تُغيَّر بعد أول استخدام حقيقي
# =====================================================================

EMBEDDING_DIM = 128
_PROJECTION_SEED = 1337

# DEFAULT_FEATURE_VECTOR مستورد أعلاه من app/constants.py — مصدر وحيد
# يشاركه أيضًا UserStyleProfile.avg_features في models.py. لا تُعرَّف
# هنا محليًا؛ أي تعديل على الديفولت يكون في app/constants.py فقط.


def _build_projection_matrix() -> np.ndarray:
    """
    يبني مصفوفة إسقاط ثابتة (FEATURE_DIM x EMBEDDING_DIM) عبر بذرة
    عشوائية مثبَّتة، بحيث نفس المدخل ينتج دائمًا نفس المتجه.
    """
    rng = np.random.RandomState(_PROJECTION_SEED)
    return rng.normal(loc=0.0, scale=1.0, size=(FEATURE_DIM, EMBEDDING_DIM))


# تُبنى مرة واحدة عند استيراد الوحدة، وتبقى ثابتة طوال عمر العملية.
_PROJECTION_MATRIX = _build_projection_matrix()


def _project(features: list[float]) -> list[float]:
    """
    يطبّق الإسقاط الخطي الثابت 22 → 128، ثم L2-normalize (قرار معماري
    ثابت — انظر رأس الملف).
    """
    if len(features) != FEATURE_DIM:
        raise ValueError(
            f"متجه الميزات يجب أن يكون بطول {FEATURE_DIM}، استُلم {len(features)}"
        )

    vector = np.asarray(features, dtype=np.float64)
    embedding = vector @ _PROJECTION_MATRIX

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()


# =====================================================================
# نقاط الدخول العامة — واحدة لكل نوع كيان
# =====================================================================


def cold_start_user_embedding(avg_features: list[float]) -> list[float]:
    """
    Embedding ابتدائي لمستخدم جديد، من UserStyleProfile.avg_features
    الحالي له (وهو بالفعل متجه كامل 22 بُعد — لا حاجة لتعويض قيم ناقصة).
    """
    return _project(avg_features)


def cold_start_item_embedding(feature_vector: list[float]) -> list[float]:
    """
    Embedding ابتدائي لقطعة ملابس، من متجه ميزات جاهز (ناتج
    extract_features الحقيقي من app/ml/features.py، مُستدعى من طبقة
    إنشاء القطعة). هذه الدالة لا تعيد استدعاء extract_features بنفسها
    لتفادي تكرار منطق تحميل خصائص القطعة من قاعدة البيانات — استخدم
    cold_start_item_embedding_from_attributes أدناه لو تفضّل تمرير
    الخصائص الخام مباشرة.
    """
    return _project(feature_vector)


def cold_start_item_embedding_from_attributes(**extract_features_kwargs) -> list[float]:
    """
    اختصار يستدعي extract_features الحقيقي مباشرة بنفس معاملاته
    (category_name, colors, season_name, ...)، لمن يفضّل تمرير خصائص
    القطعة الخام بدل حساب المتجه بنفسه أولًا.
    """
    feature_vector = extract_features(**extract_features_kwargs)
    return cold_start_item_embedding(feature_vector)


def cold_start_reel_embedding(
    outfit_style: Optional[str],
    dominant_color: Optional[str],
) -> list[float]:
    """
    [مُعدَّل] Embedding ابتدائي لريل جديد، من قيم خام (outfit_style/
    dominant_color) بدل ReelInteraction ORM object كما كانت سابقًا.

    السبب: وقت POST /reels/rerank، الريل قد يكون وصل لأول مرة من Pexels
    ولم يتفاعل معه أي مستخدم بعد — أي لا يوجد صف ReelInteraction له على
    الإطلاق. البيانات المتاحة الوحيدة في هذه اللحظة هي السياق الخفيف
    اللي يرسله Flutter مباشرة مع كل ريل (outfit_style/dominant_color من
    FashionReel، نفس البيانات المستخدمة أصلاً في PexelsReelService).

    تستخدم الآن taste_profile._partial_features_from_raw (الدالة الخام
    المشتركة) بدل _reel_partial_features (التي تتطلب ORM object) —
    نفس المنطق بالضبط، لا تكرار.

    باقي الأبعاد التي لا يوجد عليها دليل من هذا الريل تحديدًا تبقى على
    DEFAULT_FEATURE_VECTOR المحايد، بدل تخمينها.
    """
    partial = taste_profile._partial_features_from_raw(outfit_style, dominant_color)

    vector = list(DEFAULT_FEATURE_VECTOR)
    for idx, value in partial.items():
        vector[idx] = value

    return _project(vector)
