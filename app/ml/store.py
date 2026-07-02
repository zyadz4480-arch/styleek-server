"""
حفظ/تحميل StyleNeuralNet لكل مستخدم كملف torch (.pt) على القرص.
يعادل _saveModels/_loadModels عبر SharedPreferences في main.dart (سطر 4483-4527)
لكن هنا الحفظ على السيرفر بدل جهاز المستخدم.

ملاحظة: تحوّلنا من joblib (sklearn) إلى torch.save/torch.load لأن الموديل
الآن شبكة عصبية (StyleNeuralNet) وليس مجموعة نماذج sklearn.
"""
from __future__ import annotations
import logging
import os
import torch
from app.config import settings
from app.ml.neural import StyleNeuralNet

logger = logging.getLogger(__name__)

os.makedirs(settings.models_dir, exist_ok=True)


def _path_for(user_id: str) -> str:
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return os.path.join(settings.models_dir, f"{safe_id}.pt")


def load_model(user_id: str) -> StyleNeuralNet:
    """يرجع موديل مُدرَّب لو الملف موجود وسليم ومتوافق مع البنية الحالية،
    وإلا يرجع موديل جديد فارغ (is_trained=False) بدل ما يفشل الطلب بالكامل —
    مهم خصوصًا عند تغيير بنية الشبكة (مثل تغيير FEATURE_DIM) بين إصدارات."""
    path = _path_for(user_id)
    model = StyleNeuralNet()
    if not os.path.exists(path):
        return model

    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    except Exception as exc:
        # يشمل: ملف تالف، أو state_dict غير متوافق مع بنية الشبكة الحالية
        # (مثلاً بعد تعديل معماري كإضافة Embedding جديد) — بدل الفشل، نبدأ
        # بموديل نظيف ونسجّل الخطأ للمراجعة.
        logger.warning("فشل تحميل موديل %s (%s) — سيبدأ من موديل جديد.", user_id, exc)
        model = StyleNeuralNet()

    return model


def save_model(user_id: str, model: StyleNeuralNet) -> None:
    """حفظ ذرّي (Atomic): نكتب لملف مؤقت أولاً ثم نستبدل الملف الأصلي دفعة
    واحدة عبر os.replace — لو انقطعت العملية أثناء الكتابة، الملف الأصلي
    (لو كان موجودًا) يبقى سليمًا بدل ما يتلف في نص الكتابة."""
    path = _path_for(user_id)
    tmp_path = path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, path)
