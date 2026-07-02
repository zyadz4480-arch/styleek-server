"""
حفظ/تحميل StyleNeuralNet لكل مستخدم كملف torch (.pt) على القرص.
يعادل _saveModels/_loadModels عبر SharedPreferences في main.dart (سطر 4483-4527)
لكن هنا الحفظ على السيرفر بدل جهاز المستخدم.

ملاحظة: تحوّلنا من joblib (sklearn) إلى torch.save/torch.load لأن الموديل
الآن شبكة عصبية (StyleNeuralNet) وليس مجموعة نماذج sklearn.
"""
from __future__ import annotations
import os
import torch
from app.config import settings
from app.ml.neural import StyleNeuralNet

os.makedirs(settings.models_dir, exist_ok=True)


def _path_for(user_id: str) -> str:
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return os.path.join(settings.models_dir, f"{safe_id}.pt")


def load_model(user_id: str) -> StyleNeuralNet:
    path = _path_for(user_id)
    model = StyleNeuralNet()
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    return model


def save_model(user_id: str, model: StyleNeuralNet) -> None:
    torch.save(model.state_dict(), _path_for(user_id))
