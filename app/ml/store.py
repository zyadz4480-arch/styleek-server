"""
حفظ/تحميل StyleEnsemble لكل مستخدم كملف joblib على القرص.
يعادل _saveModels/_loadModels عبر SharedPreferences في main.dart (سطر 4483-4527)
لكن هنا الحفظ على السيرفر بدل جهاز المستخدم.
"""
from __future__ import annotations
import os
import joblib
from app.config import settings
from app.ml.ensemble import StyleEnsemble

os.makedirs(settings.models_dir, exist_ok=True)


def _path_for(user_id: str) -> str:
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return os.path.join(settings.models_dir, f"{safe_id}.joblib")


def load_model(user_id: str) -> StyleEnsemble:
    path = _path_for(user_id)
    if os.path.exists(path):
        return joblib.load(path)
    return StyleEnsemble()


def save_model(user_id: str, model: StyleEnsemble) -> None:
    joblib.dump(model, _path_for(user_id))
