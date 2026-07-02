"""
app/ml/neural.py — الإصدار الثاني (v2)

تحسينات على v1 بناءً على مراجعة تقنية دقيقة:

1. Embeddings متعلَّمة لفئة الملابس والمناسبة (بدل الاعتماد الكامل على جدول
   أوزان يدوي _CAT_WEIGHTS في features.py). الشبكة الآن تقرر بنفسها كيف
   تمثّل كل فئة، بدل رقم ثابت مكتوب مسبقًا.

2. رؤوس متخصصة فعليًا (لا تقلّد بعضها البعض كما في v1):
     - weather_head   → يتعلم استنتاج توافق الموسم/الطقس (season_match)
     - color_head     → يتعلم استنتاج خصائص اللون (سطوع/تشبع/دفء)
     - occasion_head  → تصنيف المناسبة (6 فئات: عمل/جامعة/خروجة/مناسبة/رياضة/بلا)
     - history_head   → يتعلم استنتاج التاريخ السلوكي (عدد الارتداء/مفضلة/أيام منذ آخر ارتداء)
     - like_head      → المهمة الرئيسية: هل ستُقبَل القطعة؟
   كل رأس له هدف تدريب مختلف فعليًا (auxiliary supervision)، فتصبح
   التمثيلات متمايزة بدل أن تتقارب لنفس الشيء.

3. طبقة الدمج أصبحت MLP صغيرة تأخذ التمثيل المشترك + مخرجات كل الرؤوس
   المتخصصة، بدل جمع خطي بسيط (Linear(5,1)) كما في v1.

4. Attention / Transformer: أُجِّلا عمدًا (قرار خريطة الطريق) — يحتاجان
   بيانات حقيقية أكبر بكثير من مرحلة الـ MVP الحالية ليكون لهما فائدة
   تفوق تعقيدهما.

5. "التعلّم المستمر" (Online Learning) الحقيقي (تحديث لكل عيّنة) لسه
   مؤجَّل، لكن أضفنا خطوة عملية بديلة: التدريب الآن Warm-start — يكمل من
   آخر أوزان محفوظة بدل إعادة البدء العشوائي الكامل في كل مرة (service.py
   يمرر الموديل المحفوظ إن وُجد بدل إنشاء موديل جديد فارغ).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

from app.ml.features import (
    FEATURE_DIM, NUM_CATEGORIES, NUM_OCCASIONS,
    IDX_CATEGORY_ID, IDX_OCCASION_ID,
    IDX_SEASON_MATCH, IDX_BRIGHTNESS, IDX_SATURATION, IDX_WARM_COLOR,
    IDX_WEAR_COUNT, IDX_FAVORITE, IDX_DAYS_SINCE,
    IDX_FORMAL, IDX_CASUAL, IDX_SPORTY,
    OCCASION_INDICES,
)

# الميزات اليدوية اللي بتكرر نفس معنى الـ Embeddings (cat_w + occasion one-hot).
# بنخليها تخفت تدريجيًا مع كل "جيل" تدريب بدل حذفها فجأة، عشان نمنع الشبكة
# تعتمد عليها كـ"طريق مختصر" (Shortcut Learning) بدل ما تتعلم الـ Embedding
# الحقيقي بنفسها. الجدول: جيل 0=100%، 1=60%، 2=30%، 3+=0% (تعتمد بالكامل
# على الـ Embeddings المتعلَّمة).
_MANUAL_SIGNAL_IDX = [IDX_FORMAL, IDX_CASUAL, IDX_SPORTY] + OCCASION_INDICES
_DECAY_SCHEDULE = [1.0, 0.6, 0.3, 0.0]

# أسماء تاريخية للتوافق مع main.dart (نفس أسماء حقول PredictionOut)،
# لكنها الآن مُعاد تعريفها لتمثيل الرؤوس المتخصصة الفعلية أدناه —
# الخريطة موثّقة بوضوح في predict_one().
EXPERT_NAMES = ["logistic", "linear", "svm", "tree", "forest"]

_CONTINUOUS_IDX = [i for i in range(FEATURE_DIM) if i not in (IDX_CATEGORY_ID, IDX_OCCASION_ID)]
_NUM_CONTINUOUS = len(_CONTINUOUS_IDX)  # 20


@dataclass
class PredictionResult:
    logistic_proba: float   # = like_head (احتمال القبول — الرأس الرئيسي)
    linear_rating: float    # = rating_head (تقييم مستمر)
    svm_confidence: float   # = weather_head (توافق الموسم/الطقس)
    tree_proba: float       # = color_head (توافق اللون)
    forest_proba: float     # = occasion_head (أعلى احتمال مناسبة)
    forest_agreement: float # = تباين الرؤوس المتخصصة (مقياس اتفاق حقيقي)
    final_score: float      # = ناتج MLP الدمج النهائي
    is_trained: bool


class _Net(nn.Module):
    def __init__(self, hidden: int = 64, emb_cat: int = 8, emb_occ: int = 4):
        super().__init__()
        self.cat_embedding = nn.Embedding(NUM_CATEGORIES, emb_cat)
        self.occ_embedding = nn.Embedding(NUM_OCCASIONS, emb_occ)

        trunk_in = _NUM_CONTINUOUS + emb_cat + emb_occ
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )  # → تمثيل مشترك (hidden أبعاد)

        # ---- رؤوس متخصصة فعليًا (كل واحد بمهمة مختلفة حقًا) ----
        self.weather_head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
        self.color_head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
        self.occasion_head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, NUM_OCCASIONS))
        self.history_head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
        self.like_head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
        self.rating_head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

        # ---- طبقة الدمج: MLP صغيرة على التمثيل المشترك + كل الرؤوس ----
        combine_in = hidden + 1 + 1 + NUM_OCCASIONS + 1 + 1  # trunk + weather + color + occ + history + like
        self.combine = nn.Sequential(
            nn.Linear(combine_in, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # عدّاد "الجيل" ونسبة الإشارات اليدوية — يُحفَظان مع الموديل (state_dict)
        # حتى يستمرا صحيحين عبر Warm-start بين جلسات التدريب المختلفة.
        self.register_buffer("training_round", torch.tensor(0))
        self.register_buffer("manual_signal_decay", torch.tensor(_DECAY_SCHEDULE[0]))

    def forward(self, x: torch.Tensor):
        cont = x[:, _CONTINUOUS_IDX].clone()

        # نُطبّق نسبة التلاشي على الإشارات اليدوية المكرِّرة فقط (cat_w +
        # occasion one-hot)؛ باقي الميزات المستمرة (لون/طقس/تاريخ/DNA) تبقى
        # كما هي لأنها ليست مكررة مع أي Embedding.
        manual_local_idx = [_CONTINUOUS_IDX.index(i) for i in _MANUAL_SIGNAL_IDX]
        cont[:, manual_local_idx] = cont[:, manual_local_idx] * self.manual_signal_decay

        cat_id = x[:, IDX_CATEGORY_ID].long().clamp(0, NUM_CATEGORIES - 1)
        occ_id = x[:, IDX_OCCASION_ID].long().clamp(0, NUM_OCCASIONS - 1)

        emb = torch.cat([self.cat_embedding(cat_id), self.occ_embedding(occ_id)], dim=-1)
        z = self.trunk(torch.cat([cont, emb], dim=-1))  # التمثيل المشترك

        weather = torch.sigmoid(self.weather_head(z))
        color = torch.sigmoid(self.color_head(z))
        occasion_logits = self.occasion_head(z)
        occasion_probs = torch.softmax(occasion_logits, dim=-1)
        history = torch.sigmoid(self.history_head(z))
        like = torch.sigmoid(self.like_head(z))
        rating = torch.sigmoid(self.rating_head(z))

        combined_in = torch.cat([z, weather, color, occasion_probs, history, like], dim=-1)
        final = torch.sigmoid(self.combine(combined_in))

        return {
            "weather": weather.squeeze(-1), "color": color.squeeze(-1),
            "occasion_logits": occasion_logits, "occasion_probs": occasion_probs,
            "history": history.squeeze(-1), "like": like.squeeze(-1),
            "rating": rating.squeeze(-1), "final": final.squeeze(-1),
        }


def _aux_targets(X: torch.Tensor) -> dict:
    """يبني أهدافًا تدريبية حقيقية لكل رأس متخصص من الميزات الهندسية نفسها
    (مهمة مساعدة/إعادة بناء تجبر التمثيل المشترك يحتفظ بمعلومات متمايزة،
    بدل الانهيار لتمثيل واحد كما حدث في v1)."""
    weather_t = X[:, IDX_SEASON_MATCH]
    color_t = (X[:, IDX_BRIGHTNESS] + X[:, IDX_SATURATION] + X[:, IDX_WARM_COLOR]) / 3.0
    history_t = (X[:, IDX_WEAR_COUNT] + X[:, IDX_FAVORITE] + (1.0 - X[:, IDX_DAYS_SINCE])) / 3.0

    occ_onehot = X[:, OCCASION_INDICES]
    has_occasion = occ_onehot.sum(dim=-1) > 0.5
    occasion_class = torch.where(
        has_occasion,
        occ_onehot.argmax(dim=-1),
        torch.full_like(occ_onehot.argmax(dim=-1), len(OCCASION_INDICES)),
    )
    return {"weather": weather_t, "color": color_t, "history": history_t, "occasion_class": occasion_class}


class StyleNeuralNet:
    """واجهة متوافقة مع النسخة القديمة (train/evaluate/predict_one/...) —
    لا حاجة لتعديل service.py أو store.py."""

    def __init__(self):
        self.net = _Net()
        self.is_trained = False
        self.train_metrics: dict = {}
        self.test_metrics: dict = {}

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 300,
        lr: float = 5e-4,
    ) -> None:
        """ملاحظة: عند استدعائها على موديل مُحمَّل مسبقًا (Warm-start عبر
        load_model في service.py)، التدريب يكمل من نفس الأوزان بدل البدء
        من العشوائية — أسرع وأكثر استقرارًا مع تراكم التفاعلات."""
        if len(np.unique(y)) < 2:
            return

        # تقدّم جيل واحد لكل استدعاء تدريب، وتحدّث نسبة الإشارات اليدوية
        # وفق الجدول (100% → 60% → 30% → 0%) بدل حذفها فجأة.
        round_idx = min(int(self.net.training_round.item()) + 1, len(_DECAY_SCHEDULE) - 1)
        with torch.no_grad():
            self.net.training_round.fill_(round_idx)
            self.net.manual_signal_decay.fill_(_DECAY_SCHEDULE[round_idx])

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        r_t = torch.tensor(ratings, dtype=torch.float32)
        aux = _aux_targets(X_t)

        opt = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        bce = nn.BCELoss()
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()

        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = self.net(X_t)

            loss = (
                bce(out["final"], y_t)
                + 0.4 * bce(out["like"], y_t)
                + 0.3 * mse(out["rating"], r_t)
                + 0.2 * mse(out["weather"], aux["weather"])
                + 0.2 * mse(out["color"], aux["color"])
                + 0.2 * mse(out["history"], aux["history"])
                + 0.2 * ce(out["occasion_logits"], aux["occasion_class"])
            )
            loss.backward()
            opt.step()

        self.is_trained = True

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not self.is_trained or len(X) == 0:
            return {}
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            out = self.net(X_t)
            preds = (out["final"].numpy() >= 0.5).astype(float)
        acc = float((preds == y).mean())
        return {
            "neural_net_v2": acc,
            "training_round": int(self.net.training_round.item()),
            "manual_signal_decay": float(self.net.manual_signal_decay.item()),
        }

    def predict_one(self, x: np.ndarray, expert_logits: list[float] | None = None) -> PredictionResult:
        if not self.is_trained:
            return PredictionResult(0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, False)

        self.net.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            out = self.net(x_t)
            weather = float(out["weather"].item())
            color = float(out["color"].item())
            occ_probs = out["occasion_probs"].squeeze(0).numpy()
            history = float(out["history"].item())
            like = float(out["like"].item())
            rating = float(out["rating"].item())
            final = float(out["final"].item())

        signals = np.array([weather, color, occ_probs.max(), history, like])
        agreement = float(1.0 - signals.std())

        return PredictionResult(
            logistic_proba=like,
            linear_rating=rating,
            svm_confidence=weather,
            tree_proba=color,
            forest_proba=float(occ_probs.max()),
            forest_agreement=max(0.0, agreement),
            final_score=final,
            is_trained=True,
        )

    def expert_predictions_for_update(self, x: np.ndarray) -> list[float]:
        r = self.predict_one(x)
        return [r.logistic_proba, r.linear_rating, r.svm_confidence, r.tree_proba, r.forest_proba]

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd) -> None:
        self.net.load_state_dict(sd)
        self.is_trained = True
