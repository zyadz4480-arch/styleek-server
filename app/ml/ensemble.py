"""
StyleEnsemble — المعادل السيرفري لِـ StyleMLOrchestrator في main.dart (سطر 4204-4536)
لكن بمكتبات scikit-learn حقيقية بدل الخوارزميات اليدوية.

الخمسة خبراء (نفس التخصصات الأصلية):
  1. LogisticRegression  → "القبول العام"           (احتمالية القبول)
  2. Ridge (linear)       → "درجة الرضا"              (قيمة مستمرة 0..1)
  3. LinearSVC(+platt)    → "ملاءمة المناسبة"         (على ميزات المناسبة فقط)
  4. DecisionTree         → "الملاءمة العملية"        (موسم/حرارة/طبقات)
  5. RandomForest         → "الذوق الشخصي طويل المدى" (تاريخ الاستخدام + DNA)

الدمج: نفس فكرة main.dart تمامًا — أوزان softmax متعلَّمة من logits تتحدّث
تلقائيًا بعد كل تفاعل بحسب دقة كل خبير (المكافأة = 1 - |توقع - فعلي|).
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from app.ml.features import PRACTICAL_INDICES, PERSONAL_TASTE_INDICES, FEATURE_DIM

EXPERT_NAMES = ["logistic", "linear", "svm", "tree", "forest"]
EXPERT_LR = 0.05          # يطابق AIConstants.lrExpert (main.dart سطر 134)
INITIAL_LOGITS = [1.00, 0.82, 0.69, 0.69, 1.20]  # يطابق main.dart سطر 4236-4242


def softmax_weights(logits: list[float]) -> list[float]:
    exps = [math.exp(v) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def update_expert_logits(logits: list[float], actual: float, preds: list[float]) -> list[float]:
    """يطابق _updateExpertWeights في main.dart سطر 4254-4271."""
    new_logits = list(logits)
    for i, pred in enumerate(preds):
        error = abs(pred - actual)
        reward = 1.0 - error
        new_logits[i] += EXPERT_LR * (reward - 0.5) * 2
        new_logits[i] = max(-2.0, min(3.0, new_logits[i]))
    return new_logits


@dataclass
class PredictionResult:
    logistic_proba: float
    linear_rating: float
    svm_confidence: float
    tree_proba: float
    forest_proba: float
    forest_agreement: float
    final_score: float
    is_trained: bool


class StyleEnsemble:
    """يُدرَّب لكل مستخدم (أو يمكن تجميعه لاحقًا عبر المستخدمين لتعلّم جماعي)."""

    def __init__(self):
        self.logistic = LogisticRegression(max_iter=1000)
        self.linear = Ridge(alpha=1.0)
        # LinearSVC لا يعطي احتمالات مباشرة → نلفّه بمعايرة Platt لنحصل على "ثقة" 0..1
        self.svm = CalibratedClassifierCV(LinearSVC(max_iter=5000), method="sigmoid", cv=3)
        self.tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, random_state=42)
        self.forest = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)

        self.is_trained = False
        self.train_metrics: dict = {}
        self.test_metrics: dict = {}

    # ------------------------------------------------------------------ #
    def train(self, X: np.ndarray, y: np.ndarray, ratings: np.ndarray) -> None:
        """
        X: (n, 20) متجهات الميزات
        y: (n,) تصنيف 0/1 (رفض/قبول)
        ratings: (n,) درجة رضا مستمرة 0..1 (يتدرّب عليها خبير linear)
        يطابق trainAll في main.dart سطر 4325-4347 (تقسيم 80/20 داخليًا في الطبقة الأعلى).
        """
        if len(np.unique(y)) < 2:
            # لا يمكن تدريب مصنّف بفئة واحدة فقط — ننتظر بيانات أكثر تنوّعًا
            return

        self.logistic.fit(X, y)
        self.linear.fit(X, ratings)

        # خبير SVM: يتخصص في ملاءمة المناسبة فقط (على ميزات المناسبة، مثل main.dart)
        self.svm.fit(X, y)

        # خبير Tree: على "الملاءمة العملية" فقط (موسم/حرارة/طبقات) — main.dart سطر 4334
        X_practical = X[:, PRACTICAL_INDICES]
        self.tree.fit(X_practical, y)

        self.forest.fit(X, y)

        self.is_trained = True

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not self.is_trained or len(X) == 0:
            return {}
        X_practical = X[:, PRACTICAL_INDICES]
        return {
            "logistic": float(accuracy_score(y, self.logistic.predict(X))),
            "svm": float(accuracy_score(y, self.svm.predict(X))),
            "tree": float(accuracy_score(y, self.tree.predict(X_practical))),
            "forest": float(accuracy_score(y, self.forest.predict(X))),
        }

    # ------------------------------------------------------------------ #
    def predict_one(self, x: np.ndarray, expert_logits: list[float]) -> PredictionResult:
        """يطابق predict() في main.dart سطر 4351-4392."""
        if not self.is_trained:
            return PredictionResult(0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, False)

        x2d = x.reshape(1, -1)
        x_practical = x[PRACTICAL_INDICES].reshape(1, -1)

        lp = float(self.logistic.predict_proba(x2d)[0][1])
        lr = float(np.clip(self.linear.predict(x2d)[0], 0.0, 1.0))
        svm_conf = float(self.svm.predict_proba(x2d)[0][1])
        tree_p = float(self.tree.predict_proba(x_practical)[0][-1])

        forest_votes = np.array([t.predict(x2d)[0] for t in self.forest.estimators_])
        forest_p = float(forest_votes.mean())
        forest_agree = float(max(forest_votes.mean(), 1 - forest_votes.mean()))

        w = softmax_weights(expert_logits)
        final = lp * w[0] + lr * w[1] + svm_conf * w[2] + tree_p * w[3] + forest_p * w[4]
        final = float(np.clip(final, 0.0, 1.0))

        return PredictionResult(lp, lr, svm_conf, tree_p, forest_p, forest_agree, final, True)

    def expert_predictions_for_update(self, x: np.ndarray) -> list[float]:
        """يعيد توقعات الخبراء الخمسة الخام لتحديث الأوزان — يطابق main.dart سطر 4292-4306."""
        r = self.predict_one(x, INITIAL_LOGITS)  # الـ logits هنا غير مهمة، فقط نحتاج توقعات كل خبير
        return [r.logistic_proba, r.linear_rating, r.svm_confidence, r.tree_proba, r.forest_proba]
