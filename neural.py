"""
app/ml/neural.py
شبكة عصبية حقيقية (PyTorch) تحل محل الـ StyleEnsemble اليدوي.

الفرق الجوهري عن النظام القديم:
  - قديمًا: 5 نماذج منفصلة (Logistic/Ridge/SVM/Tree/Forest) تُدمَج بأوزان
    (expert_logits) تُحدَّث بقاعدة رياضية مكتوبة يدويًا بعد كل تفاعل.
  - الآن: شبكة واحدة موحّدة (end-to-end) — كل الأوزان، بما فيها طريقة
    "دمج" القرار النهائي، تتعلمها الشبكة نفسها عبر backpropagation.
    لا يوجد أي وزن يُعطى يدويًا من الكود.

البنية:
  Input (20) → Trunk مشترك (طبقتين مخفيتين + Dropout) → 32 بُعد
             → 5 "رؤوس" (تعادل أسماء الخبراء القدامى تاريخيًا فقط، للتوافق مع main.dart)
             → طبقة دمج متعلَّمة (Linear(5,1)) → final_score
             → رأس منفصل لتقدير الـ rating المستمر (0..1)

كل هذه الطبقات تُدرَّب معًا في نفس الوقت (نفس الـ loss.backward()) —
يعني الشبكة "تقرر" بنفسها أهمية كل مسار أثناء التدريب، وليس نحن.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

from app.ml.features import FEATURE_DIM

# الأسماء دي للتوافق مع main.dart/service.py فقط (نفس شكل الاستجابة القديم)
EXPERT_NAMES = ["logistic", "linear", "svm", "tree", "forest"]


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


class _Net(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 32),
            nn.ReLU(),
        )
        # 5 رؤوس (بديل الخبراء الخمسة القدامى) — أوزانها متعلَّمة بالكامل
        self.heads = nn.Linear(32, 5)
        # طبقة الدمج النهائية — هي "الأوزان" الجديدة، لكن متعلَّمة لا معطاة يدويًا
        self.combine = nn.Linear(5, 1)
        self.rating_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        heads = torch.sigmoid(self.heads(z))            # (batch, 5)
        final = torch.sigmoid(self.combine(heads))       # الدمج المتعلَّم
        rating = torch.sigmoid(self.rating_head(z))
        return heads, final.squeeze(-1), rating.squeeze(-1)


class StyleNeuralNet:
    """
    واجهة متوافقة مع StyleEnsemble القديمة (نفس أسماء الدوال) بحيث
    لا يحتاج service.py أي تعديل جوهري في تدفق العمل — فقط استبدال
    الاستيراد (import) من ensemble.py إلى neural.py.
    """

    def __init__(self):
        self.net = _Net()
        self.is_trained = False
        self.train_metrics: dict = {}
        self.test_metrics: dict = {}

    # ------------------------------------------------------------------ #
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 300,
        lr: float = 1e-3,
    ) -> None:
        if len(np.unique(y)) < 2:
            # لا يمكن تدريب مصنّف بفئة واحدة فقط — ننتظر بيانات أكثر تنوّعًا
            return

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        r_t = torch.tensor(ratings, dtype=torch.float32)

        opt = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        bce = nn.BCELoss()
        mse = nn.MSELoss()

        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            heads, final, rating = self.net(X_t)
            # كل رأس يُدفَع أيضًا نحو التصنيف الفعلي (إشراف إضافي)،
            # يمنع الرؤوس من الانهيار لحل تافه ويجبرها تتعلم تمثيلات مفيدة
            heads_loss = bce(heads, y_t.unsqueeze(1).expand_as(heads))
            final_loss = bce(final, y_t)
            rating_loss = mse(rating, r_t)
            loss = final_loss + 0.3 * heads_loss + 0.3 * rating_loss
            loss.backward()
            opt.step()

        self.is_trained = True

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not self.is_trained or len(X) == 0:
            return {}
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            _, final, _ = self.net(X_t)
            preds = (final.numpy() >= 0.5).astype(float)
        acc = float((preds == y).mean())
        return {"neural_net": acc}

    # ------------------------------------------------------------------ #
    def predict_one(self, x: np.ndarray, expert_logits: list[float] | None = None) -> PredictionResult:
        """
        expert_logits موجودة فقط للتوافق مع توقيع الدالة القديمة في service.py —
        الشبكة الجديدة لا تستخدمها إطلاقًا؛ الدمج داخلي ومتعلَّم بالكامل.
        """
        if not self.is_trained:
            return PredictionResult(0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, False)

        self.net.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            heads, final, rating = self.net(x_t)
            h = heads.squeeze(0).numpy()
            f = float(final.item())
            r = float(rating.item())

        agreement = float(max(h.mean(), 1 - h.mean()))
        return PredictionResult(
            logistic_proba=float(h[0]),
            linear_rating=r,
            svm_confidence=float(h[2]),
            tree_proba=float(h[3]),
            forest_proba=float(h[4]),
            forest_agreement=agreement,
            final_score=f,
            is_trained=True,
        )

    def expert_predictions_for_update(self, x: np.ndarray) -> list[float]:
        """للتوافق فقط — لم تعد تُستخدم لتحديث أوزان يدوية (لا وجود لها الآن)."""
        r = self.predict_one(x)
        return [r.logistic_proba, r.linear_rating, r.svm_confidence, r.tree_proba, r.forest_proba]

    # ------------------------------------------------------------------ #
    # للحفظ/التحميل عبر torch (بديل pickle الخاص بـ sklearn)
    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd) -> None:
        self.net.load_state_dict(sd)
        self.is_trained = True
