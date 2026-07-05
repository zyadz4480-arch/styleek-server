"""
app/ml/two_tower.py

نموذج Two-Tower (Dual Encoder) لتدريب UserEmbedding وReelEmbedding من
تفاعلات ReelInteraction الحقيقية (عبر dataset.py). هذا أول ملف فعلي في
المرحلة 3 — يستهلك ReelTwoTowerDataset من app/ml/dataset.py.

البنية (مؤكَّدة مع المستخدم):
  User Tower:  Linear(22→64) → ReLU → Linear(64→128) → L2-normalize
  Reel Tower:  نفس البنية بالضبط، أوزان منفصلة تمامًا (لا مشاركة أوزان)

دالة الخسارة: InfoNCE بـ in-batch negatives (مؤكَّدة أيضًا) — لكل عيّنة
في الدفعة، الريل المطابق فعليًا هو الإيجابي الوحيد، وكل الريلز الأخرى
في نفس الدفعة سلبيات ضمنية. وزن العيّنة (من dataset.py، أي
taste_profile._reel_signal_weight) يُستخدَم كمُعامِل ترجيح لكل عيّنة
في المتوسط الموزون للخسارة — عيّنة بوزن أعلى (مثلاً share=1.3) تؤثر
أكثر من عيّنة بوزن أقل (مثلاً watch ضعيف=0.12).

⚠️ قيد معروف لـ in-batch negatives: لو نفس reel_id ظهر أكثر من مرة
داخل نفس الدفعة (مستخدمين مختلفين تفاعلوا مع نفس الريل)، يُعامَل
كـ"سلبي" للعيّنة الأخرى رغم كونه تطابقًا فعليًا محتملاً — قيد معروف
ومقبول لهذا الأسلوب، لا يحتاج إصلاحًا الآن.

⚠️ حجم الدفعة (batch_size) يجب أن يكون > 1 دائمًا — بدفعة بحجم 1 لا
توجد سلبيات إطلاقًا وInfoNCE تفقد معناها. هذا مسؤولية DataLoader في
trainer.py القادم، لا هذا الملف.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml.features import FEATURE_DIM

EMBEDDING_DIM = 128  # يطابق Vector(128) في UserEmbedding/ReelEmbedding بـ models.py
HIDDEN_DIM = 64


class Tower(nn.Module):
    """بنية واحدة مشتركة تُستخدَم مرتين بأوزان منفصلة (user_tower وreel_tower) —
    لا مشاركة أوزان بين الاثنين، كل واحد يتعلم تمثيله الخاص."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        # L2-normalize — قرار معماري ثابت للمشروع، يطبَّق هنا كما في
        # cold_start.py، حتى تكون مخرجات النموذج المدرَّب متوافقة مباشرة
        # مع نفس تنسيق UserEmbedding/ReelEmbedding المخزَّن في pgvector.
        return F.normalize(z, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()
        self.user_tower = Tower(input_dim, hidden_dim, embedding_dim)
        self.reel_tower = Tower(input_dim, hidden_dim, embedding_dim)

    def forward(
        self,
        user_features: torch.Tensor,
        reel_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """user_features, reel_features: (batch, FEATURE_DIM)
        يُرجع (user_embedding, reel_embedding): كلاهما (batch, EMBEDDING_DIM)،
        مُطبَّعان L2 مسبقًا (جاهزان للتخزين المباشر في pgvector)."""
        user_emb = self.user_tower(user_features)
        reel_emb = self.reel_tower(reel_features)
        return user_emb, reel_emb

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """للاستدلال المنفرد — إنتاج UserEmbedding فقط، بدون الحاجة لدفعة ريلز
        مقابلة (مفيد لـ inference.py القادم)."""
        return self.user_tower(user_features)

    def encode_reel(self, reel_features: torch.Tensor) -> torch.Tensor:
        """للاستدلال المنفرد — إنتاج ReelEmbedding فقط."""
        return self.reel_tower(reel_features)


def in_batch_infonce_loss(
    user_emb: torch.Tensor,
    reel_emb: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    user_emb, reel_emb: (batch, EMBEDDING_DIM), مُطبَّعان L2 مسبقًا.
    weights: (batch,) قوة الإشارة الإيجابية لكل عيّنة (من dataset.py،
             أي taste_profile._reel_signal_weight — لا صفر أبدًا هنا،
             تم استبعاد weight<=0 مسبقًا في dataset.py).
    temperature: قيمة ابتدائية معقولة شائعة الاستخدام لـ InfoNCE — قابلة
                 للضبط لاحقًا عبر التجريب الفعلي (hyperparameter، وليست
                 قرارًا معماريًا يحتاج تأكيدًا مسبقًا).

    كل عنصر i في الدفعة: user_emb[i] يفترض أن يتطابق مع reel_emb[i]
    (الإيجابي الوحيد)، وكل reel_emb[j] حيث j != i تُعامَل كسلبي ضمني.
    """
    if user_emb.size(0) < 2:
        raise ValueError(
            "in_batch_infonce_loss تحتاج batch_size >= 2 على الأقل — "
            "بدفعة بحجم 1 لا توجد سلبيات إطلاقًا وInfoNCE تفقد معناها"
        )

    logits = user_emb @ reel_emb.T / temperature  # (batch, batch)
    labels = torch.arange(logits.size(0), device=logits.device)

    per_sample_loss = F.cross_entropy(logits, labels, reduction="none")  # (batch,)

    weight_sum = weights.sum().clamp(min=1e-8)
    loss = (per_sample_loss * weights).sum() / weight_sum
    return loss
