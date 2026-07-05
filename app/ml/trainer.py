"""
app/ml/trainer.py

حلقة تدريب TwoTowerModel (المرحلة 3) — تدريب فقط، بدون أي كتابة
لقاعدة البيانات. هذا الملف مسؤوليته الوحيدة: قراءة بيانات التدريب،
تدريب النموذج، حفظ أفضل نسخة منه على القرص (best_model.pt).

⚠️ قرار معماري متعمَّد (بناءً على توصية المستخدم): لا يوجد هنا أي
استدعاء لكتابة user_embeddings/reel_embeddings. توليد ورفع الـ
embeddings خطوة مستقلة تمامًا في app/ml/export_embeddings.py، تُشغَّل
فقط بعد نجاح التدريب هنا والتأكد من best_model.pt. الفائدة: فشل رفع
القاعدة لا يستوجب إعادة تدريب النموذج بالكامل — فقط إعادة تشغيل خطوة
النشر.

⚠️ لا يوجد حاليًا مجموعة تحقق (validation set) منفصلة — عدد التفاعلات
الحقيقية المتاحة الآن محدود جدًا (212+ صف فقط في reel_interactions).
"أفضل نموذج" هنا يعني: أدنى متوسط خسارة تدريب لكل Epoch. هذا تبسيط
معروف ومقصود مؤقتًا؛ يُعاد النظر فيه عند نمو حجم البيانات فعليًا.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.dataset import ReelTwoTowerDataset, build_reel_training_samples
from app.ml.two_tower import TwoTowerModel, in_batch_infonce_loss


DEFAULT_MODEL_PATH = Path("app/ml/artifacts/best_model.pt")


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.1
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)


@dataclass
class TrainingStats:
    """إحصاءات تدريب لسهولة مراقبة الأداء (بند 10 في طلب المستخدم)."""

    num_samples: int
    num_batches_per_epoch: int
    epochs_run: int
    best_epoch: int
    best_avg_loss: float
    loss_per_epoch: list[float]
    duration_seconds: float


def _train_epoch(
    model: TwoTowerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float,
) -> float:
    """تشغيل Epoch واحد كاملًا، يُرجع متوسط الخسارة عبر كل الدُفعات.

    دفعات بحجم 1 (قد تحدث في آخر دفعة من epoch لو عدد العيّنات لا
    يقبل القسمة على batch_size بالتمام) تُتخطى — in_batch_infonce_loss
    ترفض batch_size < 2 صراحة (راجع two_tower.py)، ولا معنى لسلبيات
    ضمنية بعيّنة واحدة على أي حال."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for user_features, reel_features, weights in loader:
        if user_features.size(0) < 2:
            continue

        optimizer.zero_grad()
        user_emb, reel_emb = model(user_features, reel_features)
        loss = in_batch_infonce_loss(user_emb, reel_emb, weights, temperature=temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError(
            "لا توجد دفعة صالحة واحدة في هذا الـ Epoch (كل الدُفعات "
            "بحجم < 2) — قلّل batch_size أو تأكد من وجود عيّنات كافية"
        )

    return total_loss / num_batches


def _run_training_loop(
    samples,
    config: TrainingConfig,
) -> TrainingStats:
    """الجزء المتزامن بالكامل (torch لا يدعم async) — يُستدعى من
    train() بعد جلب البيانات من القاعدة بشكل async."""
    start = time.monotonic()

    dataset = ReelTwoTowerDataset(samples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    model = TwoTowerModel()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_avg_loss = float("inf")
    best_epoch = -1
    loss_per_epoch: list[float] = []

    config.model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        avg_loss = _train_epoch(model, loader, optimizer, config.temperature)
        loss_per_epoch.append(avg_loss)

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), config.model_path)

    duration = time.monotonic() - start

    return TrainingStats(
        num_samples=len(samples),
        num_batches_per_epoch=len(loader),
        epochs_run=config.epochs,
        best_epoch=best_epoch,
        best_avg_loss=best_avg_loss,
        loss_per_epoch=loss_per_epoch,
        duration_seconds=duration,
    )


async def train(db: AsyncSession, config: TrainingConfig | None = None) -> TrainingStats:
    """نقطة الدخول الوحيدة لهذا الملف — async فقط لجلب بيانات التدريب
    (build_reel_training_samples)، ثم تسليم التحكم بالكامل لحلقة
    تدريب متزامنة (torch). لا يكتب أي شيء لقاعدة البيانات."""
    config = config or TrainingConfig()

    samples = await build_reel_training_samples(db)
    if len(samples) < 2:
        raise RuntimeError(
            f"عدد عيّنات التدريب المتاحة ({len(samples)}) غير كافٍ "
            "— يلزم عيّنتان على الأقل (بعد استبعاد weight<=0) لبدء التدريب"
        )

    stats = _run_training_loop(samples, config)

    return stats


def load_best_model(model_path: Path = DEFAULT_MODEL_PATH) -> TwoTowerModel:
    """يُستخدَم من export_embeddings.py (خطوة منفصلة تمامًا، بعد نجاح
    التدريب) لاسترجاع أفضل نسخة مدرَّبة من النموذج."""
    model = TwoTowerModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
