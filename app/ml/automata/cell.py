"""
cell.py — أُعيد بناؤه هنا لأنه لم يُرفع ضمن ملفات المشروع الأصلية.
الحقول والقيم الافتراضية مُستنتَجة حصريًا من طريقة الاستخدام الفعلية في
automaton.py (لا افتراضات إضافية). إن اختلف عن الأصلي في أي تفصيل غير
مستخدم فعليًا في منطق الطاقة، فهذا لن يغيّر نتائج تحليل الاستقرار لأن كل
حسابات الطاقة تعتمد فقط على: embedding, energy, status, last_interaction_day.

[نسخة الإنتاج] أُضيفت to_dict/from_dict فقط (تسلسل للتخزين) — لا تغيير على
أي حقل أو سلوك تم التحقق منه في الجلسات السابقة.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class CellStatus(Enum):
    ALIVE = "alive"
    DORMANT = "dormant"
    DEAD = "dead"


@dataclass
class Cell:
    embedding: np.ndarray
    energy: float
    last_interaction_day: int = 0
    age: int = 0
    confidence: float = 0.5
    status: CellStatus = CellStatus.ALIVE
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    interaction_count: int = 0
    dormant_streak: int = 0
    cell_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ── [جديد — طبقة إنتاج] تسلسل/فك تسلسل للتخزين في PostgreSQL ────────
    # إضافية بحتة — لا تلمس أي منطق طاقة/دمج/ولادة تم التحقق منه أعلاه.
    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "embedding": self.embedding.tolist(),
            "energy": self.energy,
            "last_interaction_day": self.last_interaction_day,
            "age": self.age,
            "confidence": self.confidence,
            "status": self.status.value,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "interaction_count": self.interaction_count,
            "dormant_streak": self.dormant_streak,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Cell":
        return cls(
            embedding=np.array(d["embedding"], dtype=np.float64),
            energy=d["energy"],
            last_interaction_day=d.get("last_interaction_day", 0),
            age=d.get("age", 0),
            confidence=d.get("confidence", 0.5),
            status=CellStatus(d.get("status", "alive")),
            generation=d.get("generation", 0),
            parent_ids=list(d.get("parent_ids", [])),
            interaction_count=d.get("interaction_count", 0),
            dormant_streak=d.get("dormant_streak", 0),
            cell_id=d["cell_id"],
        )
