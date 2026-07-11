"""
cell.py — أُعيد بناؤه هنا لأنه لم يُرفع ضمن ملفات المشروع الأصلية.
الحقول والقيم الافتراضية مُستنتَجة حصريًا من طريقة الاستخدام الفعلية في
automaton.py (لا افتراضات إضافية). إن اختلف عن الأصلي في أي تفصيل غير
مستخدم فعليًا في منطق الطاقة، فهذا لن يغيّر نتائج تحليل الاستقرار لأن كل
حسابات الطاقة تعتمد فقط على: embedding, energy, status, last_interaction_day.

[نسخة الإنتاج] أُضيفت to_dict/from_dict فقط (تسلسل للتخزين) — لا تغيير على
أي حقل أو سلوك تم التحقق منه في الجلسات السابقة.

[جديد — وسم دلالي] tag_votes: عداد تصويت تراكمي بسيط {اسم_الوسم: قيمة}
(مثلاً {"formal": 3.0, "occasion_work": 2.0}) — إضافية بحتة، لا تدخل في
أي حساب طاقة/تشابه/دمج/ولادة. الهدف تفسيري فقط: بدل أن تكون الخلية
"متجه 128 بُعد" مجرد بلا معنى مقروء، تحمل أيضًا دليلًا نصيًا تراكميًا عن
*لماذا* تكوّنت (نفس الكلمات المفتاحية المُتحقَّق منها في
app/ml/taste_profile.py — لا منطق جديد). يُملأ من app/ml/automata/service.py
(extract_semantic_tags)، ويُدمَج تراكميًا في automaton.py عند كل تفاعل/
دمج/ولادة.
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
    tag_votes: dict[str, float] = field(default_factory=dict)

    def dominant_tag(self) -> str | None:
        """أعلى وسم تصويتًا لهذه الخلية، أو None لو ما فيها أي دليل دلالي
        بعد (مثلاً خلية وُلدت من تفاعل بلا outfit_style/dominant_color)."""
        if not self.tag_votes:
            return None
        return max(self.tag_votes.items(), key=lambda kv: kv[1])[0]

    def top_tags(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.tag_votes.items(), key=lambda kv: kv[1], reverse=True)[:n]

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
            "tag_votes": self.tag_votes,
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
            tag_votes=dict(d.get("tag_votes", {})),  # .get للتوافق مع صفوف قديمة بلا العمود
        )
