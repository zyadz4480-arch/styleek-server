"""
app/ml/embedding_repository.py  [ملف جديد]

نقطة الوصول الوحيدة لـ"ضمان وجود ReelEmbedding" — قراءة إن وُجد،
أو حساب فوري عبر cold_start_reel_embedding وتخزينه إن لم يوجد.

الهدف: rerank_reels_by_embedding (في reel_service.py) تبقى مسؤولة عن
الترتيب فقط، ولا تعرف تفاصيل "كيف يُنشأ embedding جديد". أي خدمة
مستقبلية (مثل /reels/feed لاحقًا) تستدعي get_or_create_reel_embedding
مباشرة بدل تكرار منطق القراءة/الحساب/الحفظ من جديد.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelEmbedding
from app.ml.cold_start import cold_start_reel_embedding


async def get_or_create_reel_embedding(
    db: AsyncSession,
    reel_id: str,
    outfit_style: Optional[str],
    dominant_color: Optional[str],
) -> list[float]:
    """
    يرجع embedding الريل:
      - من جدول reel_embeddings مباشرة لو موجود مسبقًا (سواء كان
        embedding_source='cold_start' من استدعاء سابق، أو 'trained'
        من export_embeddings.py بعد تدريب فعلي).
      - وإلا: يحسبه فورًا عبر cold_start_reel_embedding من السياق الخام
        المُمرَّر (outfit_style/dominant_color)، يخزّنه في القاعدة
        (embedding_source='cold_start') حتى لا يُعاد حسابه في كل طلب
        لاحق لنفس الريل، ثم يرجعه.

    ملاحظة تعمّد: لا نستبدل embedding موجود بالفعل حتى لو كان مصدره
    'cold_start' وحديثًا — استبدال قيم cold_start بقيم 'trained' الأدق
    هو مسؤولية export_embeddings.py حصرًا (بعد تدريب فعلي)، لا هذه
    الدالة. هذا يحافظ على وضوح المسؤوليات: هذه الدالة تضمن *وجود* قيمة
    فقط، لا تُقيّم *جودتها*.
    """
    row = await db.get(ReelEmbedding, reel_id)
    if row is not None:
        return row.embedding

    embedding = cold_start_reel_embedding(outfit_style, dominant_color)

    row = ReelEmbedding(
        reel_id=reel_id,
        embedding=embedding,
        embedding_source="cold_start",
    )
    db.add(row)
    await db.commit()

    return embedding
