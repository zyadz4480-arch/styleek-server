"""
إعادة ترتيب الريلز حسب قرب كل ريل من ذوق المستخدم — pgvector.

القرار المعماري: لا نُعيد ترتيب القائمة كاملة من الصفر. فقط الريلز التي
لها ReelEmbedding مخزَّن فعليًا يُعاد ترتيبها فيما بينها (الأقرب لـ
UserEmbedding أولًا)، وتُوضع كل واحدة في نفس "الفتحة" (index) التي كانت
تحتلها إحدى الريلز ذوات الـ embedding أصلًا. الريلز بلا embedding (غالبية
المحتوى الجديد كليًا) تبقى في مكانها الأصلي بالضبط دون أي تحريك.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ReelEmbedding, UserEmbedding


async def rerank_reel_ids(
    db: AsyncSession,
    user_id: str,
    reel_ids: list[str],
) -> list[str]:
    if not reel_ids:
        return reel_ids

    # ① جلب embedding المستخدم — إن لم يوجد، لا يوجد أساس لإعادة الترتيب
    user_row = await db.get(UserEmbedding, user_id)
    if user_row is None:
        return reel_ids

    user_vec = user_row.embedding

    # ② جلب الريلز التي لها embedding من بين المُرسَلة فقط، مُرتَّبة
    #    بالفعل من الأقرب للأبعد عن ذوق المستخدم (L2 distance عبر pgvector)
    stmt = (
        select(ReelEmbedding.reel_id)
        .where(ReelEmbedding.reel_id.in_(reel_ids))
        .order_by(ReelEmbedding.embedding.l2_distance(user_vec))
    )
    result = await db.execute(stmt)
    ranked_with_embedding = [row[0] for row in result.all()]

    if not ranked_with_embedding:
        # لا يوجد أي ريل من هذه الدفعة له embedding بعد — لا تغيير
        return reel_ids

    ranked_set = set(ranked_with_embedding)

    # ③ نبني القائمة النهائية: نمرّ على الترتيب الأصلي، وفي كل "فتحة" كانت
    #    تحتلها ريل ذات embedding، نضع التالي من القائمة المُرتَّبة حسب
    #    القرب — بقية الفتحات (بلا embedding) تبقى كما هي تمامًا.
    output = list(reel_ids)
    cursor = 0
    for i, rid in enumerate(reel_ids):
        if rid in ranked_set:
            output[i] = ranked_with_embedding[cursor]
            cursor += 1

    return output
