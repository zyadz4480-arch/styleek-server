"""
app/services/interactions_v2_log.py  [جديد]

Phase 0 من خطة الترحيل (Architecture Freeze v1.0، §8):
تسجيل نسخة موازية من كل تفاعل في interactions_v2 (Dual-Write) دون التأثير
على أي مسار قديم. لا يُقرأ من هذا الجدول بعد في أي منطق تشغيلي — فقط يُكتب
إليه لتجميع بيانات تدريب جاهزة قبل بدء V1 الفعلي (§3، §6 من نفس الوثيقة).

قرار تصميم مُتعمَّد لـ V1: الكتابة تحدث ضمن **نفس** الـ DB session/transaction
التي يستخدمها المتصل (بدون commit مستقل هنا) — يعني هذا السطر يُضاف إلى
نفس commit() الموجود أصلاً في service.py/reel_service.py، فلا تعقيد إضافي
(لا queue، لا مهمة خلفية منفصلة). إذا أصبح هذا الجدول عنق زجاجة لاحقًا
(معدل كتابة عالٍ جدًا)، الخطوة التالية الطبيعية هي فصله إلى كتابة غير
متزامنة (fire-and-forget) — لكن هذا تحسين مؤجَّل عمدًا، غير مطلوب في V1.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import InteractionV2

logger = logging.getLogger(__name__)


def _time_of_day(dt: datetime) -> str:
    """تصنيف بسيط لوقت اليوم — يُخزَّن من الآن (V1) لكن لا يُستخدَم في أي
    منطق تدريب/ترتيب حتى V2 (انظر Architecture Freeze، جدول §1، القرار #9)."""
    hour = dt.astimezone(timezone.utc).hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def log_interaction_v2(
    db: AsyncSession,
    *,
    user_id: str,
    event_type: str,
    label: Optional[float] = None,
    weight: Optional[float] = None,
    reel_id: Optional[str] = None,
    outfit_id: Optional[str] = None,
    item_id: Optional[str] = None,
    session_id: Optional[str] = None,       # ⚠ V2 — يُمرَّر None دائمًا حاليًا
    watch_time: Optional[float] = None,
    watch_ratio: Optional[float] = None,
    weather: Optional[str] = None,
    temperature: Optional[float] = None,
    season: Optional[str] = None,
    occasion: Optional[str] = None,
    device: Optional[str] = None,
    app_version: Optional[str] = None,
) -> None:
    """
    يضيف صفًا إلى الـ session الحالية (db.add) دون commit مستقل — المتصل
    هو من يستدعي commit() كالمعتاد (Interaction/ReelInteraction وInteractionV2
    يُحفَظان معًا في نفس المعاملة). عند أي خطأ غير متوقَّع في بناء الصف
    (مثلاً قيمة لا تطابق نوع العمود)، نسجّل تحذيرًا ولا نرفع استثناء —
    فشل هذا التسجيل الإضافي لا يجب أبدًا أن يُسقط الكتابة الأساسية.
    """
    try:
        now = datetime.now(timezone.utc)
        row = InteractionV2(
            user_id=user_id,
            session_id=session_id,
            reel_id=reel_id,
            outfit_id=outfit_id,
            item_id=item_id,
            event_type=event_type,
            label=label,
            weight=weight,
            watch_time=watch_time,
            watch_ratio=watch_ratio,
            weather=weather,
            temperature=float(temperature) if temperature is not None else None,
            season=season,
            occasion=occasion,
            time_of_day=_time_of_day(now),
            location_type=None,  # ⚠ V2 — غير مُجمَّع بعد عمدًا (خصوصية)
            device=device,
            app_version=app_version,
            created_at=now,
        )
        db.add(row)
    except Exception:
        logger.warning(
            "[interactions_v2] فشل بناء صف dual-write لـ user_id=%s event_type=%s — "
            "الكتابة الأساسية تكمل عادي.",
            user_id, event_type, exc_info=True,
        )
