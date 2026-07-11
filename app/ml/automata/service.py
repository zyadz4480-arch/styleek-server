"""
app/ml/automata/service.py

طبقة خدمة Graph Cellular Automata — مبنية على automaton.py/cell.py الأصليين
بلا أي تعديل على منطق الطاقة/الدمج/الولادة المُتحقَّق منه رياضيًا مسبقًا
(راجع FINAL_CONFIG.md وproduction_validation.py المرفقين). هذا الملف فقط
يربطها بجداول app/models.py الفعلية (AutomataCell/AutomataEdge/AutomataGraphMeta)
بدل الأسماء الشرطية في automata_service_example.py.

مساحة الـembedding هنا هي 128 بُعد pgvector — نفس فضاء UserEmbedding/
ReelEmbedding المستخدَم فعليًا في app/services/reel_service.py، النظام
الوحيد في هذا المشروع الذي يطابق embedding_dim=128 من config.py. هذا
مختلف تمامًا عن UserStyleProfile.avg_features (22 بُعد، app/ml/features.py)
— لا علاقة بينهما، ولا تُستخدَم avg_features هنا إطلاقًا.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AutomataCell, AutomataEdge, AutomataGraphMeta, ReelInteraction
from app.ml.automata.config import Config
from app.ml.automata.automaton import UserInterestGraph

logger = logging.getLogger("automata_service")

# نسخة إعدادات واحدة مشتركة — قيم فقط، بلا حالة مستخدم بداخلها.
_CFG = Config()
assert _CFG.propagation_mode == "laplacian", (
    "FATAL: propagation_mode ليس laplacian — لا تشغّل الخدمة بهذا الإعداد. "
    "راجع التحذير في app/ml/automata/config.py."
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _logical_day(first_interaction_at: datetime) -> int:
    """اليوم المنطقي = عدد الأيام الكاملة منذ أول تفاعل لهذا المستخدم —
    automaton.py يحتاج فقط عدّادًا صحيحًا متزايدًا، لا تاريخ تقويم فعلي."""
    delta = _utcnow() - first_interaction_at
    return max(0, delta.days)


# ─────────────────────────────────────────────────────────────────────────
# تحميل/حفظ الحالة الكاملة (راجع integration_guide.md §4 لتبرير عدم الـdiff
# الجزئي: max_cells_per_user=150، upsert رخيص جدًا حتى بمعدل استدعاء عالٍ)
# ─────────────────────────────────────────────────────────────────────────

async def _load_graph(db: AsyncSession, user_id: str) -> tuple[UserInterestGraph, int]:
    """يرجع (graph, day). عند فشل فك التسلسل، يسجّل الخطأ ويرجع رسمًا
    بيانيًا فارغًا بدل تعطيل الطلب بالكامل (fallback جزئي، لا يكتب فوق
    البيانات الأصلية لأن الفشل هنا لا يؤدي لاستدعاء _save_graph)."""
    meta = await db.get(AutomataGraphMeta, user_id)
    first_at = meta.first_interaction_at if meta else _utcnow()
    day = _logical_day(first_at)

    try:
        cell_rows = (
            await db.execute(select(AutomataCell).where(AutomataCell.user_id == user_id))
        ).scalars().all()
        edge_rows = (
            await db.execute(select(AutomataEdge).where(AutomataEdge.user_id == user_id))
        ).scalars().all()

        state = {
            "cells": [
                {
                    "cell_id": c.cell_id,
                    "embedding": list(c.embedding),
                    "energy": c.energy,
                    "last_interaction_day": c.last_interaction_day,
                    "age": c.age,
                    "confidence": c.confidence,
                    "status": c.status,
                    "generation": c.generation,
                    "parent_ids": c.parent_ids,
                    "interaction_count": c.interaction_count,
                    "dormant_streak": c.dormant_streak,
                }
                for c in cell_rows
            ],
            "edges": [
                {
                    "a": e.cell_a, "b": e.cell_b, "strength": e.strength,
                    "last_reinforced": e.last_reinforced_day,
                }
                for e in edge_rows
            ],
            "total_births": meta.total_births if meta else 0,
            "total_deaths": meta.total_deaths if meta else 0,
            "total_merges": meta.total_merges if meta else 0,
        }
        return UserInterestGraph.load_state(_CFG, state), day
    except Exception:
        logger.exception(
            f"[automata] فشل تحميل الحالة لـuser_id={user_id} — استخدام رسم فارغ مؤقتًا"
        )
        return UserInterestGraph(_CFG), day


async def _save_graph(db: AsyncSession, user_id: str, graph: UserInterestGraph, day: int) -> None:
    state = graph.export_state()

    await db.execute(delete(AutomataCell).where(AutomataCell.user_id == user_id))
    await db.execute(delete(AutomataEdge).where(AutomataEdge.user_id == user_id))

    db.add_all([
        AutomataCell(
            user_id=user_id, cell_id=c["cell_id"], embedding=c["embedding"],
            energy=c["energy"], status=c["status"],
            last_interaction_day=c["last_interaction_day"], age=c["age"],
            confidence=c["confidence"], generation=c["generation"],
            parent_ids=c["parent_ids"], interaction_count=c["interaction_count"],
            dormant_streak=c["dormant_streak"],
        )
        for c in state["cells"]
    ])
    db.add_all([
        AutomataEdge(
            user_id=user_id, cell_a=e["a"], cell_b=e["b"],
            strength=e["strength"], last_reinforced_day=e["last_reinforced"],
        )
        for e in state["edges"]
    ])

    existing_meta = await db.get(AutomataGraphMeta, user_id)
    if existing_meta is None:
        db.add(AutomataGraphMeta(
            user_id=user_id, first_interaction_at=_utcnow(), current_day=day,
            total_births=state["total_births"], total_deaths=state["total_deaths"],
            total_merges=state["total_merges"], last_nightly_run_at=None,
        ))
    else:
        existing_meta.current_day = day
        existing_meta.total_births = state["total_births"]
        existing_meta.total_deaths = state["total_deaths"]
        existing_meta.total_merges = state["total_merges"]

    await db.commit()


# ─────────────────────────────────────────────────────────────────────────
# الـAPI العام (راجع integration_guide.md §2)
# ─────────────────────────────────────────────────────────────────────────

async def record_interaction(db: AsyncSession, user_id: str, embedding: np.ndarray) -> None:
    """Fire-and-forget — تُستدعى من app/services/reel_service.py بعد
    taste_profile.update_from_reel، بنفس embedding الريل (128 بُعد،
    ReelEmbedding). فشلها لا يُفشل استجابة المستخدم أبدًا — avg_features
    استُدعي بالفعل بشكل مستقل قبل هذا."""
    try:
        graph, day = await _load_graph(db, user_id)
        graph.process_interaction(embedding, day)
        await _save_graph(db, user_id, graph, day)
    except Exception:
        logger.exception(f"[automata] فشل record_interaction لـuser_id={user_id}")


async def nightly_cycle_for_user(db: AsyncSession, user_id: str) -> dict:
    graph, day = await _load_graph(db, user_id)
    snapshot = graph.nightly_cycle(day)
    await _save_graph(db, user_id, graph, day)
    return snapshot


async def users_with_interaction_last_24h(db: AsyncSession) -> list[str]:
    """يُستخدَم من POST /internal/automata/nightly — نفس جدول
    ReelInteraction الموجود مسبقًا، لا حاجة لعمود إضافي."""
    since = _utcnow() - timedelta(hours=24)
    rows = (
        await db.execute(
            select(ReelInteraction.user_id)
            .where(ReelInteraction.created_at >= since)
            .distinct()
        )
    ).scalars().all()
    return list(rows)


async def get_automata_boost(
    db: AsyncSession, user_id: str, candidate_embeddings: np.ndarray
) -> np.ndarray | None:
    """يرجع None صراحة (لا مصفوفة أصفار) عند عدم وجود خلايا حية — هذا هو
    إشارة الـFallback لمستدعيها (rerank_reels_by_embedding)."""
    graph, _ = await _load_graph(db, user_id)
    alive = graph.alive_cells()
    if not alive:
        return None
    cell_matrix = np.array([c.embedding for c in alive])
    similarities = candidate_embeddings @ cell_matrix.T
    return similarities.max(axis=1)
