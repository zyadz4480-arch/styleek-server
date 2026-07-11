"""
automaton.py — Graph Cellular Automata لمجتمع اهتمامات المستخدم.

هذا الملف يطبّق بالضبط القواعد المتفَّق عليها في المراجعة:
  - عتبة جيران تكيّفية (cold/mature)
  - طاقة تتناقص زمنيًا وتتعزز بالتفاعل + الانتشار عبر الروابط (قفزة واحدة فقط)
  - حالات Alive → Dormant → Dead (لا حذف مباشر)
  - دمج (Merge) بمتوسط موزون بالطاقة، بلا فقدان معلومات (الآباء محفوظون)
  - ولادة عبر أقرب-K-جار (O(n log n) تقريبًا) بدل كل التوافيق الثلاثية O(n³)
  - حد أقصى لعدد الخلايا لكل مستخدم
  - تحديث فوري محلي عند كل تفاعل + دورة ليلية كاملة (موت/ولادة/دمج/انتشار)
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app.ml.automata.config import Config
from app.ml.automata.cell import Cell, CellStatus


class UserInterestGraph:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cells: dict[str, Cell] = {}
        # الروابط: مفتاح = (id_a, id_b) مرتّب أبجديًا لمنع التكرار، قيمة = strength
        self.edges: dict[tuple[str, str], float] = {}
        self.edge_last_reinforced: dict[tuple[str, str], int] = {}

        # مقاييس تراكمية للتقرير النهائي
        self.metrics_log: list[dict] = []
        self.total_births = 0
        self.total_deaths = 0
        self.total_merges = 0

    # ── أدوات مساعدة ──────────────────────────────────────────────────

    def _edge_key(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def _current_neighbor_threshold(self) -> float:
        n_alive = sum(1 for c in self.cells.values() if c.status == CellStatus.ALIVE)
        if n_alive >= self.cfg.maturity_cell_count:
            return self.cfg.neighbor_threshold_mature
        return self.cfg.neighbor_threshold_cold

    def alive_cells(self) -> list[Cell]:
        return [c for c in self.cells.values() if c.status == CellStatus.ALIVE]

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # كلاهما مُطبَّع L2 مسبقًا

    # ── (١) تحديث فوري عند كل إعجاب جديد ─────────────────────────────

    def process_interaction(self, embedding: np.ndarray, day: int) -> None:
        """يُنشئ خلية جديدة مؤقتة، ويعزّز أي خلية موجودة قريبة جدًا بدل تكرارها."""
        embedding = embedding / np.linalg.norm(embedding)
        threshold = self._current_neighbor_threshold()

        # هل فيه خلية قريبة جدًا فعليًا (نفس الاهتمام بالضبط)؟ عزّزها بدل خلية جديدة
        best_match, best_sim = None, -1.0
        for cell in self.alive_cells():
            sim = self._cosine(embedding, cell.embedding)
            if sim > best_sim:
                best_match, best_sim = cell, sim

        if best_match is not None and best_sim >= self.cfg.merge_similarity_threshold:
            best_match.energy += self.cfg.interaction_boost
            best_match.interaction_count += 1
            best_match.confidence = min(1.0, best_match.confidence + 0.05)
            best_match.last_interaction_day = day
            new_cell = best_match
        else:
            new_cell = Cell(
                embedding=embedding,
                energy=self.cfg.new_cell_energy,
                last_interaction_day=day,
            )
            self.cells[new_cell.cell_id] = new_cell

        # تحديث محلي: عزّز الروابط مع كل الجيران القريبين (حسب العتبة الحالية)
        for cell in self.alive_cells():
            if cell.cell_id == new_cell.cell_id:
                continue
            sim = self._cosine(new_cell.embedding, cell.embedding)
            if sim >= threshold:
                key = self._edge_key(new_cell.cell_id, cell.cell_id)
                self.edges[key] = min(1.0, self.edges.get(key, sim) + self.cfg.edge_reinforce_step)
                self.edge_last_reinforced[key] = day

    # ── (٢) الدورة الليلية الكاملة ─────────────────────────────────────

    def nightly_cycle(self, day: int) -> dict:
        self._decay_energy(day)
        self._propagate_energy_through_edges()
        self._update_statuses()
        merges = self._merge_similar_cells(day)
        births = self._birth_new_cells(day)
        self._decay_and_prune_edges(day)
        self._enforce_max_cells()

        alive = [c for c in self.cells.values() if c.status == CellStatus.ALIVE]
        dormant = [c for c in self.cells.values() if c.status == CellStatus.DORMANT]
        dead = [c for c in self.cells.values() if c.status == CellStatus.DEAD]

        snapshot = {
            "day": day,
            "alive": len(alive),
            "dormant": len(dormant),
            "dead": len(dead),
            "total_cells": len(self.cells),
            "edges": len(self.edges),
            "avg_energy_alive": float(np.mean([c.energy for c in alive])) if alive else 0.0,
            "births_today": births,
            "merges_today": merges,
        }
        self.metrics_log.append(snapshot)
        return snapshot

    def _decay_energy(self, day: int) -> None:
        for cell in self.cells.values():
            if cell.status == CellStatus.DEAD:
                continue
            days_idle = day - cell.last_interaction_day
            if days_idle > 0:
                cell.energy = max(0.0, cell.energy - self.cfg.energy_decay_per_day)
            cell.age += 1

    def _propagate_energy_through_edges(self) -> None:
        """قفزة واحدة فقط (جار مباشر). أربع صيغ قابلة للتبديل عبر
        cfg.propagation_mode — الأصلية (naive) غير مستقرة بنيويًا؛ الثلاث
        الأخرى مضمونة الاستقرار رياضيًا. كلها لا تزال 'قفزة واحدة'
        (تعتمد على طاقة الجيران في بداية الدورة فقط، بلا تراكم متعدد القفزات)."""
        mode = getattr(self.cfg, "propagation_mode", "naive")
        live_ids = [cid for cid, c in self.cells.items() if c.status != CellStatus.DEAD]
        if not live_ids:
            return
        idx = {cid: i for i, cid in enumerate(live_ids)}
        n = len(live_ids)
        E = np.array([self.cells[cid].energy for cid in live_ids])

        W = np.zeros((n, n))
        for (a, b), strength in self.edges.items():
            if a in idx and b in idx:
                i, j = idx[a], idx[b]
                W[i, j] = strength
                W[j, i] = strength

        pf = self.cfg.propagation_factor

        if mode == "naive":
            agg = W @ E
            E_new = E + pf * agg

        elif mode == "normalized":
            # decay سبق تطبيقه في _decay_energy قبل هذه الدالة — لا يُكرَّر هنا.
            deg = W.sum(axis=1)
            d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
            W_norm = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
            agg = W_norm @ E
            E_new = E + pf * agg

        elif mode == "random_walk":
            deg = W.sum(axis=1)
            d_inv = np.where(deg > 0, 1.0 / deg, 0.0)
            P = d_inv[:, None] * W
            agg = P @ E
            E_new = E + pf * agg

        elif mode == "laplacian":
            deg = W.sum(axis=1)
            L = np.diag(deg) - W
            lambda_max = 2.0 * deg.max() if deg.max() > 0 else 1.0
            safe_pf = min(pf, 1.9 / lambda_max)
            E_new = E - safe_pf * (L @ E)

        else:
            raise ValueError(f"Unknown propagation_mode: {mode}")

        E_new = np.clip(E_new, 0.0, None)
        for cid, e in zip(live_ids, E_new):
            self.cells[cid].energy = float(e)

    def _update_statuses(self) -> None:
        for cell in self.cells.values():
            if cell.status == CellStatus.DEAD:
                continue
            if cell.energy >= self.cfg.energy_alive_threshold:
                cell.status = CellStatus.ALIVE
                cell.dormant_streak = 0
            elif cell.energy >= self.cfg.energy_dormant_threshold:
                cell.status = CellStatus.DORMANT
                cell.dormant_streak = 0
            else:
                cell.status = CellStatus.DORMANT
                cell.dormant_streak += 1
                if cell.dormant_streak >= self.cfg.dormant_cycles_before_death:
                    cell.status = CellStatus.DEAD
                    self.total_deaths += 1

    def _merge_similar_cells(self, day: int) -> int:
        alive = self.alive_cells()
        merged_ids: set[str] = set()
        merges = 0
        for i in range(len(alive)):
            if alive[i].cell_id in merged_ids:
                continue
            for j in range(i + 1, len(alive)):
                if alive[j].cell_id in merged_ids:
                    continue
                sim = self._cosine(alive[i].embedding, alive[j].embedding)
                if sim >= self.cfg.merge_similarity_threshold:
                    a, b = alive[i], alive[j]
                    total_energy = a.energy + b.energy
                    w_a = a.energy / total_energy if total_energy > 0 else 0.5
                    new_embedding = w_a * a.embedding + (1 - w_a) * b.embedding
                    new_embedding = new_embedding / np.linalg.norm(new_embedding)

                    merged = Cell(
                        embedding=new_embedding,
                        energy=total_energy * self.cfg.merge_energy_retention,
                        age=max(a.age, b.age),
                        confidence=max(a.confidence, b.confidence),
                        interaction_count=a.interaction_count + b.interaction_count,
                        last_interaction_day=max(a.last_interaction_day, b.last_interaction_day),
                        parent_ids=[a.cell_id, b.cell_id],
                        generation=max(a.generation, b.generation) + 1,
                    )
                    self.cells[merged.cell_id] = merged
                    a.status = CellStatus.DEAD  # الأصل لا يُحذف، فقط يصبح ميتًا (المعلومات محفوظة بـparent_ids)
                    b.status = CellStatus.DEAD
                    merged_ids.add(a.cell_id)
                    merged_ids.add(b.cell_id)
                    merges += 1
                    self.total_merges += 1
                    break
        return merges

    def _birth_new_cells(self, day: int) -> int:
        alive = self.alive_cells()
        if len(alive) < self.cfg.birth_min_cluster_size:
            return 0

        embeddings = np.array([c.embedding for c in alive])
        k = min(self.cfg.birth_k_neighbors + 1, len(alive))  # +1 لأن أقرب جار لنفسه = نفسه
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(embeddings)
        _, indices = nn.kneighbors(embeddings)

        births = 0
        seen_centroids: list[np.ndarray] = []
        for i, cell in enumerate(alive):
            neighbor_idxs = [idx for idx in indices[i] if idx != i]
            close_neighbors = [
                alive[idx] for idx in neighbor_idxs
                if self._cosine(cell.embedding, alive[idx].embedding) >= self._current_neighbor_threshold()
            ]
            if len(close_neighbors) < self.cfg.birth_min_cluster_size - 1:
                continue

            group = [cell] + close_neighbors[: self.cfg.birth_min_cluster_size - 1]
            centroid = np.mean([c.embedding for c in group], axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            # هل centroid بعيد فعلًا عن كل الخلايا الحالية *غير الآباء أنفسهم*؟
            # (منطقة "فارغة" تعني بعيدة عن الخلايا الأخرى — ليس عن الآباء الذين
            # centroid هو أصلاً متوسطهم، فهم دائمًا قريبون منه بحكم التعريف)
            group_ids = {c.cell_id for c in group}
            others = [c for cid, c in self.cells.items() if cid not in group_ids]
            max_sim_to_existing = (
                max(self._cosine(centroid, c.embedding) for c in others) if others else -1.0
            )
            if max_sim_to_existing >= self.cfg.birth_centroid_isolation_threshold:
                continue  # ليست منطقة فارغة فعليًا — موجودة أصلًا كخلية قريبة

            # تجنّب ولادة مكررة لنفس المنطقة في نفس الليلة
            if any(self._cosine(centroid, seen) > 0.98 for seen in seen_centroids):
                continue
            seen_centroids.append(centroid)

            avg_parent_energy = float(np.mean([c.energy for c in group]))
            child = Cell(
                embedding=centroid,
                energy=avg_parent_energy * self.cfg.birth_energy_factor,
                last_interaction_day=day,
                parent_ids=[c.cell_id for c in group],
                generation=max(c.generation for c in group) + 1,
                confidence=0.4,  # ثقة أقل — استنتاج ضمني لا تفاعل مباشر
            )
            self.cells[child.cell_id] = child
            for parent in group:
                key = self._edge_key(child.cell_id, parent.cell_id)
                self.edges[key] = self._cosine(centroid, parent.embedding)
                self.edge_last_reinforced[key] = day
            births += 1
            self.total_births += 1

        return births

    def _decay_and_prune_edges(self, day: int) -> None:
        to_delete = []
        for key, strength in self.edges.items():
            days_idle = day - self.edge_last_reinforced.get(key, day)
            new_strength = strength - self.cfg.edge_decay_per_day * max(0, days_idle)
            if new_strength < self.cfg.edge_min_strength:
                to_delete.append(key)
            else:
                self.edges[key] = new_strength
        for key in to_delete:
            del self.edges[key]
            self.edge_last_reinforced.pop(key, None)

    def _enforce_max_cells(self) -> None:
        alive_and_dormant = [c for c in self.cells.values() if c.status != CellStatus.DEAD]
        if len(alive_and_dormant) <= self.cfg.max_cells_per_user:
            return
        # الأضعف طاقة أولًا يُحوَّل لميت قسرًا (soft — البيانات تبقى، فقط تخرج من الاستخدام النشط)
        excess = len(alive_and_dormant) - self.cfg.max_cells_per_user
        weakest = sorted(alive_and_dormant, key=lambda c: c.energy)[:excess]
        for cell in weakest:
            cell.status = CellStatus.DEAD
            self.total_deaths += 1

    # ── [جديد — طبقة إنتاج] تسلسل/فك تسلسل الحالة الكاملة ────────────────
    # إضافية بحتة — لا تلمس nightly_cycle/process_interaction ولا أي دالة
    # تم التحقق منها رياضيًا وعدديًا أعلاه. metrics_log لا يُحفَظ (سجل
    # تشخيصي فقط، تُعاد بناؤه من snapshots السيرفر بدل تضخيم صف قاعدة
    # البيانات — راجع integration_guide.md لمخطط التخزين).
    def export_state(self) -> dict:
        return {
            "cells": [c.to_dict() for c in self.cells.values()],
            "edges": [
                {"a": a, "b": b, "strength": strength,
                 "last_reinforced": self.edge_last_reinforced.get((a, b))}
                for (a, b), strength in self.edges.items()
            ],
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "total_merges": self.total_merges,
        }

    @classmethod
    def load_state(cls, cfg: "Config", state: dict) -> "UserInterestGraph":
        graph = cls(cfg)
        for cd in state.get("cells", []):
            cell = Cell.from_dict(cd)
            graph.cells[cell.cell_id] = cell
        for ed in state.get("edges", []):
            key = graph._edge_key(ed["a"], ed["b"])
            graph.edges[key] = ed["strength"]
            if ed.get("last_reinforced") is not None:
                graph.edge_last_reinforced[key] = ed["last_reinforced"]
        graph.total_births = state.get("total_births", 0)
        graph.total_deaths = state.get("total_deaths", 0)
        graph.total_merges = state.get("total_merges", 0)
        return graph
