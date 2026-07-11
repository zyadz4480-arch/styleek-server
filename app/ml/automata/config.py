"""
config.py — كل معامل قابل للتجربة/التعديل في مكان واحد.
لا قيم مضروبة يدويًا (hardcoded) داخل منطق المحاكاة نفسه — أي تجربة
جديدة (تشديد/تخفيف قواعد الحياة، الموت، الدمج، الولادة) تُجرى من هنا فقط.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ── فضاء الـEmbedding ──────────────────────────────────────────────
    embedding_dim: int = 128          # نفس EMBEDDING_DIM في cold_start.py الإنتاجي

    # ── عتبة الجيران (تكيّفية) ─────────────────────────────────────────
    neighbor_threshold_cold: float = 0.80      # مستخدم جديد (خلايا قليلة)
    neighbor_threshold_mature: float = 0.90    # مستخدم ناضج (خلايا كثيرة)
    maturity_cell_count: int = 20              # عتبة عدد الخلايا للتحوّل من cold→mature

    # ── الطاقة ─────────────────────────────────────────────────────────
    initial_energy: float = 1.0
    interaction_boost: float = 0.6             # طاقة تُضاف عند تفاعل مباشر مع خلية موجودة
    new_cell_energy: float = 0.8                # طاقة ابتدائية لخلية جديدة من إعجاب
    energy_decay_per_day: float = 0.05          # تناقص طاقة يومي (بلا تفاعل)
    propagation_factor: float = 0.15            # نسبة الطاقة المنقولة عبر رابط لكل قفزة واحدة فقط

    # ⚠️ [إصلاح إنتاجي حرج] كان الافتراضي هنا "naive" — وهو النمط الوحيد
    # المُثبَت رياضيًا وعدديًا أنه يخلق طاقة من العدم وينفجر (راجع
    # math_verification_results.csv: naive ينفجر لـ~1.18e30 وينهار عند
    # اليوم 124-130 من أصل 180). كل التحقق السابق (استقرار/تنوع/جودة
    # توصية/seed sweep) استخدم "laplacian" فقط عبر تجاوز صريح لهذا الحقل
    # — الافتراضي الفعلي نفسه لم يكن مُختبَرًا قط. لا تُرجعه لـ"naive"
    # بدون تكرار math_verification.py والتأكد من نفس النتيجة.
    propagation_mode: str = "laplacian"          # "naive" | "normalized" | "random_walk" | "laplacian"


    # ── حالات الحياة ───────────────────────────────────────────────────
    energy_alive_threshold: float = 0.35
    energy_dormant_threshold: float = 0.12      # تحت هذا = تصبح مرشحة للموت
    dormant_cycles_before_death: int = 5        # عدد الدورات الليلية المتتالية تحت العتبة قبل death

    # ── الدمج (Merge) ──────────────────────────────────────────────────
    merge_similarity_threshold: float = 0.97
    merge_energy_retention: float = 0.9         # فقد بسيط عند الدمج

    # ── الروابط (Edges) ────────────────────────────────────────────────
    edge_reinforce_step: float = 0.1
    edge_decay_per_day: float = 0.03
    edge_min_strength: float = 0.05             # تحت هذا → الرابط يُحذف

    # ── الولادة (Birth) ────────────────────────────────────────────────
    birth_k_neighbors: int = 6                  # نبحث ضمن أقرب K جار فقط (O(n log n) لا O(n^3))
    birth_min_cluster_size: int = 3             # عدد الجيران المتقاربين اللازم لاعتبارها "منطقة كثيفة"
    birth_centroid_isolation_threshold: float = 0.85  # centroid يجب أن يكون بعيدًا عن كل الخلايا الحالية بأكثر من هذا
    birth_energy_factor: float = 0.5            # طاقة الوليد = متوسط طاقة الآباء × هذا العامل

    # ── الحد الأقصى للخلايا ────────────────────────────────────────────
    max_cells_per_user: int = 150
    min_cells_before_pruning_relaxed: int = 50  # تحت هذا العدد، لا نفرض تقليم صارم

    # ── محرك التوصية الهجين ────────────────────────────────────────────
    # هذه القيم (0.6/0.4) هي المُختبَرة فعليًا في comparison_results.json
    # وrunning_mean_comparison (تحقّق إضافي أُجري لاحقًا). طُلب لاحقًا
    # استخدام 0.7/0.3 بدلًا منها — هذه النسبة تحديدًا لم تُختبَر بعد. إن
    # غيّرتها، الأولى تشغيل experiment_compare.py مرة واحدة على القيمة
    # الجديدة قبل الاعتماد عليها في الإنتاج.
    hybrid_avg_features_weight: float = 0.6
    hybrid_cell_weight: float = 0.4

    # ── محاكاة البيانات الاصطناعية ─────────────────────────────────────
    n_true_interests: int = 4
    simulation_days: int = 40
    min_likes_per_day: int = 1
    max_likes_per_day: int = 5
    genuine_interest_ratio: float = 0.7         # نسبة الإعجابات القريبة من اهتمام حقيقي متكرر
    cluster_noise_std: float = 0.03            # 0.12 كانت تنتج تشابهًا ~0.35 فقط بين عيّنات نفس الاهتمام
                                                 # في 128 بُعد (لعنة الأبعاد العالية) — 0.03 تعطي ~0.88-0.90،
                                                 # وهو نطاق واقعي لعيّنات "نفس الاهتمام" فعليًا (تأكَّد تجريبيًا،
                                                 # انظر ملاحظة المعايرة في report.md)
    random_seed: int = 42
