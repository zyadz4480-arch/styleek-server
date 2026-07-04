"""
app/constants.py

ثوابت مشتركة بين app/models.py و app/ml/* — عمدًا **بدون أي استيراد من
داخل حزمة app نفسها** (وحدة "ورقة" leaf module). هذا يضمن أن استيراد
أي شيء من هذا الملف لا يُشغّل app/ml/__init__.py إطلاقًا (الملف يعيش
خارج حزمة app.ml تمامًا)، وبالتالي لا يوجد أي احتمال Circular Import
عند استيراده من app/models.py — بغضّ النظر عن محتوى app/ml/__init__.py
(اللي غير متوفر للمراجعة وقت كتابة هذا الملف).

القائمتان أدناه كانتا مُعرَّفتين محليًا داخل app/ml/features.py، ونُقلتا
هنا ليصيرا مصدرًا وحيدًا يستورد منه كل من features.py و models.py —
بدل تكرار نفس القيم في مكانين (تحذير #4 في HANDOFF.md: أي قيمة/منطق
حساب يعيش في مكان واحد فقط ويُستورَد، لا يُعاد كتابته).

⚠️ الترتيب هنا حسّاس بعد أول تدريب فعلي (نفس تحذير features.py الأصلي
بخصوص CATEGORY_LIST/OCCASION_LIST) — إضافة عنصر جديد بالنهاية فقط،
لا تُغيَّر مواضع العناصر الحالية ولا تُحذف.
"""

from __future__ import annotations

CATEGORY_LIST = ["shirt", "tshirt", "hoodie", "pants", "jeans",
                  "jacket", "shoes", "boots", "accessory", "other"]
OCCASION_LIST = ["work", "university", "outing", "special", "sport"]  # + 5 = بلا مناسبة

# الديفولت المحايد لمتجه ميزات كامل (FEATURE_DIM=22 في app/ml/features.py):
# 0.5 لكل الأبعاد المستمرة (أول 20 بُعد) + فئة "other" ومناسبة "بلا مناسبة"
# للبُعدين الأخيرين — مُشتقّان برمجيًا من القائمتين أعلاه بدل رقمين ثابتين
# منسوخين يدويًا (كانا سابقًا 9.0 و5.0 مكتوبين حرفيًا في models.py).
DEFAULT_FEATURE_VECTOR: list[float] = [0.5] * 20 + [
    float(CATEGORY_LIST.index("other")),
    float(len(OCCASION_LIST)),
]
