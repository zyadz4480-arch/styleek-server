# ملخص التعديل — حل مشكلة "الريل الجديد بلا Embedding"

## الملفات وأماكنها في مشروعك

| الملف المرفق | يستبدل |
|---|---|
| `taste_profile.py` | `app/ml/taste_profile.py` (استبدال كامل) |
| `cold_start.py` | `app/ml/cold_start.py` (استبدال كامل) |
| `embedding_repository.py` | **جديد** → `app/ml/embedding_repository.py` |
| `models.py` | `app/models.py` (استبدال كامل) |
| `reel_service.py` | `app/services/reel_service.py` (استبدال كامل) |
| `reels.py` | `app/routers/reels.py` (استبدال كامل) |
| `schemas.py` | `app/schemas.py` (استبدال كامل) |
| `migration_add_embedding_source.sql` | يُشغَّل يدويًا مرة واحدة على PostgreSQL |

## خطوات التطبيق بالترتيب

1. **قاعدة البيانات أولاً:** شغّل `migration_add_embedding_source.sql` على PostgreSQL قبل رفع أي كود جديد.
2. ارفع/استبدل الملفات السبعة أعلاه في مشروعك (كلها استبدال كامل 1:1 لملفاتك الحالية، عدا `embedding_repository.py` الذي هو ملف جديد).
3. **مهم — تعديل مطلوب في Flutter (main.dart):**
   في `PexelsReelService._applyServerRerank()`، الطلب الحالي يرسل:
   ```dart
   body: jsonEncode({'user_id': userId, 'reel_ids': ids}),
   ```
   يجب أن يصبح:
   ```dart
   body: jsonEncode({
     'user_id': userId,
     'reels': locallyRanked.map((r) => {
       'reel_id': r.id,
       'outfit_style': r.outfitStyle.toString(), // أو الحقل النصي المطابق عندك
       'dominant_color': r.colors.isNotEmpty ? r.colors.first : null,
     }).toList(),
   }),
   ```
   بدون هذا التعديل، السيرفر لن يستقبل `outfit_style`/`dominant_color`، وسيقع cold-start دائمًا على القيم الافتراضية المحايدة — يعني التعديل الخلفي كله بلا فائدة عملية حتى تُطبَّق هذه الخطوة في Flutter أيضًا.
5. أعد تشغيل السيرفر على Render.

## السلوك المتوقَّع بعد التطبيق
- كل ريل جديد يصل من Pexels لأول مرة يحصل على `ReelEmbedding` فوري (`embedding_source='cold_start'`) بدل تجاهله من الترتيب.
- لاحقًا، `export_embeddings.py` (بعد تدريب Two-Tower فعلي) يستبدل هذه القيم تلقائيًا بـ`embedding_source='trained'` الأدق.
- **تنبيه سلوكي:** بما إن كل الريلز الآن تُرتَّب فعليًا (لا "تمريرة" بدون ترتيب)، رح تلاحظ تغيّر ملموس في ترتيب الريلز من أول استخدام بعد النشر — هذا متوقَّع ومقصود.
