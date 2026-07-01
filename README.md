# Styleek AI Server — قلب ستايلك المفصول

سيرفر FastAPI + PostgreSQL + scikit-learn يستضيف محرك الذكاء الاصطناعي
الذي كان يعمل محليًا داخل `main.dart` (كلاس `StyleMLOrchestrator`).

## ما الذي تم نقله فعليًا (المرحلة ١ — جاهزة وتعمل)

| الأصل في main.dart | المعادل الجديد | التغيير |
|---|---|---|
| `MLFeatureExtractor` (سطر 2931) | `app/ml/features.py` | نفس المنطق حرفيًا، Python بدل Dart |
| `LogisticRegressionModel` (يدوي، SGD) | `sklearn.LogisticRegression` | خوارزمية حقيقية مُحسَّنة |
| `LinearRegressionModel` (يدوي، GD+L2) | `sklearn.Ridge` | خوارزمية حقيقية |
| `LinearSVMModel` (يدوي، Pegasos) | `sklearn.LinearSVC` + معايرة Platt | خوارزمية حقيقية |
| `DecisionTreeModel` (يدوي، CART) | `sklearn.DecisionTreeClassifier` | خوارزمية حقيقية |
| `RandomForestModel` (يدوي، Bagging) | `sklearn.RandomForestClassifier` | خوارزمية حقيقية |
| `StyleMLOrchestrator` (الدمج + أوزان الخبراء) | `app/ml/ensemble.py` + `app/ml/service.py` | **نفس معادلة softmax بالضبط** |
| `MLDataset` (SharedPreferences، 200 عيّنة) | جدول `interactions` في PostgreSQL | تخزين دائم غير محدود، لكل المستخدمين |
| `StyleTrainingScheduler` | `maybe_autotrain()` | نفس فكرة "تدريب كل 25 تفاعل" |

**كل الأوزان والعتبات والصيغ الرياضية منسوخة رقميًا كما هي** (مثال: أوزان
الفئات، أوزان الخبراء الابتدائية `[1.00, 0.82, 0.69, 0.69, 1.20]`، معدل
التعلّم `0.05`، عتبات `0.75/0.55/0.35`) — راجع التعليقات في كل ملف، كل
واحدة تشير لرقم السطر المطابق في `main.dart` الأصلي.

## ما لم يُنقل بعد (المرحلة ٢ — يحتاج جلسة عمل منفصلة)

هذه الأنظمة معزولة بذاتها في `main.dart` ويمكن نقلها لاحقًا بنفس النمط:

- `NeuralCollaborativeFilter` + `GruSequenceModel` + `NeuralScoreBlender` (سطر 21282+) — تعلّم جماعي عبر المستخدمين
- `StylePersonaKMeans` / `ColorAffinityDBSCAN` / `StylePCA` / `LightAutoencoder` (سطر 6036+) — تحليل غير مُشرف
- `LocalClothingAnalyzer` (سطر 903) — **يُفضَّل إبقاؤه على الجهاز** (يعمل بدون إنترنت ويحلل الصور فور التقاطها)
- `SessionBasedRecommender` / `ItemGraph` / `LearningToRankEngine` (نهاية الملف) — إعادة ترتيب متقدّمة

## التشغيل محليًا

```bash
cp .env.example .env   # عدّل API_KEY لقيمة حقيقية
docker compose up --build
```

السيرفر يعمل على `http://localhost:8000`، والتوثيق التفاعلي على
`http://localhost:8000/docs`.

## أين تُخزَّن البيانات؟

قاعدة PostgreSQL جديدة مستقلة (`styleek_ml`) تخزّن فقط بيانات التدريب
(التفاعلات + النماذج) — **ليست بديلاً عن Firestore**. أبقِ Firestore كما
هو لبيانات الخزانة/المستخدمين الحيّة، وأرسل نسخة من كل تفاعل (قبول/تخطي
إطلالة) لهذا السيرفر بالتوازي. لاحقًا يمكن ربط مصادقة السيرفر بنفس
Firebase Auth (التحقق من الـ ID Token بدل مفتاح API الثابت الحالي).

## دمج السيرفر في تطبيق Flutter

استخدم `flutter_client/styleek_api_client.dart` — انسخه إلى مشروع
Flutter، غيّر `baseUrl` و `apiKey`، ثم استبدل استدعاءات
`mlOrchestrator.record(...)` و `mlOrchestrator.predict(...)` باستدعاءات
`StyleekApiClient.recordInteraction(...)` و `StyleekApiClient.predict(...)`.

## اختبار سريع بعد التشغيل

```bash
curl -X POST http://localhost:8000/style/interactions \
  -H "X-API-Key: CHANGE_ME_SECRET_KEY" -H "Content-Type: application/json" \
  -d '{"user_id":"u1","accepted":true,"item":{"category_name":"jacket","colors":["#1A237E"],"season_name":"winter","current_season":"winter","occasion_name":"work","wear_count":3,"is_favorite":true,"is_layerable":true,"dna_formal":0.7,"dna_casual":0.4}}'
```

## ملاحظة مهمة حول الأمان

`X-API-Key` الحالي بسيط جدًا ومناسب فقط للتطوير. قبل نشر السيرفر فعليًا
للإنتاج، استبدله بتحقق حقيقي من Firebase ID Token (موجود أصلًا في
`main.dart` عبر `firebase_auth`) حتى لا يتمكن أي شخص من إرسال بيانات
تدريب مزيّفة باسم مستخدم آخر.
