// ============================================================================
// StyleekApiClient — يستبدل الاتصال المباشر بـ StyleMLOrchestrator المحلي
// (main.dart سطر 4204) بمكالمات HTTP لسيرفرك الجديد.
//
// طريقة الاستخدام في main.dart:
//   بدل: final prediction = mlOrchestrator.predict(features);
//   استخدم: final prediction = await StyleekApiClient.predict(userId: uid, item: item, ...);
//
//   بدل: mlOrchestrator.record(features: f, accepted: true);
//   استخدم: await StyleekApiClient.recordInteraction(userId: uid, item: item, accepted: true);
// ============================================================================
import 'dart:convert';
import 'package:http/http.dart' as http;

class StyleekApiClient {
  // غيّر هذا لعنوان سيرفرك الفعلي بعد النشر (مثال: https://ai.styleek.app)
  static const String baseUrl = 'http://YOUR_SERVER_ADDRESS:8000';
  static const String apiKey  = 'CHANGE_ME_SECRET_KEY'; // طابق app/config.py

  static Map<String, String> get _headers => {
    'Content-Type': 'application/json',
    'X-API-Key': apiKey,
  };

  /// يعادل استدعاء orchestrator.record() في main.dart
  static Future<void> recordInteraction({
    required String userId,
    required Map<String, dynamic> item, // انظر _itemContext أدناه
    required bool accepted,
    double? rating,
    String? itemId,
  }) async {
    final res = await http.post(
      Uri.parse('$baseUrl/style/interactions'),
      headers: _headers,
      body: jsonEncode({
        'user_id': userId,
        'item': item,
        'accepted': accepted,
        'rating': rating,
        'item_id': itemId,
      }),
    );
    if (res.statusCode != 201) {
      throw Exception('فشل تسجيل التفاعل: ${res.statusCode} ${res.body}');
    }
  }

  /// يعادل استدعاء orchestrator.predict() في main.dart
  static Future<Map<String, dynamic>> predict({
    required String userId,
    required Map<String, dynamic> item,
  }) async {
    final res = await http.post(
      Uri.parse('$baseUrl/style/predict?user_id=$userId'),
      headers: _headers,
      body: jsonEncode(item),
    );
    if (res.statusCode != 200) {
      throw Exception('فشل التنبؤ: ${res.statusCode} ${res.body}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  /// تدريب فوري يدوي (عادة غير ضروري لأن السيرفر يُدرِّب تلقائيًا كل 25 تفاعل)
  static Future<Map<String, dynamic>> trainNow(String userId) async {
    final res = await http.post(
      Uri.parse('$baseUrl/style/train/$userId'),
      headers: _headers,
    );
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  static Future<Map<String, dynamic>> getSummary(String userId) async {
    final res = await http.get(
      Uri.parse('$baseUrl/style/summary/$userId'),
      headers: _headers,
    );
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  /// دالة مساعدة لبناء جسم "item" المطلوب من ClothingItem الأصلي في main.dart
  static Map<String, dynamic> buildItemContext({
    required String categoryName,
    required List<String> colors,
    required String seasonName,
    required String currentSeason,
    String? occasionName,
    int? temperature,
    required int wearCount,
    required bool isFavorite,
    DateTime? lastWornAt,
    String? brand,
    required bool isLayerable,
    required double dnaFormal,
    required double dnaCasual,
  }) => {
    'category_name': categoryName,
    'colors': colors,
    'season_name': seasonName,
    'current_season': currentSeason,
    'occasion_name': occasionName,
    'temperature': temperature,
    'wear_count': wearCount,
    'is_favorite': isFavorite,
    'last_worn_at': lastWornAt?.toIso8601String(),
    'brand': brand,
    'is_layerable': isLayerable,
    'dna_formal': dnaFormal,
    'dna_casual': dnaCasual,
  };
}
