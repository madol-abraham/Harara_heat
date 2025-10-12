import 'dart:math';
import 'dart:async';

/// ---------------------------------------------------------------------------
/// MockApiService - Simulates backend predictions for demo purposes
/// ---------------------------------------------------------------------------
class MockApiService {
  static final _rand = Random();

  /// Simulate a single live prediction
  static Future<Map<String, dynamic>> getPrediction(String town) async {
    await Future.delayed(const Duration(seconds: 1)); // simulate delay

    // Random values like your LSTM/ML model output
    final temp = 35 + _rand.nextDouble() * 10; // 35–45°C
    final risk = _getRisk(temp);
    final message = switch (risk) {
      "Extreme" => "Heatwave detected! Avoid outdoor activity 12–3 PM.",
      "Warning" => "High temperature warning. Stay hydrated.",
      _ => "Normal conditions. Stay safe."
    };

    return {
      "town": town,
      "temperature": temp.toStringAsFixed(1),
      "risk": risk,
      "message": message,
      "timestamp": DateTime.now().toIso8601String(),
    };
  }

  /// Simulate a 7-day forecast
  static Future<List<Map<String, dynamic>>> getForecast(String town) async {
    await Future.delayed(const Duration(seconds: 1));
    final now = DateTime.now();
    return List.generate(7, (i) {
      final temp = 34 + _rand.nextDouble() * 8;
      return {
        "day": now.add(Duration(days: i)).toIso8601String().split("T").first,
        "temperature": temp.toStringAsFixed(1),
      };
    });
  }

  static String _getRisk(double t) {
    if (t >= 40) return "Extreme";
    if (t >= 37) return "Warning";
    return "Safe";
    }
}
