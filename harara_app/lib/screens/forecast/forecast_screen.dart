import 'package:flutter/material.dart';
import '../../widgets/heat_app_bar.dart';

import '../../core/theme/colors.dart';

class ForecastScreen extends StatelessWidget {
  static const route = '/forecast';
  const ForecastScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final forecast = [
      {'day': 'Today', 'temp': '38°C', 'risk': 'High'},
      {'day': 'Tomorrow', 'temp': '40°C', 'risk': 'Extreme'},
      {'day': 'Wed', 'temp': '37°C', 'risk': 'High'},
      {'day': 'Thu', 'temp': '35°C', 'risk': 'Medium'},
      {'day': 'Fri', 'temp': '33°C', 'risk': 'Low'},
    ];

    return Scaffold(
      appBar: const HeatAppBar(title: "7-Day Forecast"),
      body: Column(
        children: [
          Container(
            margin: const EdgeInsets.all(16),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [AppColors.primary, AppColors.primaryLight],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Column(
              children: [
                Text(
                  "AI Prediction Summary",
                  style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700),
                ),
                SizedBox(height: 8),
                Text(
                  "High heatwave risk in 2 days. Plan indoor activities during midday.",
                  style: TextStyle(color: Colors.white70, fontSize: 14),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              itemCount: forecast.length,
              itemBuilder: (context, index) {
                final item = forecast[index];
                return Card(
                  child: ListTile(
                    leading: const Icon(Icons.wb_sunny, color: AppColors.primary),
                    title: Text(item['day']!),
                    subtitle: Text('Risk: ${item['risk']}'),
                    trailing: Text(
                      item['temp']!,
                      style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
