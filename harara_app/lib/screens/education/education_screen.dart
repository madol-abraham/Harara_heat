import 'package:flutter/material.dart';
import '../../widgets/heat_app_bar.dart';
import '../../core/theme/colors.dart';

class EducationScreen extends StatelessWidget {
  static const route = '/education';
  const EducationScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const HeatAppBar(title: "Understanding Heatwaves"),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              AppColors.bg,
              Colors.white.withOpacity(0.8),
            ],
          ),
        ),
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _Card(
              title: "Abnormal Heat: High Pressure Dome",
              body: "Warm air gets trapped near the ground by a high-pressure zone. "
                  "Less vertical mixing keeps heat close to the surface.",
            ),
            _Card(
              title: "Safety Basics",
              body: "• Hydrate every 30–45 min\n• Avoid outdoor work 12–15h\n"
                  "• Wear light clothing\n• Check the elderly and infants",
            ),
            _Card(
              title: "Symptoms of Heat Stress",
              body: "Headache • Dizziness • Nausea • Rapid pulse • Confusion. "
                    "Seek shade, cool water, and medical help if severe.",
            ),
          ],
        ),
      ),
    );
  }
}

class _Card extends StatelessWidget {
  final String title, body;
  const _Card({required this.title, required this.body});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.white.withOpacity(0.98),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 22, color: AppColors.primary)),
          const SizedBox(height: 12),
          Text(body, style: const TextStyle(fontSize: 16, height: 1.5)),
        ]),
      ),
    );
  }
}
