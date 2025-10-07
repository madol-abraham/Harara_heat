import 'package:flutter/material.dart';
import '../core/theme/colors.dart';

enum HeatLevel { safe, warning, extreme }

class HeatLevelChip extends StatelessWidget {
  final HeatLevel level;
  const HeatLevelChip(this.level, {super.key});

  @override
  Widget build(BuildContext context) {
    String label;
    Color color;
    switch (level) {
      case HeatLevel.safe:
        label = "SAFE"; color = AppColors.success; break;
      case HeatLevel.warning:
        label = "WARNING"; color = AppColors.amber; break;
      case HeatLevel.extreme:
        label = "EXTREME HEAT"; color = AppColors.danger; break;
    }
    return Chip(
      label: Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700)),
      backgroundColor: color,
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 0),
    );
  }
}
