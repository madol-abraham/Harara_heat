import 'package:flutter/material.dart';
import '../core/theme/colors.dart';

class GradientCard extends StatelessWidget {
  final Widget child;
  final EdgeInsets padding;
  const GradientCard({super.key, required this.child, this.padding = const EdgeInsets.all(16)});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      padding: padding,
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [AppColors.primary, AppColors.secondary],
          begin: Alignment.topLeft, end: Alignment.bottomRight),
        borderRadius: BorderRadius.circular(22),
        boxShadow: const [BoxShadow(
          color: AppColors.cardShadow, blurRadius: 12, offset: Offset(0,6))],
      ),
      child: child,
    );
  }
}
