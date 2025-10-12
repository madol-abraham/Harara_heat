import 'package:flutter/material.dart';

class ProbabilityChip extends StatelessWidget {
  final double probability;
  final bool isAlert;

  const ProbabilityChip({
    super.key,
    required this.probability,
    required this.isAlert,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: isAlert ? cs.error : cs.primary,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        '${(probability * 100).toStringAsFixed(0)}%',
        style: Theme.of(context).textTheme.labelMedium?.copyWith(
          color: Colors.white,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}