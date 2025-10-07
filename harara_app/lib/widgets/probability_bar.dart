import 'package:flutter/material.dart';

class ProbabilityBar extends StatelessWidget {
  final double value;
  final double threshold;

  const ProbabilityBar({
    super.key,
    required this.value,
    required this.threshold,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isHighRisk = value >= threshold;
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Risk Probability',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w500,
                color: cs.onSurface.withOpacity(0.7),
              ),
            ),
            Text(
              '${(value * 100).toStringAsFixed(0)}%',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
                color: isHighRisk ? cs.error : cs.primary,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Container(
          height: 8,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(4),
            color: Colors.grey.withOpacity(0.2),
          ),
          child: Stack(
            children: [
              FractionallySizedBox(
                widthFactor: value,
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(4),
                    gradient: LinearGradient(
                      colors: isHighRisk 
                          ? [cs.error.withOpacity(0.7), cs.error]
                          : [cs.primary.withOpacity(0.7), cs.primary],
                    ),
                  ),
                ),
              ),
              // Threshold indicator
              Positioned(
                left: threshold * MediaQuery.of(context).size.width * 0.85,
                child: Container(
                  width: 2,
                  height: 8,
                  color: Colors.black54,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}