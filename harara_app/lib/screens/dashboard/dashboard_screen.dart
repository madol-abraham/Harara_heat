import 'package:flutter/material.dart';
import '../../models/prediction.dart';
import '../../widgets/probability_bar.dart';
import '../../core/theme/colors.dart';

class DashboardScreen extends StatefulWidget {
  static const route = '/dashboard';
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  // Mock data for demonstration
  final _mockRun = PredictionRun(
    runTs: DateTime.now(),
    startDate: DateTime.now(),
    endDate: DateTime.now().add(const Duration(days: 7)),
    threshold: 0.65,
    predictions: [
      Prediction(town: 'Juba', probability: 0.85, alert: true),
      Prediction(town: 'Wau', probability: 0.45, alert: false),
      Prediction(town: 'Yambio', probability: 0.72, alert: true),
      Prediction(town: 'Bor', probability: 0.38, alert: false),
      Prediction(town: 'Malakal', probability: 0.91, alert: true),
      Prediction(town: 'Bentiu', probability: 0.52, alert: false),
    ],
  );

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.wb_sunny, color: cs.primary, size: 28),
            const SizedBox(width: 8),
            Text(
              'Harara',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.w700,
                color: cs.onSurface,
              ),
            ),
          ],
        ),
        actions: [
          Container(
            margin: const EdgeInsets.only(right: 16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: IconButton(
              tooltip: 'Analytics',
              onPressed: () {},
              icon: Icon(Icons.analytics, color: cs.primary),
            ),
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              cs.background,
              Colors.white.withOpacity(0.8),
            ],
          ),
        ),
        child: ListView(
          padding: EdgeInsets.fromLTRB(20, AppBar().preferredSize.height + 50, 20, 100),
          children: [
            _HeaderCard(run: _mockRun),
            const SizedBox(height: 32),
            Row(
              children: [
                Icon(Icons.location_city, color: cs.onSurface, size: 24),
                const SizedBox(width: 12),
                Text(
                  'Regional Risk Assessment',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: cs.onSurface,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            ..._mockRun.predictions.map((p) => _TownCard(run: _mockRun, p: p)),
          ],
        ),
      ),
      floatingActionButton: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: LinearGradient(
            colors: [cs.primary, cs.secondary],
          ),
          boxShadow: [
            BoxShadow(
              color: cs.primary.withOpacity(0.3),
              blurRadius: 12,
              offset: const Offset(0, 6),
            ),
          ],
        ),
        child: FloatingActionButton.extended(
          backgroundColor: Colors.transparent,
          foregroundColor: Colors.white,
          elevation: 0,
          onPressed: () {
            setState(() {});
          },
          icon: const Icon(Icons.play_arrow_rounded),
          label: Text(
            'Run Prediction',
            style: Theme.of(context).textTheme.labelLarge?.copyWith(
              color: Colors.white,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
    );
  }
}

class _HeaderCard extends StatelessWidget {
  final PredictionRun run;
  const _HeaderCard({required this.run});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final alertCount = run.predictions.where((p) => p.alert).length;
    
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            cs.primary,
            cs.secondary,
          ],
        ),
        boxShadow: [
          BoxShadow(
            color: cs.primary.withOpacity(0.3),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Icon(Icons.thermostat, color: Colors.white, size: 32),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Climate Monitor',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Text(
                      'Heatwave Prediction System',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: Colors.white.withOpacity(0.9),
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.15),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(Icons.date_range, color: Colors.white.withOpacity(0.9), size: 20),
                              const SizedBox(width: 8),
                              Text(
                                'Forecast Period',
                                style: Theme.of(context).textTheme.labelLarge?.copyWith(
                                  color: Colors.white.withOpacity(0.9),
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Text(
                            '${_fmt(run.startDate)} → ${_fmt(run.endDate)}',
                            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                              color: Colors.white,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Column(
                          children: [
                            Text(
                              '$alertCount',
                              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                                color: Colors.white,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            Text(
                              'Active Alerts',
                              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                color: Colors.white.withOpacity(0.9),
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Column(
                          children: [
                            Text(
                              '${(run.threshold * 100).toStringAsFixed(0)}%',
                              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                                color: Colors.white,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            Text(
                              'Alert Threshold',
                              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                color: Colors.white.withOpacity(0.9),
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _fmt(DateTime d) => '${d.day}/${d.month}/${d.year}';
}

class _TownCard extends StatelessWidget {
  final PredictionRun run;
  final Prediction p;
  const _TownCard({required this.run, required this.p});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isHighRisk = p.alert;
    final riskLevel = p.probability >= 0.8 ? 'Extreme' : p.probability >= 0.6 ? 'High' : p.probability >= 0.4 ? 'Moderate' : 'Low';
    
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        color: Colors.white,
        border: isHighRisk ? Border.all(color: cs.error.withOpacity(0.3), width: 2) : null,
        boxShadow: [
          BoxShadow(
            color: isHighRisk ? cs.error.withOpacity(0.1) : Colors.black.withOpacity(0.05),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: () {},
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: isHighRisk ? cs.error.withOpacity(0.1) : cs.primary.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Icon(
                        Icons.location_city,
                        color: isHighRisk ? cs.error : cs.primary,
                        size: 24,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            p.town,
                            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.w600,
                              color: cs.onSurface,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '$riskLevel Risk Level',
                            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: isHighRisk ? cs.error : cs.primary,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: isHighRisk ? cs.error : cs.primary,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        '${(p.probability * 100).toStringAsFixed(0)}%',
                        style: Theme.of(context).textTheme.labelLarge?.copyWith(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                ProbabilityBar(value: p.probability, threshold: run.threshold),
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: isHighRisk 
                        ? cs.error.withOpacity(0.05) 
                        : cs.primary.withOpacity(0.05),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        isHighRisk ? Icons.warning_rounded : Icons.check_circle_rounded,
                        color: isHighRisk ? cs.error : cs.primary,
                        size: 20,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          isHighRisk 
                              ? 'Heatwave risk elevated — take precautions'
                              : 'Normal conditions expected',
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: isHighRisk ? cs.error : cs.primary,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
