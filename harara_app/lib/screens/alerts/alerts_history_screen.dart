import 'package:flutter/material.dart';
import '../../widgets/heat_app_bar.dart';
import '../../widgets/empty_state.dart';
import '../../widgets/heat_level_chip.dart';
import '../../core/theme/utils/format.dart';

class AlertsHistoryScreen extends StatelessWidget {
  static const route = '/alerts';
  const AlertsHistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // TODO: bind to backend
    final items = [
      ("Juba", HeatLevel.extreme, DateTime.now().toIso8601String()),
      ("Wau", HeatLevel.warning, DateTime.now().subtract(const Duration(days: 1)).toIso8601String()),
    ];

    return Scaffold(
      appBar: const HeatAppBar(title: "Alerts History"),
      body: items.isEmpty
        ? const EmptyState(title: "No alerts yet", subtitle: "You’ll see previous alerts here.")
        : ListView.separated(
            itemCount: items.length,
            separatorBuilder: (_, __) => const Divider(height: 0),
            itemBuilder: (_, i) {
              final (town, level, ts) = items[i];
              return ListTile(
                leading: const Icon(Icons.warning_amber_rounded),
                title: Text("$town — ${_label(level)}"),
                subtitle: Text(Fmt.date(ts)),
                trailing: HeatLevelChip(level),
              );
            },
          ),
    );
  }

  static String _label(HeatLevel l) =>
      l == HeatLevel.extreme ? "Extreme Heat" : (l == HeatLevel.warning ? "Warning" : "Safe");
}
