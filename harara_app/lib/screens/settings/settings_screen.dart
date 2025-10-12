import 'package:flutter/material.dart';
import '../../widgets/heat_app_bar.dart';

class SettingsScreen extends StatefulWidget {
  static const route = '/settings';
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool sms = true;
  bool push = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const HeatAppBar(title: "Settings"),
      body: ListView(
        children: [
          SwitchListTile(
            value: push, onChanged: (v) => setState(() => push = v),
            title: const Text("Push Notifications")),
          SwitchListTile(
            value: sms, onChanged: (v) => setState(() => sms = v),
            title: const Text("SMS Alerts")),
          ListTile(
            title: const Text("Manage Towns / Locations"),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {}, // open picker
          ),
          ListTile(
            title: const Text("Privacy & Terms"),
            onTap: () {},
          ),
        ],
      ),
    );
  }
}
