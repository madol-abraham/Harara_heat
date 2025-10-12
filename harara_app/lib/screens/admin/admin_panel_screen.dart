import 'package:flutter/material.dart';
import '../../widgets/heat_app_bar.dart';

class AdminPanelScreen extends StatelessWidget {
  static const route = '/admin';
  const AdminPanelScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // Guard with auth later
    return Scaffold(
      appBar: const HeatAppBar(title: "Admin Panel"),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _Tile(icon: Icons.sync, title: "Force Data Refresh", onTap: () {}),
          _Tile(icon: Icons.sms, title: "Send Test SMS", onTap: () {}),
          _Tile(icon: Icons.rule, title: "Thresholds & Rules", onTap: () {}),
          _Tile(icon: Icons.people, title: "Manage Users", onTap: () {}),
        ],
      ),
    );
  }
}

class _Tile extends StatelessWidget {
  final IconData icon; final String title; final VoidCallback onTap;
  const _Tile({required this.icon, required this.title, required this.onTap});
  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(leading: Icon(icon), title: Text(title), trailing: const Icon(Icons.chevron_right), onTap: onTap),
    );
  }
}
