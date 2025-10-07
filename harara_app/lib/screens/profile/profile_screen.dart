import 'package:flutter/material.dart';
import '../../widgets/heat_app_bar.dart';

class ProfileScreen extends StatelessWidget {
  static const route = '/profile';
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // TODO: bind to real user
    return Scaffold(
      appBar: const HeatAppBar(title: "Profile"),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const CircleAvatar(radius: 38, child: Icon(Icons.person, size: 42)),
          const SizedBox(height: 10),
          const Center(child: Text("Madol Abraham", style: TextStyle(fontWeight: FontWeight.w800, fontSize: 18))),
          const SizedBox(height: 20),
          Card(
            child: ListTile(
              leading: const Icon(Icons.email_outlined),
              title: const Text("Email"),
              subtitle: const Text("user@example.com"),
              trailing: IconButton(icon: const Icon(Icons.edit), onPressed: () {}),
            ),
          ),
          Card(
            child: ListTile(
              leading: const Icon(Icons.location_on_outlined),
              title: const Text("Default Town"),
              subtitle: const Text("Juba"),
              trailing: IconButton(icon: const Icon(Icons.edit_location_alt), onPressed: () {}),
            ),
          ),
        ],
      ),
    );
  }
}
