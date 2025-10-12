import 'package:flutter/material.dart';
import '../core/theme/colors.dart';

/// ---------------------------------------------------------------------------
/// DrawerMenu - Sidebar navigation for Profile, About, Support, and Admin Panel
/// ---------------------------------------------------------------------------
class DrawerMenu extends StatelessWidget {
  const DrawerMenu({super.key});

  @override
  Widget build(BuildContext context) {
    return Drawer(
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.only(
          topRight: Radius.circular(20),
          bottomRight: Radius.circular(20),
        ),
      ),
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          // Drawer Header with gradient background
          DrawerHeader(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [AppColors.primary, AppColors.secondary],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                CircleAvatar(
                  radius: 28,
                  backgroundColor: Colors.white,
                  child: Icon(Icons.person, color: AppColors.primary, size: 36),
                ),
                SizedBox(height: 10),
                Text("Madol Abraham",
                    style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w800,
                        fontSize: 18)),
                Text("South Sudan", style: TextStyle(color: Colors.white70)),
              ],
            ),
          ),

          // Drawer links
          _DrawerTile(
            icon: Icons.account_circle_outlined,
            title: "Profile",
            route: '/profile',
          ),
          _DrawerTile(
            icon: Icons.info_outline,
            title: "About Harara",
            route: '/about',
          ),
          _DrawerTile(
            icon: Icons.chat_bubble_outline,
            title: "Support & Feedback",
            route: '/support',
          ),
          _DrawerTile(
            icon: Icons.admin_panel_settings_outlined,
            title: "Admin Panel",
            route: '/admin',
          ),
          const Divider(),
          _DrawerTile(
            icon: Icons.logout,
            title: "Log Out",
            route: '/onboarding', // navigate back to onboarding (for demo)
            color: AppColors.danger,
          ),
        ],
      ),
    );
  }
}

class _DrawerTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String route;
  final Color? color;
  const _DrawerTile({
    required this.icon,
    required this.title,
    required this.route,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Icon(icon, color: color ?? AppColors.primary),
      title: Text(title,
          style: TextStyle(
              color: color ?? AppColors.text,
              fontWeight: FontWeight.w600,
              fontSize: 15)),
      onTap: () {
        Navigator.pop(context); // close drawer first
        Navigator.pushNamed(context, route);
      },
    );
  }
}
