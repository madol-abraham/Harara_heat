import 'package:flutter/material.dart';
import '../core/theme/colors.dart';
import '../screens/dashboard/dashboard_screen.dart';
import '../screens/forecast/forecast_screen.dart';
import '../screens/alerts/alerts_history_screen.dart';
import '../screens/education/education_screen.dart';
import '../screens/settings/settings_screen.dart';
import '../widgets/drawer_menu.dart';

/// ---------------------------------------------------------------------------
///  BottomNavShell - Main navigation container for Harara app
///  Controls tab switching between Dashboard, Forecast, Alerts, Education, Settings
///  + integrates Drawer menu for Profile, Support, About, Admin Panel
/// ---------------------------------------------------------------------------
class BottomNavShell extends StatefulWidget {
  static const route = '/home';
  const BottomNavShell({super.key});

  @override
  State<BottomNavShell> createState() => _BottomNavShellState();
}

class _BottomNavShellState extends State<BottomNavShell> {
  int _index = 0;

  // List of screen widgets for each bottom tab
  final _pages = const [
    DashboardScreen(),
    ForecastScreen(),
    AlertsHistoryScreen(),
    EducationScreen(),
    SettingsScreen(),
  ];

  final _titles = const ["Dashboard", "Forecast", "Alerts", "Education", "Settings"];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // Adds Drawer to all tabs
      drawer: const DrawerMenu(),

      // Dynamic AppBar (title changes with tab)
      appBar: AppBar(
        title: Text(
          _titles[_index],
          style: const TextStyle(fontWeight: FontWeight.w700, color: AppColors.text),
        ),
        centerTitle: true,
        backgroundColor: AppColors.bg,
        elevation: 0,
        leading: Builder(
          builder: (context) => IconButton(
            icon: const Icon(Icons.menu_rounded, color: AppColors.primary),
            onPressed: () => Scaffold.of(context).openDrawer(),
          ),
        ),
      ),

      // Keeps each tab’s state alive
      body: IndexedStack(index: _index, children: _pages),

      // Bottom Navigation Bar configuration
      bottomNavigationBar: NavigationBarTheme(
        data: NavigationBarThemeData(
          indicatorColor: Colors.white.withOpacity(0.2),
          labelTextStyle: MaterialStateProperty.resolveWith((states) {
            if (states.contains(MaterialState.selected)) {
              return const TextStyle(fontWeight: FontWeight.w600, color: Colors.white);
            }
            return TextStyle(fontWeight: FontWeight.w500, color: Colors.white.withOpacity(0.7));
          }),
        ),
        child: NavigationBar(
          backgroundColor: AppColors.secondary,
          selectedIndex: _index,
          onDestinationSelected: (i) => setState(() => _index = i),
          destinations: const [
            NavigationDestination(
              icon: Icon(Icons.home_outlined, color: Colors.white70),
              selectedIcon: Icon(Icons.home_rounded, color: Colors.white),
              label: "Dashboard",
            ),
            NavigationDestination(
              icon: Icon(Icons.show_chart_outlined, color: Colors.white70),
              selectedIcon: Icon(Icons.show_chart_rounded, color: Colors.white),
              label: "Forecast",
            ),
            NavigationDestination(
              icon: Icon(Icons.notifications_outlined, color: Colors.white70),
              selectedIcon: Icon(Icons.notifications_active_rounded, color: Colors.white),
              label: "Alerts",
            ),
            NavigationDestination(
              icon: Icon(Icons.school_outlined, color: Colors.white70),
              selectedIcon: Icon(Icons.school_rounded, color: Colors.white),
              label: "Education",
            ),
            NavigationDestination(
              icon: Icon(Icons.settings_outlined, color: Colors.white70),
              selectedIcon: Icon(Icons.settings_rounded, color: Colors.white),
              label: "Settings",
            ),
          ],
        ),
      ),
    );
  }
}
