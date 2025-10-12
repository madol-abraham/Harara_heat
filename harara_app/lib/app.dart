import 'package:flutter/material.dart';
import 'core/theme/app_theme.dart';
import 'routes/app_routes.dart';
import 'screens/onboarding/onboarding_screen.dart';

class HararaApp extends StatelessWidget {
  const HararaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Harara',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      initialRoute: OnboardingScreen.route,
      routes: AppRoutes.all,
    );
  }
}
