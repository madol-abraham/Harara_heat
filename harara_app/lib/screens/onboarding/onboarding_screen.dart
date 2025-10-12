import 'package:flutter/material.dart';
import '../../widgets/gradient_card.dart';
import '../../core/theme/colors.dart';
// Updated import 👇
import '../../navigation/bottom_nav.dart';

/// ---------------------------------------------------------------------------
///  OnboardingScreen - Shown only once when app starts
/// ---------------------------------------------------------------------------
class OnboardingScreen extends StatelessWidget {
  static const route = '/onboarding';
  const OnboardingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final pages = [
      _Page("Welcome to Harara",
          "Real-time heatwave monitoring and life-saving alerts for your town."),
      _Page("Stay Informed",
          "Clear dashboards, AI predictions, and SMS notifications when risk rises."),
      _Page("Act Early",
          "Practical guidance to keep families, students, and workers safe."),
    ];

    return Scaffold(
      body: PageView.builder(
        itemCount: pages.length,
        itemBuilder: (_, i) =>
            _OnboardSlide(page: pages[i], index: i, total: pages.length),
      ),
    );
  }
}

class _Page {
  final String title, subtitle;
  _Page(this.title, this.subtitle);
}

class _OnboardSlide extends StatelessWidget {
  final _Page page;
  final int index, total;
  const _OnboardSlide(
      {required this.page, required this.index, required this.total});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [AppColors.primary, AppColors.primaryLight],
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
        ),
      ),
      child: SafeArea(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.wb_sunny_rounded, color: Colors.white, size: 120),
            const SizedBox(height: 24),
            Text(page.title,
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 28,
                    fontWeight: FontWeight.w800)),
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 28, vertical: 12),
              child: Text(page.subtitle,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white70, fontSize: 16)),
            ),
            const SizedBox(height: 24),
            GradientCard(
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: List.generate(total, (i) => _dot(i == index)),
              ),
            ),
            const SizedBox(height: 14),
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                padding:
                    const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                backgroundColor: Colors.white,
                foregroundColor: AppColors.primary,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
              // 🚀 Updated navigation route below 👇
              onPressed: () => Navigator.pushReplacementNamed(
                  context, BottomNavShell.route),
              icon: const Icon(Icons.arrow_forward_rounded),
              label: const Text("Get Started"),
            ),
          ],
        ),
      ),
    );
  }

  Widget _dot(bool active) => AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        width: active ? 24 : 8,
        height: 8,
        margin: const EdgeInsets.symmetric(horizontal: 4),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(active ? 0.95 : 0.5),
          borderRadius: BorderRadius.circular(6),
        ),
      );
}
