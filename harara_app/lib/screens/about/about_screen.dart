import 'package:flutter/material.dart';
import '../../widgets/heat_app_bar.dart';

class AboutScreen extends StatelessWidget {
  static const route = '/about';
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const HeatAppBar(title: "About Harara"),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: const [
          Text("Mission", style: TextStyle(fontWeight: FontWeight.w800, fontSize: 18)),
          SizedBox(height: 8),
          Text("Harara provides real-time heatwave monitoring for South Sudan, "
               "combining satellite data and AI predictions to protect communities."),
          SizedBox(height: 18),
          Text("How it works", style: TextStyle(fontWeight: FontWeight.w800, fontSize: 18)),
          SizedBox(height: 8),
          Text("We aggregate features (LST, humidity, wind, etc.) and run a model to predict "
               "heatwave risk. Alerts are delivered in-app and via SMS."),
        ],
      ),
    );
  }
}
