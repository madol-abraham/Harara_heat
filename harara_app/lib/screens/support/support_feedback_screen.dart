import 'package:flutter/material.dart';
import '../../widgets/heat_app_bar.dart';

class SupportFeedbackScreen extends StatefulWidget {
  static const route = '/support';
  const SupportFeedbackScreen({super.key});

  @override
  State<SupportFeedbackScreen> createState() => _SupportFeedbackScreenState();
}

class _SupportFeedbackScreenState extends State<SupportFeedbackScreen> {
  final _form = GlobalKey<FormState>();
  final _msg = TextEditingController();

  @override
  void dispose() { _msg.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const HeatAppBar(title: "Support & Feedback"),
      body: Form(
        key: _form,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            const Text("Tell us how we can improve:", style: TextStyle(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            TextFormField(
              controller: _msg,
              minLines: 5, maxLines: 8,
              decoration: const InputDecoration(hintText: "Type your message..."),
              validator: (v) => (v == null || v.trim().isEmpty) ? "Message required" : null,
            ),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: () {
                if (_form.currentState!.validate()) {
                  // TODO: send to backend (FastAPI)
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text("Feedback sent. Thank you!")));
                  _msg.clear();
                }
              },
              icon: const Icon(Icons.send_rounded),
              label: const Text("Submit"),
            ),
          ],
        ),
      ),
    );
  }
}
