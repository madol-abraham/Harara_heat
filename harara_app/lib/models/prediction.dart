class Prediction {
  final String town;
  final double probability;
  final bool alert;

  Prediction({
    required this.town,
    required this.probability,
    required this.alert,
  });
}

class PredictionRun {
  final DateTime runTs;
  final DateTime startDate;
  final DateTime endDate;
  final double threshold;
  final List<Prediction> predictions;

  PredictionRun({
    required this.runTs,
    required this.startDate,
    required this.endDate,
    required this.threshold,
    required this.predictions,
  });
}

class TownDetailArgs {
  final PredictionRun run;
  final Prediction prediction;

  TownDetailArgs({
    required this.run,
    required this.prediction,
  });
}