package org.example;

record PredictionLogRow(
    String predictionId,
    String requestId,
    String sampleId,
    String servingSnapshotId,
    long snapshotVersion,
    String featureHash,
    String modelRunId,
    String modelName,
    String modelVersion,
    String modelAlias,
    String modelUri,
    String runtimeSlot,
    double predictedAt,
    double failProbability,
    int predictedValue,
    String predictedLabel,
    double threshold,
    int missingCount,
    double latencyMs
) {
}
