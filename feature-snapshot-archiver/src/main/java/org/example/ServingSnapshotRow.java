package org.example;

record ServingSnapshotRow(
    String servingSnapshotId,
    String sampleId,
    double snapshotTime,
    double windowStart,
    double windowEnd,
    String snapshotStatus,
    int featureCount,
    int missingCount,
    boolean isComplete,
    String featuresJson,
    String simulationRunId,
    String driftSegment,
    double createdAt
) {
}
