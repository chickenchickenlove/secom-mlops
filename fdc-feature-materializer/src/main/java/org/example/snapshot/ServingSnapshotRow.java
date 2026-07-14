package org.example.snapshot;

public record ServingSnapshotRow(
    String servingSnapshotId,
    long snapshotVersion,
    String featureHash,
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
    String driftSegment
) {
}
