package org.example;

record FeatureEventRow(
    String eventId,
    String sampleId,
    double eventTime,
    String featureGroup,
    String featuresJson,
    String simulationRunId,
    String driftSegment,
    double createdAt
) {
}
