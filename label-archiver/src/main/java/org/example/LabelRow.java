package org.example;

record LabelRow(
    String labelEventId,
    String sampleId,
    long labelRevision,
    double measuredAt,
    int actualValue,
    String actualLabel
) {
}
