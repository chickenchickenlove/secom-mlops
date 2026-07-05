package org.example;

record LabelRow(
    String sampleId,
    int actualValue,
    String actualLabel,
    double labeledAt
) {
}
