package org.example;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LabelParserTest {
    @Test
    void parsesValidLabelEvent() {
        LabelRow label = LabelParser.parse(validPayload(), "secom-0000001");

        assertEquals("label:secom-0000001:r1", label.labelEventId());
        assertEquals("secom-0000001", label.sampleId());
        assertEquals(1L, label.labelRevision());
        assertEquals(123.45, label.measuredAt());
        assertEquals(1, label.actualValue());
        assertEquals("fail", label.actualLabel());
    }

    @Test
    void rejectsNullKafkaValue() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> LabelParser.parse(null, null)
        );

        assertEquals("Kafka message value is null", error.getMessage());
    }

    @Test
    void rejectsNonObjectJson() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> LabelParser.parse("[]", null)
        );

        assertEquals("Kafka message value must be a JSON object", error.getMessage());
    }

    @Test
    void rejectsKafkaKeyMismatch() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> LabelParser.parse(validPayload(), "secom-0000002")
        );

        assertEquals(
            "Kafka key differs from sample_id: key=secom-0000002 sample_id=secom-0000001",
            error.getMessage()
        );
    }

    @Test
    void rejectsInvalidSampleId() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> LabelParser.parse(validPayload().replace("secom-0000001", "sample-1"), null)
        );

        assertEquals("invalid sample_id: sample-1", error.getMessage());
    }

    @Test
    void rejectsInvalidActualValue() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> LabelParser.parse(validPayload().replace("\"actual_value\": 1", "\"actual_value\": 0"), null)
        );

        assertEquals("actual_value must be -1 or 1: 0", error.getMessage());
    }

    @Test
    void rejectsLabelMismatch() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> LabelParser.parse(validPayload().replace("\"actual_label\": \"fail\"", "\"actual_label\": \"pass\""), null)
        );

        assertEquals("actual_label mismatch: actual_value=1 actual_label=pass", error.getMessage());
    }

    @Test
    void rejectsNonPositiveLabelRevision() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> LabelParser.parse(
                validPayload().replace("\"label_revision\": 1", "\"label_revision\": 0"),
                null
            )
        );

        assertEquals("label_revision must be > 0: 0", error.getMessage());
    }

    @Test
    void rejectsNegativeMeasuredAt() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> LabelParser.parse(
                validPayload().replace("\"measured_at\": 123.45", "\"measured_at\": -1.0"),
                null
            )
        );

        assertEquals("measured_at must be >= 0: -1.0", error.getMessage());
    }

    private static String validPayload() {
        return """
            {
              "label_event_id": "label:secom-0000001:r1",
              "sample_id": "secom-0000001",
              "label_revision": 1,
              "measured_at": 123.45,
              "actual_value": 1,
              "actual_label": "fail"
            }
            """;
    }
}
