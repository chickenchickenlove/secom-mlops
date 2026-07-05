package org.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeatureEventParserTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void parsesValidFeatureEvent() throws Exception {
        String payload = """
            {
              "event_id": "evt-001",
              "sample_id": "secom-0000001",
              "event_time": 12.5,
              "feature_group": "early",
              "features": {
                "f000": 1.2,
                "f589": null
              },
              "simulation_run_id": "sim-001",
              "drift_segment": "stable",
              "created_at": 123.45
            }
            """;

        FeatureEventRow row = FeatureEventParser.parse(payload, "secom-0000001");

        assertEquals("evt-001", row.eventId());
        assertEquals("secom-0000001", row.sampleId());
        assertEquals(12.5, row.eventTime());
        assertEquals("early", row.featureGroup());
        assertEquals("sim-001", row.simulationRunId());
        assertEquals("stable", row.driftSegment());
        assertEquals(123.45, row.createdAt());

        JsonNode features = MAPPER.readTree(row.featuresJson());
        assertEquals(1.2, features.get("f000").asDouble());
        assertTrue(features.get("f589").isNull());
    }

    @Test
    void defaultsOptionalFields() {
        double before = System.currentTimeMillis() / 1000.0;

        FeatureEventRow row = FeatureEventParser.parse("""
            {
              "event_id": "evt-001",
              "sample_id": "secom-0000001",
              "event_time": 12.5,
              "feature_group": "middle",
              "features": {
                "f001": 2.3
              }
            }
            """, null);

        double after = System.currentTimeMillis() / 1000.0;

        assertNull(row.simulationRunId());
        assertNull(row.driftSegment());
        assertTrue(row.createdAt() >= before);
        assertTrue(row.createdAt() <= after);
    }

    @Test
    void rejectsNullKafkaValue() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureEventParser.parse(null, null)
        );

        assertEquals("Kafka message value is null", error.getMessage());
    }

    @Test
    void rejectsNonObjectJson() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureEventParser.parse("[]", null)
        );

        assertEquals("Kafka message value must be a JSON object", error.getMessage());
    }

    @Test
    void rejectsKafkaKeyMismatch() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureEventParser.parse(validPayload(), "secom-0000002")
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
            () -> FeatureEventParser.parse(validPayload().replace("secom-0000001", "sample-1"), null)
        );

        assertEquals("invalid sample_id: sample-1", error.getMessage());
    }

    @Test
    void rejectsFeatureKeyOutOfRange() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureEventParser.parse(validPayload().replace("\"f000\"", "\"f590\""), null)
        );

        assertEquals("feature key out of range: sample_id=secom-0000001 key=f590", error.getMessage());
    }

    @Test
    void rejectsNonNumericFeatureValue() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureEventParser.parse(validPayload().replace("1.2", "\"bad\""), null)
        );

        assertEquals("feature value must be numeric or null: sample_id=secom-0000001 key=f000", error.getMessage());
    }

    @Test
    void acceptsAllFeatureGroups() {
        assertDoesNotThrow(() -> FeatureEventParser.parse(validPayload("early"), null));
        assertDoesNotThrow(() -> FeatureEventParser.parse(validPayload("middle"), null));
        assertDoesNotThrow(() -> FeatureEventParser.parse(validPayload("late"), null));
    }

    private static String validPayload() {
        return validPayload("early");
    }

    private static String validPayload(String featureGroup) {
        return """
            {
              "event_id": "evt-001",
              "sample_id": "secom-0000001",
              "event_time": 12.5,
              "feature_group": "%s",
              "features": {
                "f000": 1.2
              }
            }
            """.formatted(featureGroup);
    }
}
