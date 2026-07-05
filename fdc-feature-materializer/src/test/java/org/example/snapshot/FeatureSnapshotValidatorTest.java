package org.example.snapshot;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FeatureSnapshotValidatorTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final FeatureSnapshotValidator validator = new FeatureSnapshotValidator();

    @Test
    void acceptsValidCompleteSnapshot() {
        String raw = validSnapshot().toString();

        String sampleId = validator.validate(raw, "sample-1");

        assertEquals("sample-1", sampleId);
    }

    @Test
    void rejectsKafkaKeyMismatch() {
        String raw = validSnapshot().toString();

        assertThrows(IllegalArgumentException.class, () -> validator.validate(raw, "different-sample"));
    }

    @Test
    void rejectsMissingFeatureKey() {
        ObjectNode snapshot = validSnapshot();
        snapshot.withObject("/features").remove("f123");

        assertThrows(IllegalArgumentException.class, () -> validator.validate(snapshot.toString(), "sample-1"));
    }

    @Test
    void rejectsNonNumericFeatureValue() {
        ObjectNode snapshot = validSnapshot();
        snapshot.withObject("/features").put("f123", "bad-value");

        assertThrows(IllegalArgumentException.class, () -> validator.validate(snapshot.toString(), "sample-1"));
    }

    @Test
    void rejectsStatusThatDoesNotMatchCompleteness() {
        ObjectNode snapshot = validSnapshot();
        snapshot.put("snapshot_status", "partial");

        assertThrows(IllegalArgumentException.class, () -> validator.validate(snapshot.toString(), "sample-1"));
    }

    private static ObjectNode validSnapshot() {
        ObjectNode snapshot = MAPPER.createObjectNode();
        snapshot.put("sample_id", "sample-1");
        snapshot.put("serving_snapshot_id", "snapshot-1");
        snapshot.put("snapshot_time", 1719715200.0);
        snapshot.put("snapshot_status", "complete");
        snapshot.put("feature_count", 590);
        snapshot.put("missing_count", 0);
        snapshot.put("is_complete", true);

        ObjectNode features = snapshot.putObject("features");
        for (int index = 0; index < 590; index++) {
            features.put(String.format("f%03d", index), index);
        }

        return snapshot;
    }
}
