package org.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeatureSnapshotParserTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void parsesValidCompleteSnapshot() throws Exception {
        ServingSnapshotRow row = FeatureSnapshotParser.parse(validSnapshot().toString(), "secom-0000001");

        assertEquals("state:secom-0000001:1000:3", row.servingSnapshotId());
        assertEquals("secom-0000001", row.sampleId());
        assertEquals("complete", row.snapshotStatus());
        assertEquals(590, row.featureCount());
        assertEquals(0, row.missingCount());
        assertTrue(row.isComplete());
        assertEquals("sim-001", row.simulationRunId());
        assertEquals("stable", row.driftSegment());
        assertEquals(123.45, row.createdAt());

        JsonNode features = MAPPER.readTree(row.featuresJson());
        assertEquals(0.0, features.get("f000").asDouble());
        assertEquals(589.0, features.get("f589").asDouble());
    }

    @Test
    void acceptsTimedOutPartialSnapshot() {
        ObjectNode snapshot = validSnapshot();
        snapshot.put("snapshot_status", "timed_out");
        snapshot.put("feature_count", 589);
        snapshot.put("missing_count", 1);
        snapshot.put("is_complete", false);
        ((ObjectNode) snapshot.get("features")).putNull("f589");

        ServingSnapshotRow row = FeatureSnapshotParser.parse(snapshot.toString(), null);

        assertEquals("timed_out", row.snapshotStatus());
        assertEquals(589, row.featureCount());
        assertEquals(1, row.missingCount());
    }

    @Test
    void rejectsKafkaKeyMismatch() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(validSnapshot().toString(), "secom-0000002")
        );

        assertEquals(
            "Kafka key differs from sample_id: key=secom-0000002 sample_id=secom-0000001",
            error.getMessage()
        );
    }

    @Test
    void rejectsMissingFeatureKey() {
        ObjectNode snapshot = validSnapshot();
        ((ObjectNode) snapshot.get("features")).remove("f123");

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(snapshot.toString(), null)
        );

        assertEquals("features must contain exactly 590 canonical keys: 589", error.getMessage());
    }

    @Test
    void rejectsMissingCountMismatch() {
        ObjectNode snapshot = validSnapshot();
        ((ObjectNode) snapshot.get("features")).putNull("f123");

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(snapshot.toString(), null)
        );

        assertEquals(
            "missing_count mismatch: sample_id=secom-0000001 declared=0 observed=1",
            error.getMessage()
        );
    }

    @Test
    void rejectsInvalidSampleId() {
        ObjectNode snapshot = validSnapshot();
        snapshot.put("sample_id", "sample-1");

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(snapshot.toString(), null)
        );

        assertEquals("invalid sample_id: sample-1", error.getMessage());
    }

    private static ObjectNode validSnapshot() {
        ObjectNode snapshot = MAPPER.createObjectNode();
        snapshot.put("serving_snapshot_id", "state:secom-0000001:1000:3");
        snapshot.put("sample_id", "secom-0000001");
        snapshot.put("snapshot_time", 1.0);
        snapshot.put("window_start", 0.5);
        snapshot.put("window_end", 1.0);
        snapshot.put("snapshot_status", "complete");
        snapshot.put("feature_count", 590);
        snapshot.put("missing_count", 0);
        snapshot.put("is_complete", true);
        snapshot.put("simulation_run_id", "sim-001");
        snapshot.put("drift_segment", "stable");
        snapshot.put("created_at", 123.45);

        ObjectNode features = snapshot.putObject("features");
        for (int index = 0; index < 590; index++) {
            features.put(String.format("f%03d", index), (double) index);
        }

        return snapshot;
    }
}
