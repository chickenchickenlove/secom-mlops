package org.example.snapshot;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeatureSnapshotParserTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String FEATURE_HASH = "sha256:v1:" + "0".repeat(64);

    @Test
    void parsesValidCompleteSnapshot() throws Exception {
        ServingSnapshotRow row = FeatureSnapshotParser.parse(validSnapshot().toString(), "secom-0000001");

        assertEquals("state:secom-0000001:1000:3", row.servingSnapshotId());
        assertEquals(3L, row.snapshotVersion());
        assertEquals(FEATURE_HASH, row.featureHash());
        assertEquals("secom-0000001", row.sampleId());
        assertEquals(1.0, row.snapshotTime());
        assertEquals(0.5, row.windowStart());
        assertEquals(1.0, row.windowEnd());
        assertEquals("complete", row.snapshotStatus());
        assertEquals(590, row.featureCount());
        assertEquals(0, row.missingCount());
        assertTrue(row.isComplete());
        assertEquals("sim-001", row.simulationRunId());
        assertEquals("stable", row.driftSegment());

        JsonNode features = MAPPER.readTree(row.featuresJson());
        assertEquals(0.0, features.get("f000").asDouble());
        assertEquals(589.0, features.get("f589").asDouble());
    }

    @Test
    void parsesValidPartialSnapshot() {
        ObjectNode snapshot = validSnapshot();
        snapshot.put("snapshot_status", "partial");
        snapshot.put("feature_count", 589);
        snapshot.put("missing_count", 1);
        snapshot.put("is_complete", false);
        ((ObjectNode) snapshot.get("features")).putNull("f589");

        ServingSnapshotRow row = FeatureSnapshotParser.parse(snapshot.toString(), null);

        assertEquals("partial", row.snapshotStatus());
        assertEquals(589, row.featureCount());
        assertEquals(1, row.missingCount());
    }

    @Test
    void defaultsOptionalMetadata() {
        ObjectNode snapshot = validSnapshot();
        snapshot.remove("simulation_run_id");
        snapshot.putNull("drift_segment");

        ServingSnapshotRow row = FeatureSnapshotParser.parse(snapshot.toString(), null);

        assertNull(row.simulationRunId());
        assertNull(row.driftSegment());
    }

    @Test
    void rejectsMissingInvalidOrMismatchedSnapshotVersion() {
        ObjectNode missingSourceCount = validSnapshot();
        missingSourceCount.remove("source_event_count");
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(missingSourceCount.toString(), null)
        );

        ObjectNode missingVersion = validSnapshot();
        missingVersion.remove("snapshot_version");
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(missingVersion.toString(), null)
        );

        ObjectNode zeroSourceCount = validSnapshot();
        zeroSourceCount.put("source_event_count", 0);
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(zeroSourceCount.toString(), null)
        );

        ObjectNode zeroVersion = validSnapshot();
        zeroVersion.put("snapshot_version", 0);
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(zeroVersion.toString(), null)
        );

        ObjectNode nonIntegral = validSnapshot();
        nonIntegral.put("snapshot_version", 3.5);
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(nonIntegral.toString(), null)
        );

        ObjectNode mismatch = validSnapshot();
        mismatch.put("snapshot_version", 4);
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(mismatch.toString(), null)
        );
    }

    @Test
    void rejectsMissingOrInvalidFeatureHash() {
        ObjectNode missingHash = validSnapshot();
        missingHash.remove("feature_hash");
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(missingHash.toString(), null)
        );

        ObjectNode invalidHash = validSnapshot();
        invalidHash.put("feature_hash", "sha256:v1:not-a-hash");

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(invalidHash.toString(), null)
        );

        assertEquals("invalid feature_hash: sha256:v1:not-a-hash", error.getMessage());
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
    void rejectsInvalidSampleId() {
        ObjectNode snapshot = validSnapshot();
        snapshot.put("sample_id", "sample-1");

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(snapshot.toString(), null)
        );

        assertEquals("invalid sample_id: sample-1", error.getMessage());
    }

    @Test
    void rejectsNegativeOrReversedWindow() {
        ObjectNode negativeSnapshotTime = validSnapshot();
        negativeSnapshotTime.put("snapshot_time", -1.0);
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(negativeSnapshotTime.toString(), null)
        );

        ObjectNode negativeWindowEnd = validSnapshot();
        negativeWindowEnd.put("window_end", -1.0);
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(negativeWindowEnd.toString(), null)
        );

        ObjectNode reversedWindow = validSnapshot();
        reversedWindow.put("window_end", 0.25);
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(reversedWindow.toString(), null)
        );
    }

    @Test
    void rejectsLegacyArchiverOnlyStatuses() {
        for (String status : new String[] {"timed_out", "late_update"}) {
            ObjectNode snapshot = validSnapshot();
            snapshot.put("snapshot_status", status);

            assertThrows(
                IllegalArgumentException.class,
                () -> FeatureSnapshotParser.parse(snapshot.toString(), null)
            );
        }
    }

    @Test
    void rejectsStatusThatDoesNotMatchCompleteness() {
        ObjectNode snapshot = validSnapshot();
        snapshot.put("snapshot_status", "partial");

        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(snapshot.toString(), null)
        );
    }

    @Test
    void rejectsBooleanThatDoesNotMatchFeatureCount() {
        ObjectNode snapshot = validSnapshot();
        snapshot.put("feature_count", 589);

        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(snapshot.toString(), null)
        );
    }

    @Test
    void rejectsMissingOrNonCanonicalFeatureKey() {
        ObjectNode missingFeature = validSnapshot();
        ((ObjectNode) missingFeature.get("features")).remove("f123");

        IllegalArgumentException sizeError = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(missingFeature.toString(), null)
        );
        assertEquals("features must contain exactly 590 canonical keys: 589", sizeError.getMessage());

        ObjectNode nonCanonicalFeature = validSnapshot();
        ObjectNode features = (ObjectNode) nonCanonicalFeature.get("features");
        features.remove("f123");
        features.put("f590", 590.0);

        IllegalArgumentException keyError = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(nonCanonicalFeature.toString(), null)
        );
        assertEquals(
            "missing canonical feature key: sample_id=secom-0000001 key=f123",
            keyError.getMessage()
        );
    }

    @Test
    void rejectsNonNumericFeatureValue() {
        ObjectNode snapshot = validSnapshot();
        ((ObjectNode) snapshot.get("features")).put("f123", "bad-value");

        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(snapshot.toString(), null)
        );
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
    void rejectsInvalidOptionalMetadataTypes() {
        ObjectNode invalidSimulationRun = validSnapshot();
        invalidSimulationRun.put("simulation_run_id", 1);
        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureSnapshotParser.parse(invalidSimulationRun.toString(), null)
        );

    }

    @Test
    void rejectsNullAndNonObjectPayloads() {
        assertThrows(IllegalArgumentException.class, () -> FeatureSnapshotParser.parse(null, null));
        assertThrows(IllegalArgumentException.class, () -> FeatureSnapshotParser.parse("[]", null));
        assertThrows(IllegalArgumentException.class, () -> FeatureSnapshotParser.parse("not-json", null));
    }

    private static ObjectNode validSnapshot() {
        ObjectNode snapshot = MAPPER.createObjectNode();
        snapshot.put("serving_snapshot_id", "state:secom-0000001:1000:3");
        snapshot.put("sample_id", "secom-0000001");
        snapshot.put("source_event_count", 3);
        snapshot.put("snapshot_version", 3);
        snapshot.put("feature_hash", FEATURE_HASH);
        snapshot.put("snapshot_time", 1.0);
        snapshot.put("window_start", 0.5);
        snapshot.put("window_end", 1.0);
        snapshot.put("snapshot_status", "complete");
        snapshot.put("feature_count", 590);
        snapshot.put("missing_count", 0);
        snapshot.put("is_complete", true);
        snapshot.put("simulation_run_id", "sim-001");
        snapshot.put("drift_segment", "stable");

        ObjectNode features = snapshot.putObject("features");
        for (int index = 0; index < 590; index++) {
            features.put(String.format("f%03d", index), (double) index);
        }

        return snapshot;
    }
}
