package org.example;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PredictionLogParserTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String FEATURE_HASH = "sha256:v1:" + "0".repeat(64);

    @Test
    void parsesValidPredictionLogWithoutFeatures() {
        PredictionLogRow row = PredictionLogParser.parse(validEvent().toString(), "secom-0000001");

        assertEquals("pred-001", row.predictionId());
        assertEquals("req-001", row.requestId());
        assertEquals("secom-0000001", row.sampleId());
        assertEquals("state:secom-0000001:1000:3", row.servingSnapshotId());
        assertEquals(3L, row.snapshotVersion());
        assertEquals(FEATURE_HASH, row.featureHash());
        assertEquals("run-001", row.modelRunId());
        assertEquals("champion", row.modelAlias());
        assertEquals("online", row.runtimeSlot());
        assertEquals(-1, row.predictedValue());
        assertEquals("pass", row.predictedLabel());
        assertEquals(44, row.missingCount());
    }

    @Test
    void defaultsRuntimeSlot() {
        ObjectNode event = validEvent();
        event.remove("runtime_slot");

        PredictionLogRow row = PredictionLogParser.parse(event.toString(), null);

        assertEquals("unknown", row.runtimeSlot());
    }

    @Test
    void keepsMissingOptionalModelFieldsNull() {
        ObjectNode event = validEvent();
        event.remove("model_name");
        event.remove("model_version");
        event.remove("model_alias");
        event.remove("model_uri");

        PredictionLogRow row = PredictionLogParser.parse(event.toString(), null);

        assertNull(row.modelName());
        assertNull(row.modelVersion());
        assertNull(row.modelAlias());
        assertNull(row.modelUri());
    }

    @Test
    void rejectsKafkaKeyMismatch() {
        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(validEvent().toString(), "secom-0000002")
        );

        assertEquals(
            "Kafka key differs from sample_id: key=secom-0000002 sample_id=secom-0000001",
            error.getMessage()
        );
    }

    @Test
    void rejectsPredictedLabelMismatch() {
        ObjectNode event = validEvent();
        event.put("predicted_label", "fail");

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(event.toString(), null)
        );

        assertEquals(
            "predicted_label mismatch: predicted_value=-1 predicted_label=fail",
            error.getMessage()
        );
    }

    @Test
    void rejectsNegativeMissingCount() {
        ObjectNode event = validEvent();
        event.put("missing_count", -1);

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(event.toString(), null)
        );

        assertEquals(
            "missing_count must be between 0 and 590: sample_id=secom-0000001 missing_count=-1",
            error.getMessage()
        );
    }

    @Test
    void rejectsMissingCountAboveFeatureCount() {
        ObjectNode event = validEvent();
        event.put("missing_count", 591);

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(event.toString(), null)
        );

        assertEquals(
            "missing_count must be between 0 and 590: sample_id=secom-0000001 missing_count=591",
            error.getMessage()
        );
    }

    @Test
    void rejectsBlankServingSnapshotId() {
        ObjectNode event = validEvent();
        event.put("serving_snapshot_id", " ");

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(event.toString(), null)
        );

        assertEquals(
            "required text field missing or invalid: serving_snapshot_id",
            error.getMessage()
        );
    }

    @Test
    void rejectsNonIntegralSnapshotVersion() {
        ObjectNode event = validEvent();
        event.put("snapshot_version", 3.5);

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(event.toString(), null)
        );

        assertEquals(
            "required long field missing or invalid: snapshot_version",
            error.getMessage()
        );
    }

    @Test
    void rejectsNonPositiveSnapshotVersion() {
        ObjectNode event = validEvent();
        event.put("snapshot_version", 0);

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(event.toString(), null)
        );

        assertEquals("long field must be >= 1: snapshot_version", error.getMessage());
    }

    @Test
    void rejectsMissingOrInvalidFeatureHash() {
        ObjectNode missingHash = validEvent();
        missingHash.remove("feature_hash");
        assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(missingHash.toString(), null)
        );

        ObjectNode invalidHash = validEvent();
        invalidHash.put("feature_hash", "sha256:v1:not-a-hash");

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(invalidHash.toString(), null)
        );

        assertEquals("invalid feature_hash: sha256:v1:not-a-hash", error.getMessage());
    }

    @Test
    void acceptsSnapshotVersionLargerThanIntegerRange() {
        ObjectNode event = validEvent();
        event.put("snapshot_version", 3_000_000_000L);

        PredictionLogRow row = PredictionLogParser.parse(event.toString(), null);

        assertEquals(3_000_000_000L, row.snapshotVersion());
    }

    @Test
    void rejectsInvalidProbability() {
        ObjectNode event = validEvent();
        event.put("fail_probability", 1.1);

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(event.toString(), null)
        );

        assertEquals("probability field must be <= 1.0: fail_probability", error.getMessage());
    }

    private static ObjectNode validEvent() {
        ObjectNode event = MAPPER.createObjectNode();
        event.put("prediction_id", "pred-001");
        event.put("request_id", "req-001");
        event.put("sample_id", "secom-0000001");
        event.put("serving_snapshot_id", "state:secom-0000001:1000:3");
        event.put("snapshot_version", 3L);
        event.put("feature_hash", FEATURE_HASH);
        event.put("model_run_id", "run-001");
        event.put("model_name", "secom-xgb");
        event.put("model_version", "1");
        event.put("model_alias", "champion");
        event.put("model_uri", "models:/secom-xgb/1");
        event.put("runtime_slot", "online");
        event.put("predicted_at", 1.0);
        event.put("fail_probability", 0.2);
        event.put("predicted_value", -1);
        event.put("predicted_label", "pass");
        event.put("threshold", 0.5);
        event.put("missing_count", 44);
        event.put("latency_ms", 12.3);

        return event;
    }
}
