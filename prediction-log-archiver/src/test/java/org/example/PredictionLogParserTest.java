package org.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PredictionLogParserTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void parsesValidPredictionLog() throws Exception {
        PredictionLogRow row = PredictionLogParser.parse(validEvent().toString(), "secom-0000001");

        assertEquals("pred-001", row.predictionId());
        assertEquals("req-001", row.requestId());
        assertEquals("secom-0000001", row.sampleId());
        assertEquals("run-001", row.modelRunId());
        assertEquals("champion", row.modelAlias());
        assertEquals("online", row.runtimeSlot());
        assertEquals(-1, row.predictedValue());
        assertEquals("pass", row.predictedLabel());
        assertEquals(0, row.missingCount());

        JsonNode features = MAPPER.readTree(row.featuresJson());
        assertEquals(590, features.size());
        assertEquals(0.0, features.get(0).asDouble());
        assertEquals(589.0, features.get(589).asDouble());
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
    void rejectsMissingCountMismatch() {
        ObjectNode event = validEvent();
        ((ArrayNode) event.get("features")).set(10, NullNode.getInstance());

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(event.toString(), null)
        );

        assertEquals(
            "missing_count mismatch: sample_id=secom-0000001 missing_count=0 computed_missing_count=1",
            error.getMessage()
        );
    }

    @Test
    void rejectsNonNumericFeatureValue() {
        ObjectNode event = validEvent();
        ((ArrayNode) event.get("features")).set(10, TextNode.valueOf("bad"));

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> PredictionLogParser.parse(event.toString(), null)
        );

        assertEquals(
            "feature value must be numeric or null: sample_id=secom-0000001 index=10",
            error.getMessage()
        );
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
        event.put("missing_count", 0);
        event.put("latency_ms", 12.3);

        ArrayNode features = event.putArray("features");
        for (int index = 0; index < 590; index++) {
            features.add((double) index);
        }

        return event;
    }
}
