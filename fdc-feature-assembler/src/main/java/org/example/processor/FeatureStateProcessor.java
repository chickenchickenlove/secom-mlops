package org.example.processor;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.kafka.streams.processor.api.Processor;
import org.apache.kafka.streams.processor.api.ProcessorContext;
import org.apache.kafka.streams.processor.api.Record;
import org.apache.kafka.streams.state.KeyValueStore;

import java.util.regex.Pattern;

public final class FeatureStateProcessor implements Processor<String, String, String, String> {

    private static final int NUM_FEATURES = 590;
    private static final Pattern FEATURE_KEY_PATTERN = Pattern.compile("^f\\d{3}$");
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String storeName;
    private ProcessorContext<String, String> context;
    private KeyValueStore<String, String> store;

    public FeatureStateProcessor(String storeName) {
        this.storeName = storeName;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void init(ProcessorContext<String, String> context) {
        this.context = context;
        this.store = context.getStateStore(storeName);
    }

    @Override
    public void process(org.apache.kafka.streams.processor.api.Record<String, String> record) {
        if (record.value() == null) {
            throw new IllegalArgumentException("feature patch value must not be null");
        }

        JsonNode event = parseJson(record.value());
        String sampleId = requiredText(event, "sample_id");

        if (record.key() != null && !record.key().equals(sampleId)) {
            throw new IllegalArgumentException(
                "Kafka record key differs from event sample_id: key="
                    + record.key()
                    + " sample_id="
                    + sampleId
            );
        }

        double eventTime = requiredNumber(event, "event_time");
        JsonNode patchFeatures = requiredObject(event, "features");

        ObjectNode state = loadOrCreateState(sampleId);
        ObjectNode observedFeatures = ensureObject(state, "features");

        patchFeatures.fields().forEachRemaining(entry -> {
            String featureKey = entry.getKey();
            JsonNode value = entry.getValue();

            validateFeatureKey(sampleId, featureKey);
            putFeatureValue(observedFeatures, sampleId, featureKey, value);
        });

        updateWindow(state, eventTime);
        state.put("source_event_count", state.path("source_event_count").asInt(0) + 1);
        state.put("last_event_time", eventTime);
        copyOptionalText(event, state, "feature_group", "last_feature_group");
        copyOptionalText(event, state, "event_id", "last_event_id");
        copyOptionalText(event, state, "simulation_run_id", "simulation_run_id");
        copyOptionalText(event, state, "drift_segment", "drift_segment");
        copyOptionalNumber(event, state, "source_row_index", "source_row_index");

        ObjectNode output = buildOutput(sampleId, state, observedFeatures);

        store.put(sampleId, stringify(state));
        context.forward(new Record<>(sampleId, stringify(output), record.timestamp()));
    }

    private ObjectNode loadOrCreateState(String sampleId) {
        String raw = store.get(sampleId);
        if (raw == null) {
            ObjectNode state = MAPPER.createObjectNode();
            state.put("sample_id", sampleId);
            state.set("features", MAPPER.createObjectNode());
            state.put("source_event_count", 0);
            return state;
        }

        JsonNode parsed = parseJson(raw);
        if (!parsed.isObject()) {
            throw new IllegalArgumentException("stored state must be a JSON object: sample_id=" + sampleId);
        }

        return (ObjectNode) parsed;
    }

    private static ObjectNode buildOutput(String sampleId, ObjectNode state, ObjectNode observedFeatures) {
        ObjectNode canonicalFeatures = MAPPER.createObjectNode();
        int missingCount = 0;

        for (int index = 0; index < NUM_FEATURES; index++) {
            String key = featureKey(index);
            JsonNode value = observedFeatures.get(key);

            if (value == null || value.isNull()) {
                canonicalFeatures.putNull(key);
                missingCount++;
            } else if (value.isNumber()) {
                canonicalFeatures.set(key, value);
            } else {
                throw new IllegalArgumentException(
                    "stored feature value must be numeric or null: sample_id="
                        + sampleId
                        + " key="
                        + key
                );
            }
        }

        int featureCount = observedFeatures.size();
        boolean isComplete = featureCount == NUM_FEATURES;
        double snapshotTime = state.path("window_end").asDouble(state.path("last_event_time").asDouble());
        int sourceEventCount = state.path("source_event_count").asInt();

        ObjectNode output = MAPPER.createObjectNode();
        output.put("serving_snapshot_id", "state:" + sampleId + ":" + (long) (snapshotTime * 1000) + ":" + sourceEventCount);
        output.put("sample_id", sampleId);
        output.put("snapshot_time", snapshotTime);
        output.put("window_start", state.path("window_start").asDouble(snapshotTime));
        output.put("window_end", state.path("window_end").asDouble(snapshotTime));
        output.put("snapshot_status", isComplete ? "complete" : "partial");
        output.put("feature_count", featureCount);
        output.put("missing_count", missingCount);
        output.put("is_complete", isComplete);
        output.set("features", canonicalFeatures);
        output.put("source_event_count", sourceEventCount);
        putIfPresent(output, state, "last_feature_group");
        putIfPresent(output, state, "last_event_id");
        putIfPresent(output, state, "simulation_run_id");
        putIfPresent(output, state, "drift_segment");

        return output;
    }

    private static void updateWindow(ObjectNode state, double eventTime) {
        if (!state.hasNonNull("window_start")) {
            state.put("window_start", eventTime);
        } else {
            state.put("window_start", Math.min(state.path("window_start").asDouble(), eventTime));
        }

        if (!state.hasNonNull("window_end")) {
            state.put("window_end", eventTime);
        } else {
            state.put("window_end", Math.max(state.path("window_end").asDouble(), eventTime));
        }
    }

    private static JsonNode parseJson(String raw) {
        try {
            return MAPPER.readTree(raw);
        } catch (JsonProcessingException error) {
            throw new IllegalArgumentException("invalid JSON payload", error);
        }
    }

    private static String stringify(JsonNode node) {
        try {
            return MAPPER.writeValueAsString(node);
        } catch (JsonProcessingException error) {
            throw new IllegalStateException("failed to serialize JSON", error);
        }
    }

    private static String requiredText(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException("required text field missing or invalid: " + fieldName);
        }
        return value.asText();
    }

    private static double requiredNumber(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isNumber()) {
            throw new IllegalArgumentException("required numeric field missing or invalid: " + fieldName);
        }
        return value.asDouble();
    }

    private static JsonNode requiredObject(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isObject()) {
            throw new IllegalArgumentException("required object field missing or invalid: " + fieldName);
        }
        return value;
    }

    private static ObjectNode ensureObject(ObjectNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null) {
            ObjectNode created = MAPPER.createObjectNode();
            node.set(fieldName, created);
            return created;
        }

        if (!value.isObject()) {
            throw new IllegalArgumentException("state field must be an object: " + fieldName);
        }

        return (ObjectNode) value;
    }

    private static void validateFeatureKey(String sampleId, String featureKey) {
        if (!FEATURE_KEY_PATTERN.matcher(featureKey).matches()) {
            throw new IllegalArgumentException(
                "unexpected feature key format: sample_id=" + sampleId + " key=" + featureKey
            );
        }

        int index = Integer.parseInt(featureKey.substring(1));
        if (index < 0 || index >= NUM_FEATURES) {
            throw new IllegalArgumentException(
                "feature key out of range: sample_id=" + sampleId + " key=" + featureKey
            );
        }
    }

    private static void putFeatureValue(ObjectNode features, String sampleId, String featureKey, JsonNode value) {
        if (value == null || value.isNull()) {
            features.putNull(featureKey);
            return;
        }

        if (value.isNumber()) {
            features.put(featureKey, value.asDouble());
            return;
        }

        throw new IllegalArgumentException(
            "feature value must be numeric or null: sample_id="
                + sampleId
                + " key="
                + featureKey
        );
    }

    private static void copyOptionalText(JsonNode source, ObjectNode target, String sourceField, String targetField) {
        JsonNode value = source.get(sourceField);
        if (value == null || value.isNull()) {
            return;
        }
        if (!value.isTextual()) {
            throw new IllegalArgumentException("optional field must be text: " + sourceField);
        }
        target.put(targetField, value.asText());
    }

    private static void copyOptionalNumber(JsonNode source, ObjectNode target, String sourceField, String targetField) {
        JsonNode value = source.get(sourceField);
        if (value == null || value.isNull()) {
            return;
        }
        if (!value.isNumber()) {
            throw new IllegalArgumentException("optional field must be numeric: " + sourceField);
        }
        target.put(targetField, value.asInt());
    }

    private static void putIfPresent(ObjectNode output, ObjectNode state, String fieldName) {
        JsonNode value = state.get(fieldName);
        if (value != null && !value.isNull()) {
            output.set(fieldName, value);
        }
    }

    private static String featureKey(int index) {
        return String.format("f%03d", index);
    }
}