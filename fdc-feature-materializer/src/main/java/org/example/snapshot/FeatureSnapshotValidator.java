package org.example.snapshot;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.regex.Pattern;

public final class FeatureSnapshotValidator {
    private static final int NUM_FEATURES = 590;
    private static final Pattern FEATURE_KEY_PATTERN = Pattern.compile("^f\\d{3}$");

    private final ObjectMapper mapper;

    public FeatureSnapshotValidator() {
        this(new ObjectMapper());
    }

    FeatureSnapshotValidator(ObjectMapper mapper) {
        this.mapper = mapper;
    }

    public String validate(String raw, String kafkaKey) {
        if (raw == null) {
            throw new IllegalArgumentException("Kafka message value is null");
        }

        JsonNode snapshot = parseJson(raw);
        return validateSnapshot(snapshot, kafkaKey);
    }

    private JsonNode parseJson(String raw) {
        try {
            JsonNode parsed = mapper.readTree(raw);
            if (!parsed.isObject()) {
                throw new IllegalArgumentException("Kafka message value must be a JSON object");
            }
            return parsed;
        } catch (JsonProcessingException error) {
            throw new IllegalArgumentException("invalid JSON payload", error);
        }
    }

    private static String validateSnapshot(JsonNode snapshot, String kafkaKey) {
        String sampleId = requiredText(snapshot, "sample_id");

        if (kafkaKey != null && !kafkaKey.equals(sampleId)) {
            throw new IllegalArgumentException(
                "Kafka key differs from sample_id: key=" + kafkaKey + " sample_id=" + sampleId
            );
        }

        requiredText(snapshot, "serving_snapshot_id");
        requiredNumber(snapshot, "snapshot_time");

        String status = requiredText(snapshot, "snapshot_status");
        int featureCount = requiredInt(snapshot, "feature_count");
        int missingCount = requiredInt(snapshot, "missing_count");
        boolean isComplete = requiredBoolean(snapshot, "is_complete");

        if (!status.equals("partial") && !status.equals("complete")) {
            throw new IllegalArgumentException("snapshot_status must be partial or complete: " + status);
        }
        if (featureCount < 0 || featureCount > NUM_FEATURES) {
            throw new IllegalArgumentException("feature_count out of range: " + featureCount);
        }
        if (missingCount < 0 || missingCount > NUM_FEATURES) {
            throw new IllegalArgumentException("missing_count out of range: " + missingCount);
        }
        if (isComplete != (featureCount == NUM_FEATURES)) {
            throw new IllegalArgumentException("is_complete must match feature_count == 590");
        }

        String expectedStatus = isComplete ? "complete" : "partial";
        if (!status.equals(expectedStatus)) {
            throw new IllegalArgumentException("snapshot_status must be " + expectedStatus);
        }

        JsonNode features = snapshot.get("features");
        if (features == null || !features.isObject()) {
            throw new IllegalArgumentException("features must be a JSON object");
        }
        if (features.size() != NUM_FEATURES) {
            throw new IllegalArgumentException("features must contain exactly 590 keys: " + features.size());
        }

        for (int index = 0; index < NUM_FEATURES; index++) {
            String key = String.format("f%03d", index);
            JsonNode value = features.get(key);

            if (value == null) {
                throw new IllegalArgumentException("missing canonical feature key: " + key);
            }
            if (!FEATURE_KEY_PATTERN.matcher(key).matches()) {
                throw new IllegalArgumentException("invalid feature key: " + key);
            }
            if (!value.isNull() && !value.isNumber()) {
                throw new IllegalArgumentException("feature value must be numeric or null: " + key);
            }
            if (value.isNumber() && !Double.isFinite(value.asDouble())) {
                throw new IllegalArgumentException("feature value must be finite: " + key);
            }
        }

        return sampleId;
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

    private static int requiredInt(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isInt()) {
            throw new IllegalArgumentException("required integer field missing or invalid: " + fieldName);
        }
        return value.asInt();
    }

    private static boolean requiredBoolean(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isBoolean()) {
            throw new IllegalArgumentException("required boolean field missing or invalid: " + fieldName);
        }
        return value.asBoolean();
    }
}
