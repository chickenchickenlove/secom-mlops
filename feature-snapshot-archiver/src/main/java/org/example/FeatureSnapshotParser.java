package org.example;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Set;
import java.util.regex.Pattern;

final class FeatureSnapshotParser {
    private static final int NUM_FEATURES = 590;
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern SAMPLE_ID_PATTERN = Pattern.compile("^secom-\\d{7}$");
    private static final Pattern FEATURE_KEY_PATTERN = Pattern.compile("^f\\d{3}$");
    private static final Set<String> SNAPSHOT_STATUSES = Set.of(
        "partial",
        "timed_out",
        "complete",
        "late_update"
    );

    private FeatureSnapshotParser() {
    }

    static ServingSnapshotRow parse(String raw, String kafkaKey) {
        if (raw == null) {
            throw new IllegalArgumentException("Kafka message value is null");
        }

        JsonNode snapshot = parseJson(raw);
        return validateServingSnapshot(snapshot, kafkaKey);
    }

    private static JsonNode parseJson(String raw) {
        try {
            JsonNode parsed = MAPPER.readTree(raw);
            if (!parsed.isObject()) {
                throw new IllegalArgumentException("Kafka message value must be a JSON object");
            }
            return parsed;
        } catch (JsonProcessingException error) {
            throw new IllegalArgumentException("invalid JSON payload", error);
        }
    }

    private static ServingSnapshotRow validateServingSnapshot(JsonNode snapshot, String kafkaKey) {
        String servingSnapshotId = requiredText(snapshot, "serving_snapshot_id");
        String sampleId = requiredText(snapshot, "sample_id");
        double snapshotTime = requiredNumber(snapshot, "snapshot_time");
        double windowStart = requiredNumber(snapshot, "window_start");
        double windowEnd = requiredNumber(snapshot, "window_end");
        String snapshotStatus = requiredText(snapshot, "snapshot_status");
        int featureCount = requiredInt(snapshot, "feature_count");
        int missingCount = requiredInt(snapshot, "missing_count");
        boolean isComplete = requiredBoolean(snapshot, "is_complete");
        JsonNode features = requiredObject(snapshot, "features");

        if (!SAMPLE_ID_PATTERN.matcher(sampleId).matches()) {
            throw new IllegalArgumentException("invalid sample_id: " + sampleId);
        }
        if (kafkaKey != null && !kafkaKey.equals(sampleId)) {
            throw new IllegalArgumentException(
                "Kafka key differs from sample_id: key=" + kafkaKey + " sample_id=" + sampleId
            );
        }
        if (snapshotTime < 0.0) {
            throw new IllegalArgumentException("snapshot_time must be >= 0");
        }
        if (windowStart < 0.0) {
            throw new IllegalArgumentException("window_start must be >= 0");
        }
        if (windowEnd < windowStart) {
            throw new IllegalArgumentException("window_end must be >= window_start");
        }
        if (!SNAPSHOT_STATUSES.contains(snapshotStatus)) {
            throw new IllegalArgumentException("unexpected snapshot_status: " + snapshotStatus);
        }
        if (featureCount < 0 || featureCount > NUM_FEATURES) {
            throw new IllegalArgumentException("feature_count out of range: " + featureCount);
        }
        if (missingCount < 0 || missingCount > NUM_FEATURES) {
            throw new IllegalArgumentException("missing_count out of range: " + missingCount);
        }

        if (features.size() != NUM_FEATURES) {
            throw new IllegalArgumentException("features must contain exactly 590 canonical keys: " + features.size());
        }
        validateCanonicalFeatures(sampleId, features);

        if (isComplete != (featureCount == NUM_FEATURES)) {
            throw new IllegalArgumentException("is_complete must match feature_count == 590");
        }
        if (isComplete && !snapshotStatus.equals("complete")) {
            throw new IllegalArgumentException("complete snapshots must have snapshot_status=complete");
        }

        int observedMissingCount = countNullFeatures(features);
        if (observedMissingCount != missingCount) {
            throw new IllegalArgumentException(
                "missing_count mismatch: sample_id="
                    + sampleId
                    + " declared="
                    + missingCount
                    + " observed="
                    + observedMissingCount
            );
        }

        return new ServingSnapshotRow(
            servingSnapshotId,
            sampleId,
            snapshotTime,
            windowStart,
            windowEnd,
            snapshotStatus,
            featureCount,
            missingCount,
            isComplete,
            stringify(features),
            optionalText(snapshot, "simulation_run_id"),
            optionalText(snapshot, "drift_segment"),
            optionalNumber(snapshot, "created_at", System.currentTimeMillis() / 1000.0)
        );
    }

    private static void validateCanonicalFeatures(String sampleId, JsonNode features) {
        for (int index = 0; index < NUM_FEATURES; index++) {
            String key = featureKey(index);
            JsonNode value = features.get(key);

            if (value == null) {
                throw new IllegalArgumentException("missing canonical feature key: sample_id=" + sampleId + " key=" + key);
            }
            if (!FEATURE_KEY_PATTERN.matcher(key).matches()) {
                throw new IllegalArgumentException("invalid feature key: sample_id=" + sampleId + " key=" + key);
            }
            if (!value.isNull() && !value.isNumber()) {
                throw new IllegalArgumentException(
                    "feature value must be numeric or null: sample_id=" + sampleId + " key=" + key
                );
            }
            if (value.isNumber() && !Double.isFinite(value.asDouble())) {
                throw new IllegalArgumentException("feature value must be finite: sample_id=" + sampleId + " key=" + key);
            }
        }
    }

    private static int countNullFeatures(JsonNode features) {
        int count = 0;

        for (int index = 0; index < NUM_FEATURES; index++) {
            JsonNode value = features.get(featureKey(index));
            if (value == null || value.isNull()) {
                count++;
            }
        }

        return count;
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

        double number = value.asDouble();
        if (!Double.isFinite(number)) {
            throw new IllegalArgumentException("numeric field must be finite: " + fieldName);
        }
        return number;
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

    private static JsonNode requiredObject(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isObject()) {
            throw new IllegalArgumentException("required object field missing or invalid: " + fieldName);
        }
        return value;
    }

    private static String optionalText(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || value.isNull()) {
            return null;
        }
        if (!value.isTextual()) {
            throw new IllegalArgumentException("optional field must be text: " + fieldName);
        }
        return value.asText();
    }

    private static double optionalNumber(JsonNode node, String fieldName, double defaultValue) {
        JsonNode value = node.get(fieldName);
        if (value == null || value.isNull()) {
            return defaultValue;
        }
        if (!value.isNumber()) {
            throw new IllegalArgumentException("optional field must be numeric: " + fieldName);
        }

        double number = value.asDouble();
        if (!Double.isFinite(number)) {
            throw new IllegalArgumentException("numeric field must be finite: " + fieldName);
        }
        return number;
    }

    private static String stringify(JsonNode node) {
        try {
            return MAPPER.writeValueAsString(node);
        } catch (JsonProcessingException error) {
            throw new IllegalStateException("failed to serialize JSON", error);
        }
    }

    private static String featureKey(int index) {
        return String.format("f%03d", index);
    }
}
