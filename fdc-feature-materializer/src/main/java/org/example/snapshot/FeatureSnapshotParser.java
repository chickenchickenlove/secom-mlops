package org.example.snapshot;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Set;
import java.util.regex.Pattern;

public final class FeatureSnapshotParser {
    private static final int NUM_FEATURES = 590;
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern SAMPLE_ID_PATTERN = Pattern.compile("^secom-\\d{7}$");
    private static final Set<String> SNAPSHOT_STATUSES = Set.of("partial", "complete");

    private FeatureSnapshotParser() {
    }

    public static ServingSnapshotRow parse(String raw, String kafkaKey) {
        if (raw == null) {
            throw new IllegalArgumentException("Kafka message value is null");
        }

        final JsonNode snapshot = parseJson(raw);
        return parseSnapshot(snapshot, kafkaKey);
    }

    private static JsonNode parseJson(String raw) {
        try {
            final JsonNode parsed = MAPPER.readTree(raw);
            if (parsed == null || !parsed.isObject()) {
                throw new IllegalArgumentException("Kafka message value must be a JSON object");
            }
            return parsed;
        } catch (JsonProcessingException error) {
            throw new IllegalArgumentException("invalid JSON payload", error);
        }
    }

    private static ServingSnapshotRow parseSnapshot(JsonNode snapshot, String kafkaKey) {
        final String servingSnapshotId = requiredText(snapshot, "serving_snapshot_id");
        final String sampleId = requiredText(snapshot, "sample_id");
        final long sourceEventCount = requiredLong(snapshot, "source_event_count");
        final long snapshotVersion = requiredLong(snapshot, "snapshot_version");
        final double snapshotTime = requiredNumber(snapshot, "snapshot_time");
        final double windowStart = requiredNumber(snapshot, "window_start");
        final double windowEnd = requiredNumber(snapshot, "window_end");
        final String snapshotStatus = requiredText(snapshot, "snapshot_status");
        final int featureCount = requiredInt(snapshot, "feature_count");
        final int missingCount = requiredInt(snapshot, "missing_count");
        final boolean isComplete = requiredBoolean(snapshot, "is_complete");
        final JsonNode features = requiredObject(snapshot, "features");

        validateIdentity(sampleId, kafkaKey);
        validateSnapshotVersion(sourceEventCount, snapshotVersion);
        validateWindow(snapshotTime, windowStart, windowEnd);
        validateCompleteness(snapshotStatus, featureCount, missingCount, isComplete);
        validateCanonicalFeatures(sampleId, features, missingCount);

        return new ServingSnapshotRow(
            servingSnapshotId,
            snapshotVersion,
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
            optionalText(snapshot, "drift_segment")
        );
    }

    private static void validateIdentity(String sampleId, String kafkaKey) {
        if (!SAMPLE_ID_PATTERN.matcher(sampleId).matches()) {
            throw new IllegalArgumentException("invalid sample_id: " + sampleId);
        }
        if (kafkaKey != null && !kafkaKey.equals(sampleId)) {
            throw new IllegalArgumentException(
                "Kafka key differs from sample_id: key=" + kafkaKey + " sample_id=" + sampleId
            );
        }
    }

    private static void validateWindow(double snapshotTime, double windowStart, double windowEnd) {
        if (snapshotTime < 0.0) {
            throw new IllegalArgumentException("snapshot_time must be >= 0");
        }
        if (windowStart < 0.0) {
            throw new IllegalArgumentException("window_start must be >= 0");
        }
        if (windowEnd < 0.0) {
            throw new IllegalArgumentException("window_end must be >= 0");
        }
        if (windowEnd < windowStart) {
            throw new IllegalArgumentException("window_end must be >= window_start");
        }
    }

    private static void validateSnapshotVersion(long sourceEventCount, long snapshotVersion) {
        if (sourceEventCount < 1) {
            throw new IllegalArgumentException("source_event_count must be >= 1");
        }
        if (snapshotVersion < 1) {
            throw new IllegalArgumentException("snapshot_version must be >= 1");
        }
        if (snapshotVersion != sourceEventCount) {
            throw new IllegalArgumentException(
                "snapshot_version must match source_event_count: snapshot_version="
                    + snapshotVersion
                    + " source_event_count="
                    + sourceEventCount
            );
        }
    }

    private static void validateCompleteness(
        String snapshotStatus,
        int featureCount,
        int missingCount,
        boolean isComplete
    ) {
        if (!SNAPSHOT_STATUSES.contains(snapshotStatus)) {
            throw new IllegalArgumentException("snapshot_status must be partial or complete: " + snapshotStatus);
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

        final String expectedStatus = isComplete ? "complete" : "partial";
        if (!snapshotStatus.equals(expectedStatus)) {
            throw new IllegalArgumentException("snapshot_status must be " + expectedStatus);
        }
    }

    private static void validateCanonicalFeatures(String sampleId, JsonNode features, int missingCount) {
        if (features.size() != NUM_FEATURES) {
            throw new IllegalArgumentException("features must contain exactly 590 canonical keys: " + features.size());
        }

        int observedMissingCount = 0;
        for (int index = 0; index < NUM_FEATURES; index++) {
            String key = featureKey(index);
            JsonNode value = features.get(key);

            if (value == null) {
                throw new IllegalArgumentException("missing canonical feature key: sample_id=" + sampleId + " key=" + key);
            }
            if (!value.isNull() && !value.isNumber()) {
                throw new IllegalArgumentException(
                    "feature value must be numeric or null: sample_id=" + sampleId + " key=" + key
                );
            }
            if (value.isNumber() && !Double.isFinite(value.asDouble())) {
                throw new IllegalArgumentException("feature value must be finite: sample_id=" + sampleId + " key=" + key);
            }
            if (value.isNull()) {
                observedMissingCount++;
            }
        }

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
    }

    private static String requiredText(JsonNode node, String fieldName) {
        final JsonNode value = node.get(fieldName);
        if (value == null || !value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException("required text field missing or invalid: " + fieldName);
        }
        return value.asText();
    }

    private static double requiredNumber(JsonNode node, String fieldName) {
        final JsonNode value = node.get(fieldName);
        if (value == null || !value.isNumber()) {
            throw new IllegalArgumentException("required numeric field missing or invalid: " + fieldName);
        }

        final double number = value.asDouble();
        if (!Double.isFinite(number)) {
            throw new IllegalArgumentException("numeric field must be finite: " + fieldName);
        }
        return number;
    }

    private static int requiredInt(JsonNode node, String fieldName) {
        final JsonNode value = node.get(fieldName);
        if (value == null || !value.isInt()) {
            throw new IllegalArgumentException("required integer field missing or invalid: " + fieldName);
        }
        return value.asInt();
    }

    private static long requiredLong(JsonNode node, String fieldName) {
        final JsonNode value = node.get(fieldName);
        if (value == null || !value.isIntegralNumber() || !value.canConvertToLong()) {
            throw new IllegalArgumentException("required integer field missing or invalid: " + fieldName);
        }
        return value.asLong();
    }

    private static boolean requiredBoolean(JsonNode node, String fieldName) {
        final JsonNode value = node.get(fieldName);
        if (value == null || !value.isBoolean()) {
            throw new IllegalArgumentException("required boolean field missing or invalid: " + fieldName);
        }
        return value.asBoolean();
    }

    private static JsonNode requiredObject(JsonNode node, String fieldName) {
        final JsonNode value = node.get(fieldName);
        if (value == null || !value.isObject()) {
            throw new IllegalArgumentException("required object field missing or invalid: " + fieldName);
        }
        return value;
    }

    private static String optionalText(JsonNode node, String fieldName) {
        final JsonNode value = node.get(fieldName);
        if (value == null || value.isNull()) {
            return null;
        }
        if (!value.isTextual()) {
            throw new IllegalArgumentException("optional field must be text: " + fieldName);
        }
        return value.asText();
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
