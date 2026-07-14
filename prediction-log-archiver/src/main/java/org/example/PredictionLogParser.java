package org.example;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.regex.Pattern;

final class PredictionLogParser {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern SAMPLE_ID_PATTERN = Pattern.compile("^secom-\\d{7}$");
    private static final Pattern FEATURE_HASH_PATTERN = Pattern.compile("^sha256:v1:[0-9a-f]{64}$");
    private static final int MAX_MISSING_COUNT = 590;

    private PredictionLogParser() {
    }

    static PredictionLogRow parse(String raw, String kafkaKey) {
        JsonNode event = parseJson(raw);
        return validatePredictionEvent(event, kafkaKey);
    }

    private static JsonNode parseJson(String raw) {
        if (raw == null) {
            throw new IllegalArgumentException("Kafka message value is null");
        }

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

    private static PredictionLogRow validatePredictionEvent(JsonNode event, String kafkaKey) {
        String predictionId = requiredText(event, "prediction_id");
        String requestId = requiredText(event, "request_id");
        String sampleId = requiredText(event, "sample_id");
        String servingSnapshotId = requiredText(event, "serving_snapshot_id");
        long snapshotVersion = requiredPositiveLong(event, "snapshot_version");
        String featureHash = requiredText(event, "feature_hash");

        if (!SAMPLE_ID_PATTERN.matcher(sampleId).matches()) {
            throw new IllegalArgumentException("invalid sample_id: " + sampleId);
        }
        if (kafkaKey != null && !kafkaKey.equals(sampleId)) {
            throw new IllegalArgumentException(
                "Kafka key differs from sample_id: key=" + kafkaKey + " sample_id=" + sampleId
            );
        }
        if (!FEATURE_HASH_PATTERN.matcher(featureHash).matches()) {
            throw new IllegalArgumentException("invalid feature_hash: " + featureHash);
        }

        String modelRunId = requiredText(event, "model_run_id");
        String modelName = optionalText(event, "model_name");
        String modelVersion = optionalText(event, "model_version");
        String modelAlias = optionalText(event, "model_alias");
        String modelUri = optionalText(event, "model_uri");
        String runtimeSlot = optionalText(event, "runtime_slot");
        if (runtimeSlot == null) {
            runtimeSlot = "unknown";
        }

        double predictedAt = requiredNonNegativeNumber(event, "predicted_at");
        double failProbability = requiredProbability(event, "fail_probability");
        int predictedValue = requiredInt(event, "predicted_value");
        String predictedLabel = requiredText(event, "predicted_label");
        double threshold = requiredProbability(event, "threshold");
        int missingCount = requiredInt(event, "missing_count");
        double latencyMs = requiredNonNegativeNumber(event, "latency_ms");

        if (predictedValue != -1 && predictedValue != 1) {
            throw new IllegalArgumentException("predicted_value must be -1 or 1: " + predictedValue);
        }

        String expectedLabel = predictedValue == 1 ? "fail" : "pass";
        if (!predictedLabel.equals(expectedLabel)) {
            throw new IllegalArgumentException(
                "predicted_label mismatch: predicted_value=" + predictedValue + " predicted_label=" + predictedLabel
            );
        }

        if (missingCount < 0 || missingCount > MAX_MISSING_COUNT) {
            throw new IllegalArgumentException(
                "missing_count must be between 0 and 590: sample_id=" + sampleId
                    + " missing_count=" + missingCount
            );
        }

        return new PredictionLogRow(
            predictionId,
            requestId,
            sampleId,
            servingSnapshotId,
            snapshotVersion,
            featureHash,
            modelRunId,
            modelName,
            modelVersion,
            modelAlias,
            modelUri,
            runtimeSlot,
            predictedAt,
            failProbability,
            predictedValue,
            predictedLabel,
            threshold,
            missingCount,
            latencyMs
        );
    }

    private static String requiredText(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException("required text field missing or invalid: " + fieldName);
        }
        return value.asText();
    }

    private static String optionalText(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || value.isNull()) {
            return null;
        }
        if (!value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException("optional text field invalid: " + fieldName);
        }
        return value.asText();
    }

    private static int requiredInt(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isInt()) {
            throw new IllegalArgumentException("required integer field missing or invalid: " + fieldName);
        }
        return value.asInt();
    }

    private static long requiredPositiveLong(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isIntegralNumber() || !value.canConvertToLong()) {
            throw new IllegalArgumentException("required long field missing or invalid: " + fieldName);
        }

        long number = value.asLong();
        if (number < 1L) {
            throw new IllegalArgumentException("long field must be >= 1: " + fieldName);
        }
        return number;
    }

    private static double requiredProbability(JsonNode node, String fieldName) {
        double value = requiredNonNegativeNumber(node, fieldName);
        if (value > 1.0) {
            throw new IllegalArgumentException("probability field must be <= 1.0: " + fieldName);
        }
        return value;
    }

    private static double requiredNonNegativeNumber(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isNumber()) {
            throw new IllegalArgumentException("required numeric field missing or invalid: " + fieldName);
        }

        double number = value.asDouble();
        if (!Double.isFinite(number) || number < 0.0) {
            throw new IllegalArgumentException("numeric field must be finite and >= 0: " + fieldName);
        }
        return number;
    }

}
