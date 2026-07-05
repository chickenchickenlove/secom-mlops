package org.example;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Set;
import java.util.regex.Pattern;

final class FeatureEventParser {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern SAMPLE_ID_PATTERN = Pattern.compile("^secom-\\d{7}$");
    private static final Pattern FEATURE_KEY_PATTERN = Pattern.compile("^f\\d{3}$");
    private static final int NUM_FEATURES = 590;
    private static final Set<String> FEATURE_GROUPS = Set.of("early", "middle", "late");

    private FeatureEventParser() {
    }

    static FeatureEventRow parse(String raw, String kafkaKey) {
        if (raw == null) {
            throw new IllegalArgumentException("Kafka message value is null");
        }

        JsonNode event = parseJson(raw);
        return validateFeatureEvent(event, kafkaKey);
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

    private static FeatureEventRow validateFeatureEvent(JsonNode event, String kafkaKey) {
        String eventId = requiredText(event, "event_id");
        String sampleId = requiredText(event, "sample_id");
        double eventTime = requiredNumber(event, "event_time");
        String featureGroup = requiredText(event, "feature_group");
        JsonNode features = requiredObject(event, "features");

        if (!SAMPLE_ID_PATTERN.matcher(sampleId).matches()) {
            throw new IllegalArgumentException("invalid sample_id: " + sampleId);
        }
        if (kafkaKey != null && !kafkaKey.equals(sampleId)) {
            throw new IllegalArgumentException(
                "Kafka key differs from sample_id: key=" + kafkaKey + " sample_id=" + sampleId
            );
        }
        if (eventTime < 0.0) {
            throw new IllegalArgumentException("event_time must be >= 0");
        }
        if (!FEATURE_GROUPS.contains(featureGroup)) {
            throw new IllegalArgumentException("unexpected feature_group: " + featureGroup);
        }

        validateFeatures(sampleId, features);

        return new FeatureEventRow(
            eventId,
            sampleId,
            eventTime,
            featureGroup,
            stringify(features),
            optionalText(event, "simulation_run_id"),
            optionalText(event, "drift_segment"),
            optionalNumber(event, "created_at", System.currentTimeMillis() / 1000.0)
        );
    }

    private static void validateFeatures(String sampleId, JsonNode features) {
        if (features.size() == 0) {
            throw new IllegalArgumentException("features must not be empty: sample_id=" + sampleId);
        }

        features.fields().forEachRemaining(entry -> {
            String key = entry.getKey();
            JsonNode value = entry.getValue();

            if (!FEATURE_KEY_PATTERN.matcher(key).matches()) {
                throw new IllegalArgumentException("invalid feature key: sample_id=" + sampleId + " key=" + key);
            }

            int index = Integer.parseInt(key.substring(1));
            if (index < 0 || index >= NUM_FEATURES) {
                throw new IllegalArgumentException("feature key out of range: sample_id=" + sampleId + " key=" + key);
            }

            if (!value.isNull() && !value.isNumber()) {
                throw new IllegalArgumentException(
                    "feature value must be numeric or null: sample_id=" + sampleId + " key=" + key
                );
            }

            if (value.isNumber() && !Double.isFinite(value.asDouble())) {
                throw new IllegalArgumentException("feature value must be finite: sample_id=" + sampleId + " key=" + key);
            }
        });
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
}
