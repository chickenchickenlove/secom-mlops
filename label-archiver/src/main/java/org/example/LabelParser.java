package org.example;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.regex.Pattern;

final class LabelParser {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern SAMPLE_ID_PATTERN = Pattern.compile("^secom-\\d{7}$");

    private LabelParser() {
    }

    static LabelRow parse(String raw, String kafkaKey) {
        if (raw == null) {
            throw new IllegalArgumentException("Kafka message value is null");
        }

        JsonNode event = parseJson(raw);
        return validateLabelEvent(event, kafkaKey);
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

    private static LabelRow validateLabelEvent(JsonNode event, String kafkaKey) {
        String labelEventId = requiredText(event, "label_event_id");
        String sampleId = requiredText(event, "sample_id");

        if (!SAMPLE_ID_PATTERN.matcher(sampleId).matches()) {
            throw new IllegalArgumentException("invalid sample_id: " + sampleId);
        }

        if (kafkaKey != null && !kafkaKey.equals(sampleId)) {
            throw new IllegalArgumentException(
                "Kafka key differs from sample_id: key=" + kafkaKey + " sample_id=" + sampleId
            );
        }

        long labelRevision = requiredLong(event, "label_revision");
        double measuredAt = requiredNumber(event, "measured_at");

        if (labelRevision <= 0) {
            throw new IllegalArgumentException(
                "label_revision must be > 0: " + labelRevision
            );
        }

        if (measuredAt < 0.0) {
            throw new IllegalArgumentException("measured_at must be >= 0: " + measuredAt);
        }

        int actualValue = requiredInt(event, "actual_value");
        String actualLabel = requiredText(event, "actual_label");

        if (actualValue != -1 && actualValue != 1) {
            throw new IllegalArgumentException("actual_value must be -1 or 1: " + actualValue);
        }

        String expectedLabel = actualValue == 1 ? "fail" : "pass";
        if (!actualLabel.equals(expectedLabel)) {
            throw new IllegalArgumentException(
                "actual_label mismatch: actual_value=" + actualValue + " actual_label=" + actualLabel
            );
        }

        return new LabelRow(
            labelEventId,
            sampleId,
            labelRevision,
            measuredAt,
            actualValue,
            actualLabel
        );
    }

    private static String requiredText(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException("required text field missing or invalid: " + fieldName);
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

    private static long requiredLong(JsonNode node, String fieldName) {
        JsonNode value = node.get(fieldName);
        if (value == null || !value.isIntegralNumber() || !value.canConvertToLong()) {
            throw new IllegalArgumentException(
                "required integer field missing or invalid: " + fieldName
            );
        }
        return value.asLong();
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
}
