package org.example.processor;

import com.fasterxml.jackson.databind.JsonNode;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

final class FeatureVectorHash {
    static final String HASH_PREFIX = "sha256:v1:";

    private static final int NUM_FEATURES = 590;
    private static final byte NULL_MARKER = 0x00;
    private static final byte NUMBER_MARKER = 0x01;
    private static final byte[] DOMAIN_PREFIX =
        "secom-feature-vector:v1\u0000".getBytes(StandardCharsets.UTF_8);

    private FeatureVectorHash() {
    }

    static String compute(JsonNode features) {
        if (features == null || !features.isObject()) {
            throw new IllegalArgumentException("features must be a JSON object");
        }
        if (features.size() != NUM_FEATURES) {
            throw new IllegalArgumentException(
                "features must contain exactly 590 canonical keys: " + features.size()
            );
        }

        MessageDigest digest = newSha256Digest();
        digest.update(DOMAIN_PREFIX);
        ByteBuffer numberBytes = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.BIG_ENDIAN);

        for (int index = 0; index < NUM_FEATURES; index++) {
            String key = featureKey(index);
            JsonNode value = features.get(key);

            if (value == null) {
                throw new IllegalArgumentException("missing canonical feature key: " + key);
            }
            if (value.isNull()) {
                digest.update(NULL_MARKER);
                continue;
            }
            if (!value.isNumber()) {
                throw new IllegalArgumentException("feature value must be numeric or null: key=" + key);
            }

            double number = value.asDouble();
            if (!Double.isFinite(number)) {
                throw new IllegalArgumentException("feature value must be finite: key=" + key);
            }
            if (number == 0.0d) {
                number = 0.0d;
            }

            digest.update(NUMBER_MARKER);
            numberBytes.clear();
            numberBytes.putLong(Double.doubleToLongBits(number));
            digest.update(numberBytes.array());
        }

        return HASH_PREFIX + HexFormat.of().formatHex(digest.digest());
    }

    private static MessageDigest newSha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }

    private static String featureKey(int index) {
        return String.format("f%03d", index);
    }
}
