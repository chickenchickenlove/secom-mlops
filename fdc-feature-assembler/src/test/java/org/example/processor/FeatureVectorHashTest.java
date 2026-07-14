package org.example.processor;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeatureVectorHashTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void computesVersionedSha256IndependentOfObjectInsertionOrder() {
        ObjectNode ascending = features(false);
        ObjectNode descending = features(true);

        String ascendingHash = FeatureVectorHash.compute(ascending);
        String descendingHash = FeatureVectorHash.compute(descending);

        assertEquals(ascendingHash, descendingHash);
        assertTrue(ascendingHash.matches("^sha256:v1:[0-9a-f]{64}$"));
    }

    @Test
    void distinguishesMissingFromNumericValues() {
        ObjectNode missing = features(false);
        missing.putNull("f123");

        ObjectNode numeric = features(false);
        numeric.put("f123", 0.0d);

        assertNotEquals(
            FeatureVectorHash.compute(missing),
            FeatureVectorHash.compute(numeric)
        );
    }

    @Test
    void normalizesNegativeZeroToPositiveZero() {
        ObjectNode negativeZero = features(false);
        negativeZero.put("f123", -0.0d);

        ObjectNode positiveZero = features(false);
        positiveZero.put("f123", 0.0d);

        assertEquals(
            FeatureVectorHash.compute(negativeZero),
            FeatureVectorHash.compute(positiveZero)
        );
    }

    @Test
    void rejectsNonFiniteValues() {
        ObjectNode features = features(false);
        features.put("f123", Double.POSITIVE_INFINITY);

        IllegalArgumentException error = assertThrows(
            IllegalArgumentException.class,
            () -> FeatureVectorHash.compute(features)
        );

        assertEquals("feature value must be finite: key=f123", error.getMessage());
    }

    private static ObjectNode features(boolean descending) {
        ObjectNode features = MAPPER.createObjectNode();

        for (int offset = 0; offset < 590; offset++) {
            int index = descending ? 589 - offset : offset;
            features.put(String.format("f%03d", index), (double) index);
        }

        return features;
    }
}
