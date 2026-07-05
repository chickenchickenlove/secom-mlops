package org.example.config;

public record MaterializerConfig(
    String bootstrapServers,
    String topic,
    String groupId,
    String clientId,
    String autoOffsetReset,
    int pollTimeoutMs,
    int maxPollRecords,
    String valkeyHost,
    int valkeyPort,
    int valkeyDatabase,
    int valkeyTimeoutMs,
    int valkeyPoolMaxTotal,
    int valkeyPoolMaxIdle,
    String keyPrefix,
    boolean verifyWrite
) {
    public static MaterializerConfig fromEnv() {
        return new MaterializerConfig(
            requiredEnv("KAFKA_BOOTSTRAP_SERVERS"),
            requiredEnv("FEATURE_STATE_UPDATES_TOPIC"),
            requiredEnv("KAFKA_GROUP_ID"),
            requiredEnv("KAFKA_CLIENT_ID"),
            optionalEnv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
            optionalIntEnv("KAFKA_POLL_TIMEOUT_MS", 1000),
            optionalIntEnv("KAFKA_MAX_POLL_RECORDS", 100),
            requiredEnv("VALKEY_HOST"),
            requiredIntEnv("VALKEY_PORT"),
            optionalIntEnv("VALKEY_DATABASE", 0),
            optionalIntEnv("VALKEY_TIMEOUT_MS", 2000),
            optionalIntEnv("VALKEY_POOL_MAX_TOTAL", 8),
            optionalIntEnv("VALKEY_POOL_MAX_IDLE", 8),
            optionalEnv("VALKEY_KEY_PREFIX", "online_feature_snapshot"),
            optionalBooleanEnv("VALKEY_VERIFY_WRITE", true)
        );
    }

    private static String requiredEnv(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("required environment variable missing: " + name);
        }
        return value;
    }

    private static int requiredIntEnv(String name) {
        return Integer.parseInt(requiredEnv(name));
    }

    private static String optionalEnv(String name, String defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        return value;
    }

    private static int optionalIntEnv(String name, int defaultValue) {
        return Integer.parseInt(optionalEnv(name, Integer.toString(defaultValue)));
    }

    private static boolean optionalBooleanEnv(String name, boolean defaultValue) {
        return Boolean.parseBoolean(optionalEnv(name, Boolean.toString(defaultValue)));
    }
}
