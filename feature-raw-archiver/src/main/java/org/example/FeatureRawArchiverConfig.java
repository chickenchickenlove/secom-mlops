package org.example;

record FeatureRawArchiverConfig(
    String bootstrapServers,
    String topic,
    String groupId,
    String clientId,
    String autoOffsetReset,
    int pollTimeoutMs,
    int maxPollRecords,
    String dbUrl,
    String dbUser,
    String dbPassword
) {
    static FeatureRawArchiverConfig fromEnv() {
        return new FeatureRawArchiverConfig(
            requiredEnv("KAFKA_BOOTSTRAP_SERVERS"),
            requiredEnv("FEATURE_PATCHES_TOPIC"),
            requiredEnv("KAFKA_GROUP_ID"),
            requiredEnv("KAFKA_CLIENT_ID"),
            optionalEnv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
            optionalIntEnv("KAFKA_POLL_TIMEOUT_MS", 1000),
            optionalIntEnv("KAFKA_MAX_POLL_RECORDS", 100),
            requiredEnv("MONITORING_JDBC_URL"),
            requiredEnv("MONITORING_DB_USER"),
            requiredEnv("MONITORING_DB_PASSWORD")
        );
    }

    private static String requiredEnv(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("required environment variable missing: " + name);
        }
        return value;
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
}
