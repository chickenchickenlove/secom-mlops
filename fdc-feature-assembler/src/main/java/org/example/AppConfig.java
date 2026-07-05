package org.example;

public final class AppConfig {
    private final String bootstrapServers;
    private final String applicationId;
    private final String inputTopic;
    private final String outputTopic;
    private final String stateDir;
    private final String autoOffsetReset;

    private AppConfig(
        String bootstrapServers,
        String applicationId,
        String inputTopic,
        String outputTopic,
        String stateDir,
        String autoOffsetReset
    ) {
        this.bootstrapServers = bootstrapServers;
        this.applicationId = applicationId;
        this.inputTopic = inputTopic;
        this.outputTopic = outputTopic;
        this.stateDir = stateDir;
        this.autoOffsetReset = autoOffsetReset;
    }

    public static AppConfig fromEnv() {
        return new AppConfig(
            requiredEnv("KAFKA_BOOTSTRAP_SERVERS"),
            requiredEnv("FEATURE_ASSEMBLER_APPLICATION_ID"),
            requiredEnv("FEATURE_PATCHES_TOPIC"),
            requiredEnv("FEATURE_STATE_UPDATES_TOPIC"),
            requiredEnv("FEATURE_ASSEMBLER_STATE_DIR"),
            optionalEnv("KAFKA_AUTO_OFFSET_RESET", "latest")
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

    public String bootstrapServers() {
        return bootstrapServers;
    }

    public String applicationId() {
        return applicationId;
    }

    public String inputTopic() {
        return inputTopic;
    }

    public String outputTopic() {
        return outputTopic;
    }

    public String stateDir() {
        return stateDir;
    }

    public String autoOffsetReset() {
        return autoOffsetReset;
    }
}
