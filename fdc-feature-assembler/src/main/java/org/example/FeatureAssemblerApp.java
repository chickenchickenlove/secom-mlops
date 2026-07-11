package org.example;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.Topology;
import org.apache.kafka.streams.errors.StreamsUncaughtExceptionHandler;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.state.StoreBuilder;
import org.apache.kafka.streams.state.Stores;
import org.example.processor.FeatureStateProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;

public final class FeatureAssemblerApp {
    private static final Logger LOG = LoggerFactory.getLogger(FeatureAssemblerApp.class);

    private FeatureAssemblerApp() {
    }

    public static void main(String[] args) {
        final AppConfig config = AppConfig.fromEnv();
        final Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, config.applicationId());
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, config.bootstrapServers());
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.StringSerde.class.getName());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.StringSerde.class.getName());
        props.put(StreamsConfig.STATE_DIR_CONFIG, config.stateDir());
        props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, "1000");
        props.put(StreamsConfig.consumerPrefix("auto.offset.reset"), config.autoOffsetReset());

        CountDownLatch shutdownLatch = new CountDownLatch(1);

        try (KafkaStreams streams = new KafkaStreams(buildTopology(config), props)) {
            Runtime
                .getRuntime()
                .addShutdownHook(new Thread(() -> {
                    shutdownLatch.countDown();
                    streams.close(Duration.ofSeconds(10));
                }));

            streams.setUncaughtExceptionHandler(error -> {
                LOG.error("feature_assembler_stream_thread_failed", error);
                shutdownLatch.countDown();
                return StreamsUncaughtExceptionHandler.StreamThreadExceptionResponse.SHUTDOWN_APPLICATION;
            });

            LOG.info(
                "feature_assembler_started input_topic={} output_topic={} application_id={} bootstrap_servers={} state_dir={}",
                config.inputTopic(),
                config.outputTopic(),
                config.applicationId(),
                config.bootstrapServers(),
                config.stateDir()
            );
            streams.start();
            shutdownLatch.await();
            LOG.info("feature_assembler_stopped");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    private static Topology buildTopology(AppConfig config) {

        final String storeName = "secom-feature-state-store";
        StoreBuilder<KeyValueStore<String, String>> storeBuilder = Stores.keyValueStoreBuilder(
            Stores.persistentKeyValueStore(storeName),
            Serdes.String(),
            Serdes.String()
        );

        final Topology topology = new Topology();

        topology.addSource(
            "feature-patch-source",
            Serdes.String().deserializer(),
            Serdes.String().deserializer(),
            config.inputTopic()
        );
        topology.addProcessor(
            "feature-state-processor",
            () -> new FeatureStateProcessor(storeName),
            "feature-patch-source"
        );

        topology.addStateStore(storeBuilder, "feature-state-processor");
        topology.addSink(
            "feature-state-sink",
            config.outputTopic(),
            Serdes.String().serializer(),
            Serdes.String().serializer(),
            "feature-state-processor"
        );

        return topology;
    }
}
