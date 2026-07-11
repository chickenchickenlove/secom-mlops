package org.example;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.common.errors.WakeupException;
import org.example.config.MaterializerConfig;
import org.example.kafka.KafkaFeatureStateConsumer;
import org.example.store.PostgresServingSnapshotSink;
import org.example.store.ServingSnapshotSink;
import org.example.store.SnapshotStore;
import org.example.store.ValkeySnapshotStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;

public final class FeatureStateMaterializerApp {
    private static final Logger LOG = LoggerFactory.getLogger(FeatureStateMaterializerApp.class);

    private final MaterializerConfig config;
    private final AtomicBoolean running = new AtomicBoolean(true);

    private FeatureStateMaterializerApp(MaterializerConfig config) {
        this.config = config;
    }

    public static void main(String[] args) {
        final MaterializerConfig config = MaterializerConfig.fromEnv();
        new FeatureStateMaterializerApp(config).run();
    }

    private void run() {
        try (
            final KafkaFeatureStateConsumer consumer = new KafkaFeatureStateConsumer(config);
            final SnapshotStore store = new ValkeySnapshotStore(config);
            final ServingSnapshotSink snapshotSink = new PostgresServingSnapshotSink(config)
        ) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                running.set(false);
                consumer.wakeup();
            }));

            store.verifyConnection();
            snapshotSink.verifyConnection();

            consumer.subscribe();

            FeatureStateRecordHandler recordHandler = new FeatureStateRecordHandler(
                store,
                snapshotSink,
                consumer::commit,
                config.keyPrefix()
            );

            LOG.info(
                "materializer_started topic={} group_id={} bootstrap_servers={} valkey={}:{}/{} db_url={}",
                config.topic(),
                config.groupId(),
                config.bootstrapServers(),
                config.valkeyHost(),
                config.valkeyPort(),
                config.valkeyDatabase(),
                config.dbUrl()
            );

            while (running.get()) {
                final ConsumerRecords<String, String> records = consumer.poll();

                for (ConsumerRecord<String, String> record : records) {
                    recordHandler.handle(record);
                }
            }

            LOG.info("materializer_stopped");
        } catch (WakeupException error) {
            if (running.get()) {
                throw error;
            }
            LOG.info("materializer_stopped");
        } catch (RuntimeException error) {
            LOG.error("materializer_failed", error);
            throw error;
        }
    }

}
