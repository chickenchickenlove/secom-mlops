package org.example;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.common.errors.WakeupException;
import org.example.config.MaterializerConfig;
import org.example.kafka.KafkaFeatureStateConsumer;
import org.example.snapshot.FeatureSnapshotValidator;
import org.example.store.SnapshotStore;
import org.example.store.ValkeySnapshotStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;

public final class FeatureStateMaterializerApp {
    private static final Logger LOG = LoggerFactory.getLogger(FeatureStateMaterializerApp.class);

    private final MaterializerConfig config;
    private final FeatureSnapshotValidator validator;
    private final AtomicBoolean running = new AtomicBoolean(true);

    private FeatureStateMaterializerApp(MaterializerConfig config) {
        this.config = config;
        this.validator = new FeatureSnapshotValidator();
    }

    public static void main(String[] args) {
        MaterializerConfig config = MaterializerConfig.fromEnv();
        new FeatureStateMaterializerApp(config).run();
    }

    private void run() {
        try (
            KafkaFeatureStateConsumer consumer = new KafkaFeatureStateConsumer(config);
            SnapshotStore store = new ValkeySnapshotStore(config)
        ) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                running.set(false);
                consumer.wakeup();
            }));

            store.verifyConnection();

            consumer.subscribe();

            LOG.info(
                "materializer_started topic={} group_id={} bootstrap_servers={} valkey={}:{}/{}",
                config.topic(),
                config.groupId(),
                config.bootstrapServers(),
                config.valkeyHost(),
                config.valkeyPort(),
                config.valkeyDatabase()
            );

            while (running.get()) {
                ConsumerRecords<String, String> records = consumer.poll();

                for (ConsumerRecord<String, String> record : records) {
                    materializeThenCommit(consumer, store, record);
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

    private void materializeThenCommit(
        KafkaFeatureStateConsumer consumer,
        SnapshotStore store,
        ConsumerRecord<String, String> record
    ) {
        String sampleId = validator.validate(record.value(), record.key());
        String valkeyKey = config.keyPrefix() + ":" + sampleId;

        store.put(valkeyKey, record.value());
        consumer.commit(record);

        LOG.debug(
            "materialized sample_id={} key={} topic={} partition={} offset={} committed_offset={}",
            sampleId,
            valkeyKey,
            record.topic(),
            record.partition(),
            record.offset(),
            record.offset() + 1
        );
    }
}
