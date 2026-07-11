package org.example;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.example.snapshot.FeatureSnapshotParser;
import org.example.snapshot.ServingSnapshotRow;
import org.example.store.ServingSnapshotSink;
import org.example.store.SnapshotWriteResult;
import org.example.store.SnapshotStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.function.Consumer;

public final class FeatureStateRecordHandler {
    private static final Logger LOG = LoggerFactory.getLogger(FeatureStateRecordHandler.class);

    private final SnapshotStore store;
    private final ServingSnapshotSink snapshotSink;
    private final Consumer<ConsumerRecord<String, String>> offsetCommitter;
    private final String keyPrefix;

    FeatureStateRecordHandler(
        SnapshotStore store,
        ServingSnapshotSink snapshotSink,
        Consumer<ConsumerRecord<String, String>> offsetCommitter,
        String keyPrefix
    ) {
        this.store = Objects.requireNonNull(store, "store");
        this.snapshotSink = Objects.requireNonNull(snapshotSink, "snapshotSink");
        this.offsetCommitter = Objects.requireNonNull(offsetCommitter, "offsetCommitter");
        this.keyPrefix = Objects.requireNonNull(keyPrefix, "keyPrefix");
    }

    void handle(ConsumerRecord<String, String> record) {
        final ServingSnapshotRow row = FeatureSnapshotParser.parse(record.value(), record.key());
        final String valkeyKey = keyPrefix + ":" + row.sampleId();

        final SnapshotWriteResult writeResult = store.put(valkeyKey, record.value());
        final int inserted = snapshotSink.persist(row, writeResult.availableAt());
        offsetCommitter.accept(record);

        LOG.debug(
            "materialized_and_archived snapshot_id={} snapshot_version={} sample_id={} status={} key={} write_available_at={} inserted={} topic={} partition={} offset={} committed_offset={}",
            row.servingSnapshotId(),
            row.snapshotVersion(),
            row.sampleId(),
            row.snapshotStatus(),
            valkeyKey,
            writeResult.availableAt(),
            inserted,
            record.topic(),
            record.partition(),
            record.offset(),
            record.offset() + 1
        );
    }
}
