package org.example;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.example.snapshot.ServingSnapshotRow;
import org.example.store.ServingSnapshotSink;
import org.example.store.SnapshotWriteResult;
import org.example.store.SnapshotStore;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FeatureStateRecordHandlerTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String FEATURE_HASH = "sha256:v1:" + "0".repeat(64);

    @Test
    void writesValkeyThenDatabaseThenCommitsKafkaOffset() {
        List<String> calls = new ArrayList<>();
        RecordingSnapshotStore store = new RecordingSnapshotStore(calls, null);
        RecordingSnapshotSink sink = new RecordingSnapshotSink(calls, null);
        FeatureStateRecordHandler handler = new FeatureStateRecordHandler(
            store,
            sink,
            ignored -> calls.add("kafka_commit"),
            "online_feature_snapshot"
        );

        handler.handle(validRecord());

        assertEquals(List.of("valkey_put", "db_commit", "kafka_commit"), calls);
        assertEquals("online_feature_snapshot:secom-0000001", store.lastKey);
        assertEquals("state:secom-0000001:1000:3", sink.lastRow.servingSnapshotId());
        assertEquals(3L, sink.lastRow.snapshotVersion());
        assertEquals(FEATURE_HASH, sink.lastRow.featureHash());
        assertEquals(123.456, sink.lastAvailableAt);
    }

    @Test
    void parseFailureHasNoSideEffects() {
        List<String> calls = new ArrayList<>();
        FeatureStateRecordHandler handler = handler(calls, null, null);
        ConsumerRecord<String, String> record = new ConsumerRecord<>(
            "secom-feature-state-updates",
            0,
            10L,
            "secom-0000001",
            "{}"
        );

        assertThrows(IllegalArgumentException.class, () -> handler.handle(record));
        assertEquals(List.of(), calls);
    }

    @Test
    void valkeyFailureSkipsDatabaseAndKafkaCommit() {
        List<String> calls = new ArrayList<>();
        RuntimeException failure = new IllegalStateException("valkey unavailable");
        FeatureStateRecordHandler handler = handler(calls, failure, null);

        IllegalStateException error = assertThrows(
            IllegalStateException.class,
            () -> handler.handle(validRecord())
        );

        assertEquals(failure, error);
        assertEquals(List.of("valkey_put"), calls);
    }

    @Test
    void databaseFailureSkipsKafkaCommit() {
        List<String> calls = new ArrayList<>();
        RuntimeException failure = new IllegalStateException("database unavailable");
        FeatureStateRecordHandler handler = handler(calls, null, failure);

        IllegalStateException error = assertThrows(
            IllegalStateException.class,
            () -> handler.handle(validRecord())
        );

        assertEquals(failure, error);
        assertEquals(List.of("valkey_put", "db_commit"), calls);
    }

    @Test
    void kafkaCommitFailureHappensAfterBothStoresComplete() {
        List<String> calls = new ArrayList<>();
        RuntimeException failure = new IllegalStateException("kafka unavailable");
        RecordingSnapshotStore store = new RecordingSnapshotStore(calls, null);
        RecordingSnapshotSink sink = new RecordingSnapshotSink(calls, null);
        FeatureStateRecordHandler handler = new FeatureStateRecordHandler(
            store,
            sink,
            ignored -> {
                calls.add("kafka_commit");
                throw failure;
            },
            "online_feature_snapshot"
        );

        IllegalStateException error = assertThrows(
            IllegalStateException.class,
            () -> handler.handle(validRecord())
        );

        assertEquals(failure, error);
        assertEquals(List.of("valkey_put", "db_commit", "kafka_commit"), calls);
    }

    private static FeatureStateRecordHandler handler(
        List<String> calls,
        RuntimeException storeFailure,
        RuntimeException sinkFailure
    ) {
        return new FeatureStateRecordHandler(
            new RecordingSnapshotStore(calls, storeFailure),
            new RecordingSnapshotSink(calls, sinkFailure),
            ignored -> calls.add("kafka_commit"),
            "online_feature_snapshot"
        );
    }

    private static ConsumerRecord<String, String> validRecord() {
        ObjectNode snapshot = MAPPER.createObjectNode();
        snapshot.put("serving_snapshot_id", "state:secom-0000001:1000:3");
        snapshot.put("sample_id", "secom-0000001");
        snapshot.put("source_event_count", 3);
        snapshot.put("snapshot_version", 3);
        snapshot.put("feature_hash", FEATURE_HASH);
        snapshot.put("snapshot_time", 1.0);
        snapshot.put("window_start", 0.5);
        snapshot.put("window_end", 1.0);
        snapshot.put("snapshot_status", "complete");
        snapshot.put("feature_count", 590);
        snapshot.put("missing_count", 0);
        snapshot.put("is_complete", true);

        ObjectNode features = snapshot.putObject("features");
        for (int index = 0; index < 590; index++) {
            features.put(String.format("f%03d", index), (double) index);
        }

        return new ConsumerRecord<>(
            "secom-feature-state-updates",
            0,
            10L,
            "secom-0000001",
            snapshot.toString()
        );
    }

    private static final class RecordingSnapshotStore implements SnapshotStore {
        private final List<String> calls;
        private final RuntimeException failure;
        private String lastKey;

        private RecordingSnapshotStore(List<String> calls, RuntimeException failure) {
            this.calls = calls;
            this.failure = failure;
        }

        @Override
        public void verifyConnection() {
        }

        @Override
        public SnapshotWriteResult put(String key, String value) {
            calls.add("valkey_put");
            lastKey = key;
            if (failure != null) {
                throw failure;
            }
            return new SnapshotWriteResult(123.456);
        }

        @Override
        public void close() {
        }
    }

    private static final class RecordingSnapshotSink implements ServingSnapshotSink {
        private final List<String> calls;
        private final RuntimeException failure;
        private ServingSnapshotRow lastRow;
        private double lastAvailableAt;

        private RecordingSnapshotSink(List<String> calls, RuntimeException failure) {
            this.calls = calls;
            this.failure = failure;
        }

        @Override
        public void verifyConnection() {
        }

        @Override
        public int persist(ServingSnapshotRow row, double availableAt) {
            calls.add("db_commit");
            lastRow = row;
            lastAvailableAt = availableAt;
            if (failure != null) {
                throw failure;
            }
            return 1;
        }

        @Override
        public void close() {
        }
    }
}
