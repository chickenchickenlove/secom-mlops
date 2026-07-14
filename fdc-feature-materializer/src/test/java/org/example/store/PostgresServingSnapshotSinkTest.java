package org.example.store;

import org.example.snapshot.ServingSnapshotRow;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Proxy;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PostgresServingSnapshotSinkTest {
    @Test
    void commitsAfterSnapshotInsertSucceeds() {
        List<String> calls = new ArrayList<>();
        Connection connection = connection(calls, false);
        PostgresServingSnapshotSink sink = new PostgresServingSnapshotSink(
            connection,
            (ignored, row, availableAt) -> {
                calls.add("insert:" + availableAt);
                return 1;
            }
        );

        int inserted = sink.persist(row(), 123.456);

        assertEquals(1, inserted);
        assertEquals(List.of("auto_commit_false", "insert:123.456", "commit"), calls);
    }

    @Test
    void rollsBackWhenSnapshotInsertFails() {
        List<String> calls = new ArrayList<>();
        Connection connection = connection(calls, false);
        PostgresServingSnapshotSink sink = new PostgresServingSnapshotSink(
            connection,
            (ignored, row, availableAt) -> {
                calls.add("insert");
                throw new SQLException("write failed");
            }
        );

        assertThrows(IllegalStateException.class, () -> sink.persist(row(), 123.456));
        assertEquals(List.of("auto_commit_false", "insert", "rollback"), calls);
    }

    @Test
    void rollsBackWhenCommitFails() {
        List<String> calls = new ArrayList<>();
        Connection connection = connection(calls, true);
        PostgresServingSnapshotSink sink = new PostgresServingSnapshotSink(
            connection,
            (ignored, row, availableAt) -> {
                calls.add("insert");
                return 1;
            }
        );

        assertThrows(IllegalStateException.class, () -> sink.persist(row(), 123.456));
        assertEquals(List.of("auto_commit_false", "insert", "commit", "rollback"), calls);
    }

    @Test
    void rejectsInvalidAvailableAtBeforeDatabaseWrite() {
        List<String> calls = new ArrayList<>();
        Connection connection = connection(calls, false);
        PostgresServingSnapshotSink sink = new PostgresServingSnapshotSink(
            connection,
            (ignored, row, availableAt) -> {
                calls.add("insert");
                return 1;
            }
        );

        assertThrows(IllegalArgumentException.class, () -> sink.persist(row(), Double.NaN));
        assertEquals(List.of("auto_commit_false"), calls);
    }

    private static Connection connection(List<String> calls, boolean failCommit) {
        return (Connection) Proxy.newProxyInstance(
            Connection.class.getClassLoader(),
            new Class<?>[] {Connection.class},
            (proxy, method, args) -> switch (method.getName()) {
                case "setAutoCommit" -> {
                    calls.add("auto_commit_" + args[0]);
                    yield null;
                }
                case "commit" -> {
                    calls.add("commit");
                    if (failCommit) {
                        throw new SQLException("commit failed");
                    }
                    yield null;
                }
                case "rollback" -> {
                    calls.add("rollback");
                    yield null;
                }
                case "close" -> {
                    calls.add("close");
                    yield null;
                }
                case "isClosed" -> false;
                case "isWrapperFor" -> false;
                case "unwrap" -> throw new SQLException("not a wrapper");
                case "toString" -> "RecordingConnection";
                case "hashCode" -> System.identityHashCode(proxy);
                case "equals" -> proxy == args[0];
                default -> throw new UnsupportedOperationException(method.getName());
            }
        );
    }

    private static ServingSnapshotRow row() {
        return new ServingSnapshotRow(
            "state:secom-0000001:1000:3",
            3L,
            "sha256:v1:" + "0".repeat(64),
            "secom-0000001",
            1.0,
            0.5,
            1.0,
            "complete",
            590,
            0,
            true,
            "{}",
            null,
            null
        );
    }
}
