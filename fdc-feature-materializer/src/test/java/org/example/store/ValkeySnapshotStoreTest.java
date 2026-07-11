package org.example.store;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ValkeySnapshotStoreTest {
    @Test
    void capturesAvailableAtImmediatelyAfterSetWithoutVerification() {
        List<String> calls = new ArrayList<>();

        SnapshotWriteResult result = ValkeySnapshotStore.writeAndVerify(
            "snapshot:key",
            "value",
            false,
            (key, value) -> {
                calls.add("set");
                return "OK";
            },
            key -> {
                calls.add("get");
                return "value";
            },
            () -> {
                calls.add("clock");
                return 123.456;
            }
        );

        assertEquals(123.456, result.availableAt());
        assertEquals(List.of("set", "clock"), calls);
    }

    @Test
    void capturesAvailableAtBeforeReadAfterWriteVerification() {
        List<String> calls = new ArrayList<>();

        SnapshotWriteResult result = ValkeySnapshotStore.writeAndVerify(
            "snapshot:key",
            "value",
            true,
            (key, value) -> {
                calls.add("set");
                return "OK";
            },
            key -> {
                calls.add("get");
                return "value";
            },
            () -> {
                calls.add("clock");
                return 123.456;
            }
        );

        assertEquals(123.456, result.availableAt());
        assertEquals(List.of("set", "clock", "get"), calls);
    }

    @Test
    void setFailureDoesNotCaptureTimeOrReadBack() {
        List<String> calls = new ArrayList<>();

        assertThrows(
            IllegalStateException.class,
            () -> ValkeySnapshotStore.writeAndVerify(
                "snapshot:key",
                "value",
                true,
                (key, value) -> {
                    calls.add("set");
                    return "ERROR";
                },
                key -> {
                    calls.add("get");
                    return "value";
                },
                () -> {
                    calls.add("clock");
                    return 123.456;
                }
            )
        );

        assertEquals(List.of("set"), calls);
    }

    @Test
    void readBackMismatchFailsAfterTimeWasCaptured() {
        List<String> calls = new ArrayList<>();

        assertThrows(
            IllegalStateException.class,
            () -> ValkeySnapshotStore.writeAndVerify(
                "snapshot:key",
                "value",
                true,
                (key, value) -> {
                    calls.add("set");
                    return "OK";
                },
                key -> {
                    calls.add("get");
                    return "different";
                },
                () -> {
                    calls.add("clock");
                    return 123.456;
                }
            )
        );

        assertEquals(List.of("set", "clock", "get"), calls);
    }
}
