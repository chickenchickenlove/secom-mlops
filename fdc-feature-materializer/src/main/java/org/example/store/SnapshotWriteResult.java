package org.example.store;

public record SnapshotWriteResult(double availableAt) {
    
    public SnapshotWriteResult {
        if (!Double.isFinite(availableAt) || availableAt < 0.0) {
            throw new IllegalArgumentException("availableAt must be finite and >= 0");
        }
    }
}
