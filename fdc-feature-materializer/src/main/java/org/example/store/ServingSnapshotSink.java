package org.example.store;

import org.example.snapshot.ServingSnapshotRow;

public interface ServingSnapshotSink extends AutoCloseable {
    
    void verifyConnection();

    int persist(ServingSnapshotRow row, double availableAt);

    @Override
    void close();
}
