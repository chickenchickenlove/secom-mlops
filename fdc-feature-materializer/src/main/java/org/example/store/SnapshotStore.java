package org.example.store;

public interface SnapshotStore extends AutoCloseable {
    
    void verifyConnection();

    SnapshotWriteResult put(String key, String value);

    @Override
    void close();
}
