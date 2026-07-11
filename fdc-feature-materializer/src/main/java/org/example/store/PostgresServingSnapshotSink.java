package org.example.store;

import org.example.config.MaterializerConfig;
import org.example.snapshot.ServingSnapshotRow;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;

public final class PostgresServingSnapshotSink implements ServingSnapshotSink {
    private static final Logger LOG = LoggerFactory.getLogger(PostgresServingSnapshotSink.class);

    private final Connection connection;
    private final SnapshotWriter snapshotWriter;

    public PostgresServingSnapshotSink(MaterializerConfig config) {
        this(openConnection(config), new ServingSnapshotRepository()::insert);
    }

    PostgresServingSnapshotSink(Connection connection, SnapshotWriter snapshotWriter) {
        this.connection = connection;
        this.snapshotWriter = snapshotWriter;

        try {
            connection.setAutoCommit(false);
        } catch (SQLException error) {
            closeQuietly(connection);
            throw new IllegalStateException("failed to configure serving snapshot database transaction", error);
        }
    }

    @Override
    public void verifyConnection() {
        try (PreparedStatement statement = connection.prepareStatement("SELECT 1")) {
            statement.executeQuery();
            connection.commit();
        } catch (SQLException error) {
            rollbackQuietly();
            throw new IllegalStateException("failed to verify serving snapshot database connection", error);
        }
    }

    @Override
    public int persist(ServingSnapshotRow row, double availableAt) {
        if (!Double.isFinite(availableAt) || availableAt < 0.0) {
            throw new IllegalArgumentException("availableAt must be finite and >= 0");
        }

        try {
            final int inserted = snapshotWriter.insert(connection, row, availableAt);
            connection.commit();
            return inserted;
        } catch (SQLException error) {
            rollbackQuietly();
            throw new IllegalStateException(
                "failed to persist serving snapshot: " + row.servingSnapshotId(),
                error
            );
        }
    }

    @Override
    public void close() {
        try {
            connection.close();
        } catch (SQLException error) {
            throw new IllegalStateException("failed to close serving snapshot database connection", error);
        }
    }

    private static Connection openConnection(MaterializerConfig config) {
        try {
            return DriverManager.getConnection(config.dbUrl(), config.dbUser(), config.dbPassword());
        } catch (SQLException error) {
            throw new IllegalStateException("failed to open serving snapshot database connection", error);
        }
    }

    private void rollbackQuietly() {
        try {
            connection.rollback();
        } catch (SQLException rollbackError) {
            LOG.error("serving_snapshot_rollback_failed", rollbackError);
        }
    }

    private static void closeQuietly(Connection connection) {
        try {
            connection.close();
        } catch (SQLException closeError) {
            LOG.error("serving_snapshot_connection_close_failed", closeError);
        }
    }

    @FunctionalInterface
    interface SnapshotWriter {
        int insert(Connection connection, ServingSnapshotRow row, double availableAt) throws SQLException;
    }
}
