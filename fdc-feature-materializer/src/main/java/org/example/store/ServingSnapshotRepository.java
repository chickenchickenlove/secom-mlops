package org.example.store;

import org.example.snapshot.ServingSnapshotRow;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

final class ServingSnapshotRepository {
    private static final String INSERT_SQL = """
          INSERT INTO serving_feature_snapshots (
              serving_snapshot_id,
              sample_id,
              snapshot_version,
              feature_hash,
              snapshot_time,
              window_start,
              window_end,
              snapshot_status,
              feature_count,
              missing_count,
              is_complete,
              features_json,
              simulation_run_id,
              drift_segment,
              available_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::jsonb, ?, ?, ?)
          ON CONFLICT (serving_snapshot_id) DO NOTHING
          """;

    int insert(Connection connection, ServingSnapshotRow row, double availableAt) throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement(INSERT_SQL)) {
            statement.setString(1, row.servingSnapshotId());
            statement.setString(2, row.sampleId());
            statement.setLong(3, row.snapshotVersion());
            statement.setString(4, row.featureHash());
            statement.setDouble(5, row.snapshotTime());
            statement.setDouble(6, row.windowStart());
            statement.setDouble(7, row.windowEnd());
            statement.setString(8, row.snapshotStatus());
            statement.setInt(9, row.featureCount());
            statement.setInt(10, row.missingCount());
            statement.setBoolean(11, row.isComplete());
            statement.setString(12, row.featuresJson());
            statement.setString(13, row.simulationRunId());
            statement.setString(14, row.driftSegment());
            statement.setDouble(15, availableAt);
            return statement.executeUpdate();
        }
    }
}
