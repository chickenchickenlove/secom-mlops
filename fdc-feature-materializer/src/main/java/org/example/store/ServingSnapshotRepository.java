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
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::jsonb, ?, ?, ?)
          ON CONFLICT (serving_snapshot_id) DO NOTHING
          """;

    int insert(Connection connection, ServingSnapshotRow row, double availableAt) throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement(INSERT_SQL)) {
            statement.setString(1, row.servingSnapshotId());
            statement.setString(2, row.sampleId());
            statement.setLong(3, row.snapshotVersion());
            statement.setDouble(4, row.snapshotTime());
            statement.setDouble(5, row.windowStart());
            statement.setDouble(6, row.windowEnd());
            statement.setString(7, row.snapshotStatus());
            statement.setInt(8, row.featureCount());
            statement.setInt(9, row.missingCount());
            statement.setBoolean(10, row.isComplete());
            statement.setString(11, row.featuresJson());
            statement.setString(12, row.simulationRunId());
            statement.setString(13, row.driftSegment());
            statement.setDouble(14, availableAt);
            return statement.executeUpdate();
        }
    }
}
