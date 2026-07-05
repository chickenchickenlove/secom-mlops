package org.example;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

final class ServingSnapshotRepository {
    private static final String UPSERT_SQL = """
          INSERT INTO serving_feature_snapshots (
              serving_snapshot_id,
              sample_id,
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
              created_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?::jsonb, ?, ?, ?)
          ON CONFLICT (serving_snapshot_id) DO UPDATE SET
              sample_id = EXCLUDED.sample_id,
              snapshot_time = EXCLUDED.snapshot_time,
              window_start = EXCLUDED.window_start,
              window_end = EXCLUDED.window_end,
              snapshot_status = EXCLUDED.snapshot_status,
              feature_count = EXCLUDED.feature_count,
              missing_count = EXCLUDED.missing_count,
              is_complete = EXCLUDED.is_complete,
              features_json = EXCLUDED.features_json,
              simulation_run_id = EXCLUDED.simulation_run_id,
              drift_segment = EXCLUDED.drift_segment,
              created_at = EXCLUDED.created_at
          """;

    int upsert(Connection connection, ServingSnapshotRow row) throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement(UPSERT_SQL)) {
            statement.setString(1, row.servingSnapshotId());
            statement.setString(2, row.sampleId());
            statement.setDouble(3, row.snapshotTime());
            statement.setDouble(4, row.windowStart());
            statement.setDouble(5, row.windowEnd());
            statement.setString(6, row.snapshotStatus());
            statement.setInt(7, row.featureCount());
            statement.setInt(8, row.missingCount());
            statement.setBoolean(9, row.isComplete());
            statement.setString(10, row.featuresJson());
            statement.setString(11, row.simulationRunId());
            statement.setString(12, row.driftSegment());
            statement.setDouble(13, row.createdAt());
            return statement.executeUpdate();
        }
    }
}
