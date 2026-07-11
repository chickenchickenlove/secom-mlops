package org.example;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

final class PredictionLogRepository {
    private static final String INSERT_SQL = """
          INSERT INTO prediction_logs (
              prediction_id,
              request_id,
              sample_id,
              serving_snapshot_id,
              snapshot_version,
              model_run_id,
              model_name,
              model_version,
              model_alias,
              model_uri,
              runtime_slot,
              predicted_at,
              fail_probability,
              predicted_value,
              predicted_label,
              threshold,
              missing_count,
              latency_ms
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (prediction_id) DO NOTHING
          """;

    int insert(Connection connection, PredictionLogRow row) throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement(INSERT_SQL)) {
            statement.setString(1, row.predictionId());
            statement.setString(2, row.requestId());
            statement.setString(3, row.sampleId());
            statement.setString(4, row.servingSnapshotId());
            statement.setLong(5, row.snapshotVersion());
            statement.setString(6, row.modelRunId());
            statement.setString(7, row.modelName());
            statement.setString(8, row.modelVersion());
            statement.setString(9, row.modelAlias());
            statement.setString(10, row.modelUri());
            statement.setString(11, row.runtimeSlot());
            statement.setDouble(12, row.predictedAt());
            statement.setDouble(13, row.failProbability());
            statement.setInt(14, row.predictedValue());
            statement.setString(15, row.predictedLabel());
            statement.setDouble(16, row.threshold());
            statement.setInt(17, row.missingCount());
            statement.setDouble(18, row.latencyMs());
            return statement.executeUpdate();
        }
    }
}
