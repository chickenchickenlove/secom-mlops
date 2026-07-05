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
              features_json,
              missing_count,
              latency_ms
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::jsonb, ?, ?)
          ON CONFLICT (prediction_id) DO NOTHING
          """;

    int insert(Connection connection, PredictionLogRow row) throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement(INSERT_SQL)) {
            statement.setString(1, row.predictionId());
            statement.setString(2, row.requestId());
            statement.setString(3, row.sampleId());
            statement.setString(4, row.modelRunId());
            statement.setString(5, row.modelName());
            statement.setString(6, row.modelVersion());
            statement.setString(7, row.modelAlias());
            statement.setString(8, row.modelUri());
            statement.setString(9, row.runtimeSlot());
            statement.setDouble(10, row.predictedAt());
            statement.setDouble(11, row.failProbability());
            statement.setInt(12, row.predictedValue());
            statement.setString(13, row.predictedLabel());
            statement.setDouble(14, row.threshold());
            statement.setString(15, row.featuresJson());
            statement.setInt(16, row.missingCount());
            statement.setDouble(17, row.latencyMs());
            return statement.executeUpdate();
        }
    }
}
