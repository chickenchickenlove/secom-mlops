package org.example;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

final class FeatureEventRepository {
    private static final String INSERT_SQL = """
          INSERT INTO feature_events (
              event_id,
              sample_id,
              event_time,
              feature_group,
              features_json,
              simulation_run_id,
              drift_segment,
              created_at
          )
          VALUES (?, ?, ?, ?, ?::jsonb, ?, ?, ?)
          ON CONFLICT (event_id) DO NOTHING
          """;

    int insert(Connection connection, FeatureEventRow row) throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement(INSERT_SQL)) {
            statement.setString(1, row.eventId());
            statement.setString(2, row.sampleId());
            statement.setDouble(3, row.eventTime());
            statement.setString(4, row.featureGroup());
            statement.setString(5, row.featuresJson());
            statement.setString(6, row.simulationRunId());
            statement.setString(7, row.driftSegment());
            statement.setDouble(8, row.createdAt());
            return statement.executeUpdate();
        }
    }
}
