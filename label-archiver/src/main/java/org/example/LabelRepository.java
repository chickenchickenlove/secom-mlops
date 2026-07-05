package org.example;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

final class LabelRepository {
    private static final String UPSERT_SQL = """
          INSERT INTO actual_labels (
              sample_id,
              actual_value,
              actual_label,
              labeled_at
          )
          VALUES (?, ?, ?, ?)
          ON CONFLICT (sample_id) DO UPDATE SET
              actual_value = EXCLUDED.actual_value,
              actual_label = EXCLUDED.actual_label,
              labeled_at = EXCLUDED.labeled_at
          """;

    void upsert(Connection connection, LabelRow label) throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement(UPSERT_SQL)) {
            statement.setString(1, label.sampleId());
            statement.setInt(2, label.actualValue());
            statement.setString(3, label.actualLabel());
            statement.setDouble(4, label.labeledAt());
            statement.executeUpdate();
        }
    }
}
