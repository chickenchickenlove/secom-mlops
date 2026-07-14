package org.example;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

final class LabelRepository {
    private static final String INSERT_SQL = """
          INSERT INTO label_events (
              label_event_id,
              sample_id,
              label_revision,
              measured_at,
              actual_value,
              actual_label
          )
          VALUES (?, ?, ?, ?, ?, ?)
          ON CONFLICT (label_event_id) DO NOTHING
          """;

    private static final String SELECT_SEMANTIC_FIELDS_SQL = """
          SELECT
              sample_id,
              label_revision,
              actual_value,
              actual_label
          FROM label_events
          WHERE label_event_id = ?
          """;

    boolean insert(Connection connection, LabelRow label) throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement(INSERT_SQL)) {
            statement.setString(1, label.labelEventId());
            statement.setString(2, label.sampleId());
            statement.setLong(3, label.labelRevision());
            statement.setDouble(4, label.measuredAt());
            statement.setInt(5, label.actualValue());
            statement.setString(6, label.actualLabel());

            int insertedRows = statement.executeUpdate();
            if (insertedRows == 1) {
                return true;
            }
            if (insertedRows != 0) {
                throw new SQLException("unexpected inserted row count: " + insertedRows);
            }
        }

        verifyReplayPayload(connection, label);
        return false;
    }

    private static void verifyReplayPayload(Connection connection, LabelRow label)
        throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement(SELECT_SEMANTIC_FIELDS_SQL)) {
            statement.setString(1, label.labelEventId());

            try (ResultSet result = statement.executeQuery()) {
                if (!result.next()) {
                    throw new SQLException(
                        "label replay row not found: label_event_id=" + label.labelEventId()
                    );
                }

                boolean samePayload =
                    label.sampleId().equals(result.getString("sample_id"))
                        && label.labelRevision() == result.getLong("label_revision")
                        && label.actualValue() == result.getInt("actual_value")
                        && label.actualLabel().equals(result.getString("actual_label"));

                if (!samePayload) {
                    throw new SQLException(
                        "label_event_id payload conflict: label_event_id="
                            + label.labelEventId()
                    );
                }
            }
        }
    }
}
