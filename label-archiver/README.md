# label-archiver

Consumes label events from Kafka and appends them to PostgreSQL
`label_events`.

Each label event contains `label_event_id`, `sample_id`, `label_revision`,
`measured_at`, `actual_value`, and `actual_label`. PostgreSQL assigns
`available_at` when the label is first inserted.

Labels are stored as an append-only correction history. A higher
`label_revision` represents a newer authoritative label for the same sample.
Replaying the same `label_event_id` with the same payload is idempotent, while
a conflicting payload is treated as an error.

The Kafka offset is committed only after the PostgreSQL transaction succeeds.

This service is configured with environment variables. CLI arguments are not
supported.

## Required Environment Variables

| Name | Example | Description |
| --- | --- | --- |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Kafka bootstrap servers |
| `LABEL_EVENTS_TOPIC` | `secom-label-events` | Input topic |
| `KAFKA_GROUP_ID` | `secom-label-archiver` | Consumer group id |
| `KAFKA_CLIENT_ID` | `secom-label-archiver` | Kafka client id |
| `MONITORING_JDBC_URL` | `jdbc:postgresql://postgres:5432/monitoring` | PostgreSQL JDBC URL |
| `MONITORING_DB_USER` | `mlops` | PostgreSQL user |
| `MONITORING_DB_PASSWORD` | `mlops` | PostgreSQL password |

## Optional Environment Variables

| Name | Default | Description |
| --- | --- | --- |
| `KAFKA_AUTO_OFFSET_RESET` | `earliest` | Kafka consumer offset reset policy |
| `KAFKA_POLL_TIMEOUT_MS` | `1000` | Poll timeout in milliseconds |
| `KAFKA_MAX_POLL_RECORDS` | `100` | Maximum records per poll |
