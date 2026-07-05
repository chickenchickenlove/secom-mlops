# label-archiver

Consumes label events from Kafka and materializes them into PostgreSQL
`actual_labels`.

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
