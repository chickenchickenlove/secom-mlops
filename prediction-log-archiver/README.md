# prediction-log-archiver

Consumes prediction events from Kafka and archives them into PostgreSQL
`prediction_logs`.

Each prediction event must identify the immutable serving snapshot used for
inference with a non-blank `serving_snapshot_id` and a positive integral
`snapshot_version`. Operational events are emitted by `/predict-by-id`; the
caller-feature `/predict` endpoint is debug-only and does not emit prediction
evidence.

Prediction events and `prediction_logs` do not contain the full feature vector.
The three drift feature consumers join `prediction_logs` to
`serving_feature_snapshots` on the logical triple
`serving_snapshot_id + sample_id + snapshot_version` and read the serving
snapshot's `features_json`. There is intentionally no foreign key, while the
small `missing_count` scalar remains in `prediction_logs`.

This service is configured with environment variables. CLI arguments are not
supported.

## Required Environment Variables

| Name | Example | Description |
| --- | --- | --- |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Kafka bootstrap servers |
| `PREDICTION_EVENTS_TOPIC` | `secom-prediction-events` | Input topic |
| `KAFKA_GROUP_ID` | `secom-prediction-log-archiver` | Consumer group id |
| `KAFKA_CLIENT_ID` | `secom-prediction-log-archiver` | Kafka client id |
| `MONITORING_JDBC_URL` | `jdbc:postgresql://postgres:5432/monitoring` | PostgreSQL JDBC URL |
| `MONITORING_DB_USER` | `mlops` | PostgreSQL user |
| `MONITORING_DB_PASSWORD` | `mlops` | PostgreSQL password |

## Optional Environment Variables

| Name | Default | Description |
| --- | --- | --- |
| `KAFKA_AUTO_OFFSET_RESET` | `earliest` | Kafka consumer offset reset policy |
| `KAFKA_POLL_TIMEOUT_MS` | `1000` | Poll timeout in milliseconds |
| `KAFKA_MAX_POLL_RECORDS` | `100` | Maximum records per poll |
