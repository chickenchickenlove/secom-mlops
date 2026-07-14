# fdc-feature-materializer

Consumes feature state updates from Kafka and materializes each validated snapshot
into both online and durable stores. For every record, it writes the latest snapshot
for the sample to Valkey, archives the same snapshot in PostgreSQL
`serving_feature_snapshots` after the Valkey `SET` succeeds, and then commits the
Kafka offset.

If the PostgreSQL write fails, the Kafka offset is not committed. The record can
therefore be replayed instead of leaving an unarchived snapshot as successfully
consumed evidence.

Each snapshot carries a sample-local `snapshot_version` assigned by the assembler.
After the Valkey write succeeds, the materializer records `available_at` in
PostgreSQL as the time that version was confirmed available online. Reprocessing the
same snapshot id does not replace its first recorded `available_at`.

Each snapshot also carries the assembler-generated `feature_hash`. The
materializer validates its versioned SHA-256 format and preserves the same value
in both the Valkey payload and PostgreSQL `serving_feature_snapshots`. It does
not recalculate the hash. Downstream consumers use it to verify that prediction
evidence and durable snapshot history refer to the same Feature vector.

This service is configured with environment variables. CLI arguments are not
supported.

## Required Environment Variables

| Name | Example | Description |
| --- | --- | --- |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Kafka bootstrap servers |
| `FEATURE_STATE_UPDATES_TOPIC` | `secom-feature-state-updates` | Input topic |
| `KAFKA_GROUP_ID` | `secom-feature-state-materializer` | Consumer group id |
| `KAFKA_CLIENT_ID` | `secom-feature-state-materializer` | Kafka client id |
| `VALKEY_HOST` | `valkey` | Valkey host |
| `VALKEY_PORT` | `6379` | Valkey port |
| `MONITORING_JDBC_URL` | `jdbc:postgresql://postgres:5432/monitoring` | PostgreSQL JDBC URL |
| `MONITORING_DB_USER` | `mlops` | PostgreSQL user |
| `MONITORING_DB_PASSWORD` | `mlops` | PostgreSQL password |

## Optional Environment Variables

| Name | Default | Description |
| --- | --- | --- |
| `KAFKA_AUTO_OFFSET_RESET` | `earliest` | Kafka consumer offset reset policy |
| `KAFKA_POLL_TIMEOUT_MS` | `1000` | Poll timeout in milliseconds |
| `KAFKA_MAX_POLL_RECORDS` | `100` | Maximum records per poll |
| `VALKEY_DATABASE` | `0` | Valkey database index |
| `VALKEY_TIMEOUT_MS` | `2000` | Valkey connection timeout in milliseconds |
| `VALKEY_POOL_MAX_TOTAL` | `8` | Maximum total Valkey pool connections |
| `VALKEY_POOL_MAX_IDLE` | `8` | Maximum idle Valkey pool connections |
| `VALKEY_KEY_PREFIX` | `online_feature_snapshot` | Prefix for materialized snapshot keys |
| `VALKEY_VERIFY_WRITE` | `true` | Verify writes by reading back values |
