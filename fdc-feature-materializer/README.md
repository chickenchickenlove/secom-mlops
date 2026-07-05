# fdc-feature-materializer

Consumes feature state updates from Kafka and writes the latest snapshot for each
sample into Valkey.

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
