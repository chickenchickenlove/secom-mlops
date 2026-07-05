# fdc-feature-assembler

Consumes feature patches from Kafka, merges feature state by `sample_id`, and
emits canonical feature state updates.

This service is configured with environment variables. CLI arguments are not
supported.

## Required Environment Variables

| Name | Example | Description |
| --- | --- | --- |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Kafka bootstrap servers |
| `FEATURE_ASSEMBLER_APPLICATION_ID` | `secom-feature-assembler` | Kafka Streams application id |
| `FEATURE_PATCHES_TOPIC` | `secom-feature-patches` | Input topic |
| `FEATURE_STATE_UPDATES_TOPIC` | `secom-feature-state-updates` | Output topic |
| `FEATURE_ASSEMBLER_STATE_DIR` | `/var/lib/secom-feature-assembler` | Kafka Streams state directory |

## Optional Environment Variables

| Name | Default | Description |
| --- | --- | --- |
| `KAFKA_AUTO_OFFSET_RESET` | `latest` | Kafka Streams input offset reset policy |
