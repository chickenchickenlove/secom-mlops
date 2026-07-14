# fdc-feature-assembler

Consumes feature patches from Kafka, merges feature state by `sample_id`, and
emits canonical feature state updates.

Before emitting a state update, the assembler builds a canonical Feature vector
containing exactly `f000` through `f589` and computes its `feature_hash`.

The hash uses the versioned format `sha256:v1:<64 lowercase hex>`. It is
deterministic regardless of JSON key insertion order, distinguishes `null` from
numeric zero, normalizes `-0.0` to `0.0`, and rejects non-finite numeric values.
Downstream services propagate this hash instead of recalculating it.

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
