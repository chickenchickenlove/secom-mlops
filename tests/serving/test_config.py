import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from secom_mlops.serving.api.config import ServingApiConfig
from secom_mlops.serving.runtime.app import (
    seed_runtime_deployment_state_if_missing,
)
from secom_mlops.serving.runtime.config import ModelRuntimeConfig
from secom_mlops.serving.runtime.model import ModelRuntime


class ServingApiConfigTest(unittest.TestCase):
    def test_requires_predictor_slot(self) -> None:
        environ = {
            "MODEL_RUNTIME_URL": "http://model-server-release:28091",
            "SHADOW_MODEL_RUNTIME_URL": "http://model-server-shadow:28093",
        }

        with patch.dict(os.environ, environ, clear=True):
            with self.assertRaises(ValidationError):
                ServingApiConfig()

    def test_requires_runtime_urls(self) -> None:
        with patch.dict(os.environ, {"PREDICTOR_SLOT": "release"}, clear=True):
            with self.assertRaises(ValidationError):
                ServingApiConfig()

    def test_allows_shadow_runtime_to_be_omitted(self) -> None:
        environ = {
            "PREDICTOR_SLOT": "canary",
            "MODEL_RUNTIME_URL": "http://model-server-canary:28092",
        }

        with patch.dict(os.environ, environ, clear=True):
            config = ServingApiConfig()

        self.assertIsNone(config.shadow_model_runtime_url)

    def test_uses_optional_defaults(self) -> None:
        environ = {
            "PREDICTOR_SLOT": "release",
            "MODEL_RUNTIME_URL": "http://model-server-release:28091",
            "SHADOW_MODEL_RUNTIME_URL": "http://model-server-shadow:28093",
        }

        with patch.dict(os.environ, environ, clear=True):
            config = ServingApiConfig()

        self.assertEqual("127.0.0.1:9092", config.kafka_bootstrap_servers)
        self.assertEqual("release", config.predictor_slot)
        self.assertEqual(
            "http://model-server-release:28091",
            config.model_runtime_url,
        )
        self.assertEqual("/invocations", config.model_runtime_path)
        self.assertEqual(
            "http://model-server-shadow:28093",
            config.shadow_model_runtime_url,
        )
        self.assertEqual("/invocations", config.shadow_model_runtime_path)
        self.assertEqual(0.02, config.model_batch_max_wait_seconds)
        self.assertEqual(
            0.1,
            config.prediction_event_batch_max_wait_seconds,
        )

    def test_reads_and_converts_environment_values(self) -> None:
        environ = {
            "KAFKA_BOOTSTRAP_SERVERS": "kafka:29092",
            "VALKEY_URL": "valkey://valkey:6379/0",
            "VALKEY_PORT": "6380",
            "PREDICTOR_SLOT": "canary",
            "MODEL_RUNTIME_URL": "http://model-server-canary:28092",
            "MODEL_RUNTIME_PATH": "/invocations",
            "SHADOW_MODEL_RUNTIME_URL": "http://model-server-shadow:28093",
            "SHADOW_MODEL_RUNTIME_PATH": "/invocations",
            "MODEL_BATCH_MAX_SIZE": "32",
            "MODEL_BATCH_MAX_WAIT_MS": "25",
            "PREDICTION_EVENT_BATCH_MAX_WAIT_MS": "125",
        }

        with patch.dict(os.environ, environ, clear=True):
            config = ServingApiConfig()

        self.assertEqual("kafka:29092", config.kafka_bootstrap_servers)
        self.assertEqual("valkey://valkey:6379/0", config.valkey_url)
        self.assertEqual(6380, config.valkey_port)
        self.assertEqual("canary", config.predictor_slot)
        self.assertEqual(
            "http://model-server-canary:28092",
            config.model_runtime_url,
        )
        self.assertEqual(
            "http://model-server-shadow:28093",
            config.shadow_model_runtime_url,
        )
        self.assertEqual(32, config.model_batch_max_size)
        self.assertEqual(0.025, config.model_batch_max_wait_seconds)
        self.assertEqual(
            0.125,
            config.prediction_event_batch_max_wait_seconds,
        )

    def test_rejects_invalid_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PREDICTOR_SLOT": "release",
                "MODEL_RUNTIME_URL": "http://model-server-release:28091",
                "SHADOW_MODEL_RUNTIME_URL": "http://model-server-shadow:28093",
                "MODEL_BATCH_MAX_SIZE": "0",
            },
            clear=True,
        ):
            with self.assertRaises(ValidationError):
                ServingApiConfig()

        with patch.dict(
            os.environ,
            {
                "PREDICTOR_SLOT": "release",
                "MODEL_RUNTIME_URL": "http://model-server-release:28091",
                "SHADOW_MODEL_RUNTIME_URL": "http://model-server-shadow:28093",
                "SHADOW_MODEL_RUNTIME_PATH": "shadow/invocations",
            },
            clear=True,
        ):
            with self.assertRaises(ValidationError):
                ServingApiConfig()


class ModelRuntimeConfigTest(unittest.TestCase):
    def test_uses_local_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = ModelRuntimeConfig()

        self.assertEqual(
            "http://localhost:5100",
            config.mlflow_tracking_uri,
        )
        self.assertEqual("secom-fail-detector", config.ml_model_name)
        self.assertEqual("champion", config.ml_model_alias)
        self.assertEqual("release", config.model_runtime_slot)
        self.assertIsNone(config.ml_model_version)
        self.assertIsNone(config.ml_model_uri)

    def test_accepts_existing_ml_url_environment_name(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ML_URL": "http://mlflow:5100",
                "ML_MODEL_URI": "models:/secom-fail-detector@champion",
                "MODEL_RUNTIME_SLOT": "shadow",
            },
            clear=True,
        ):
            config = ModelRuntimeConfig()

        self.assertEqual("http://mlflow:5100", config.mlflow_tracking_uri)
        self.assertEqual(
            "models:/secom-fail-detector@champion",
            config.ml_model_uri,
        )
        self.assertEqual("shadow", config.model_runtime_slot)

    def test_prefers_canonical_mlflow_tracking_uri(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MLFLOW_TRACKING_URI": "http://canonical-mlflow:5100",
                "ML_URL": "http://legacy-mlflow:5100",
            },
            clear=True,
        ):
            config = ModelRuntimeConfig()

        self.assertEqual(
            "http://canonical-mlflow:5100",
            config.mlflow_tracking_uri,
        )

    def test_accepts_explicit_field_name(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = ModelRuntimeConfig(
                mlflow_tracking_uri="http://explicit-mlflow:5100"
            )

        self.assertEqual(
            "http://explicit-mlflow:5100",
            config.mlflow_tracking_uri,
        )

    def test_rejects_unknown_runtime_slot(self) -> None:
        with patch.dict(
            os.environ,
            {"MODEL_RUNTIME_SLOT": "production"},
            clear=True,
        ):
            with self.assertRaises(ValidationError):
                ModelRuntimeConfig()

    def test_runtime_bootstrap_receives_explicit_database_url(self) -> None:
        runtime = ModelRuntime(
            model=object(),
            model_uri="models:/secom-fail-detector/1",
            model_name="secom-fail-detector",
            model_version="1",
            model_alias="champion",
            model_run_id="run-001",
            threshold=0.5,
            runtime_slot="release",
            loaded_at_utc="2026-07-19T00:00:00+00:00",
            reload_request_id=None,
        )
        database_url = "postgresql://mlops:mlops@postgres:5432/monitoring"

        with patch(
            "secom_mlops.serving.runtime.app."
            "insert_runtime_deployment_state_if_missing",
            return_value=None,
        ) as insert_state:
            seed_runtime_deployment_state_if_missing(runtime, database_url)

        self.assertEqual(database_url, insert_state.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
