"""Privacy invariants for the repository's offline pytest gates."""

from __future__ import annotations

import os


def test_mlflow_telemetry_is_disabled_for_test_session() -> None:
    """The offline suite must opt out before any MLflow client can start."""
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"].lower() == "true"

    from mlflow.telemetry.utils import is_telemetry_disabled

    assert is_telemetry_disabled() is True
