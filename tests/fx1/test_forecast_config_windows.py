"""SYNTHETIC offline coverage of serialized config-to-pipeline window contracts."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from fx1.forecast.config import load_harness_config
from fx1.forecast.features import OhlcvFeaturePipeline
from fx1.forecast.runner import resolve_pipeline


def _config_path(tmp_path: Path, suffix: str, features: dict[str, Any]) -> Path:
    payload = {"data": {"provider": "synthetic"}, "features": features}
    path = tmp_path / f"harness.{suffix}"
    text = json.dumps(payload) if suffix == "json" else yaml.safe_dump(payload)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("suffix", ["yaml", "json"])
@pytest.mark.parametrize("field", ["lookbacks", "vol_window"])
@pytest.mark.parametrize("value", [True, False, 2.0, 2.5, "2", None])
def test_serialized_windows_reject_coercion(
    tmp_path: Path, suffix: str, field: str, value: Any
) -> None:
    features = {field: [value] if field == "lookbacks" else value}
    path = _config_path(tmp_path, suffix, features)
    # Exercise the same loader and builder used by inference, without injecting
    # a pipeline or bypassing validation through model_construct/model_copy.
    with pytest.raises(ValidationError) as error:
        resolve_pipeline(load_harness_config(path))
    location = ("features", field, 0) if field == "lookbacks" else ("features", field)
    assert any(item["loc"] == location for item in error.value.errors())


@pytest.mark.parametrize("suffix", ["yaml", "json"])
@pytest.mark.parametrize(
    "features, lookbacks, vol_window",
    [({}, (1, 5, 20), 20), ({"lookbacks": [5, 1, 3], "vol_window": 2}, (1, 3, 5), 2)],
)
def test_serialized_integer_windows_and_defaults_reach_pipeline(
    tmp_path: Path,
    suffix: str,
    features: dict[str, Any],
    lookbacks: tuple[int, ...],
    vol_window: int,
) -> None:
    path = _config_path(tmp_path, suffix, features)
    pipeline = resolve_pipeline(load_harness_config(path))
    assert isinstance(pipeline, OhlcvFeaturePipeline)
    assert pipeline.lookbacks == lookbacks
    assert pipeline.vol_window == vol_window
    assert all(type(value) is int for value in pipeline.lookbacks)
    assert type(pipeline.vol_window) is int


@pytest.mark.parametrize("suffix", ["yaml", "json"])
@pytest.mark.parametrize(
    "features",
    [
        {"lookbacks": []},
        {"lookbacks": [0]},
        {"lookbacks": [-1]},
        {"lookbacks": [1, 1]},
        {"vol_window": 1},
    ],
)
def test_serialized_window_bounds_and_uniqueness_remain_enforced(
    tmp_path: Path, suffix: str, features: dict[str, Any]
) -> None:
    path = _config_path(tmp_path, suffix, features)
    with pytest.raises(ValidationError):
        resolve_pipeline(load_harness_config(path))
