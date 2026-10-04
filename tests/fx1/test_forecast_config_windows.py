"""SYNTHETIC offline coverage of serialized config-to-pipeline window contracts."""

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from fx1.forecast.config import FeatureSection, Fx1HarnessConfig, load_harness_config
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


@pytest.mark.parametrize("scalar", [int, np.int32, np.int64, np.uint64])
def test_programmatic_integral_windows_match_direct_pipeline(scalar: Any) -> None:
    features = {"lookbacks": [scalar(5), scalar(1), scalar(3)], "vol_window": scalar(2)}
    section = FeatureSection.model_validate(features)
    config = Fx1HarnessConfig.model_validate(
        {"data": {"provider": "synthetic"}, "features": features}
    )
    direct = OhlcvFeaturePipeline(**features)
    resolved = resolve_pipeline(config)
    assert isinstance(resolved, OhlcvFeaturePipeline)
    assert section.lookbacks == [5, 1, 3]
    assert all(type(value) is int for value in section.lookbacks)
    assert type(section.vol_window) is int
    assert resolved.lookbacks == direct.lookbacks == (1, 3, 5)
    assert resolved.vol_window == direct.vol_window == 2
    assert resolved.feature_columns() == direct.feature_columns()
    assert json.loads(section.model_dump_json())["lookbacks"] == [5, 1, 3]


@pytest.mark.parametrize("field", ["lookbacks", "vol_window"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        2.0,
        2.5,
        np.float64(2),
        "2",
        None,
        Decimal(2),
        Fraction(2, 1),
        np.array(2),
    ],
)
def test_programmatic_windows_reject_nonintegral_scalars(field: str, value: Any) -> None:
    features = {field: [value] if field == "lookbacks" else value}
    with pytest.raises(ValidationError) as error:
        Fx1HarnessConfig.model_validate({"data": {"provider": "synthetic"}, "features": features})
    location = ("features", field, 0) if field == "lookbacks" else ("features", field)
    assert any(item["loc"] == location for item in error.value.errors())
    with pytest.raises(ValueError):
        OhlcvFeaturePipeline(**{"lookbacks": [1], "vol_window": 2, **features})


@pytest.mark.parametrize(
    "features",
    [
        {"lookbacks": [np.int64(0)]},
        {"lookbacks": [np.int64(-1)]},
        {"lookbacks": [1, np.int64(1)]},
        {"vol_window": np.int64(1)},
        {"vol_window": np.int64(-1)},
    ],
)
def test_numpy_window_bounds_and_uniqueness_match_pipeline(features: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FeatureSection.model_validate(features)
    with pytest.raises(ValueError):
        OhlcvFeaturePipeline(**{"lookbacks": [1], "vol_window": 2, **features})


def test_integral_normalization_preserves_other_config_behavior(tmp_path: Path) -> None:
    config = Fx1HarnessConfig.model_validate(
        {
            "data": {"provider": "parquet", "root": str(tmp_path), "synthetic_seed": "7"},
            "features": {"horizon_bars": "2"},
            "inference": {
                "output_parquet": str(tmp_path / "forecasts.parquet"),
                "output_meta": str(tmp_path / "forecasts.meta.json"),
            },
        }
    )
    assert config.data.root == tmp_path
    assert config.data.synthetic_seed == 7
    assert config.features.horizon_bars == 2
    assert config.features.lookbacks == [1, 5, 20]
    assert config.features.vol_window == 20
    assert config.inference.output_parquet == tmp_path / "forecasts.parquet"
    assert config.inference.output_meta == tmp_path / "forecasts.meta.json"
    with pytest.raises(ValidationError):
        FeatureSection.model_validate({"unknown_window": 5})
