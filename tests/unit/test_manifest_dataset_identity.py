"""Dataset identity must be impossible to omit silently from a manifest.

Every training dispatch emits ``model_artifact.v1`` carrying a verified
identity block (payload hash, feature set, label/horizon, full dataset
identity, config hash, git revision + worktree hash). ``save_training_artifact``
takes identity as a required keyword argument, so omission is a ``TypeError``;
``verify_artifact_manifest`` fails closed when the block is absent or partial.

All rows here are SYNTHETIC (correctness fixtures); they are not market
evidence and are labeled as such through ``config.data.source == "synthetic"``.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import ValidationConfig
from quant_fund.models.base import save_joblib_artifact
from quant_fund.pipeline.artifact_manifest import (
    ArtifactManifestError,
    DatasetIdentityError,
    bind_artifact_identity,
    build_dataset_identity,
    dataset_identity_from_artifact,
    identity_for_training,
    manifest_path,
    save_training_artifact,
    verify_artifact_manifest,
)
from quant_fund.pipeline.train import train_calibration_auto, train_family
from quant_fund.utils.hashing import canonical_frame_fingerprint

LABELS = (
    "future_return_5",
    "future_log_return_5",
    "future_realized_var_5",
    "future_tail_event_5",
    "future_idio_return_1",
)
D0 = datetime(2020, 1, 1, tzinfo=UTC)


def _frame(n_dates: int = 64, n_secs: int = 6) -> pl.DataFrame:
    rng = np.random.default_rng(20261007)
    rows = []
    for day in range(n_dates):
        stamp = D0 + timedelta(days=day)
        mkt_vol = 0.02 + abs(float(rng.normal(0, 0.003)))
        for sec in range(n_secs):
            rows.append(
                {
                    "event_time": stamp,
                    "security_id": f"S{sec:02d}",
                    "ret_1": float(rng.normal(0.0004, 0.01)),
                    "vol_20": 0.02 + abs(float(rng.normal(0, 0.003))),
                    "vol_ewma": 0.02 + abs(float(rng.normal(0, 0.002))),
                    "vol_parkinson": 0.02 + abs(float(rng.normal(0, 0.003))),
                    "vol_of_vol": 0.005 + abs(float(rng.normal(0, 0.001))),
                    "mom_20": float(rng.normal(0, 0.01)),
                    "cs_pct_mom_20": float(rng.normal(0, 0.01)),
                    "reversal_1": float(rng.normal(0, 0.01)),
                    "amihud": 1e-6 + abs(float(rng.normal(0, 1e-7))),
                    "future_idio_return_1": float(rng.normal(0, 0.01)),
                    "future_log_return_5": float(rng.normal(0, 0.02)),
                    "future_return_5": float(rng.normal(0, 0.02)),
                    "future_realized_var_5": 0.0004 + abs(float(rng.normal(0, 0.0002))),
                    "future_tail_event_5": bool(rng.random() < 0.15),
                    "mkt_ret_1": float(rng.normal(0, 0.01)),
                    "mkt_vol_20": mkt_vol,
                    "cs_dispersion": 0.01 + abs(float(rng.normal(0, 0.002))),
                    "breadth": 0.5 + float(rng.normal(0, 0.05)),
                    "high": 101.0 + float(rng.normal(0, 1.0)),
                    "low": 99.0 + float(rng.normal(0, 1.0)),
                }
            )
    return pl.DataFrame(rows)


def _materialize_sources(root: Path, frame: pl.DataFrame) -> None:
    (root / "gold").mkdir(parents=True, exist_ok=True)
    (root / "silver").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(root / "gold" / "features.parquet")
    frame.select(["event_time", "security_id", *LABELS]).write_parquet(
        root / "gold" / "labels.parquet"
    )
    frame.select(["event_time", "security_id"]).unique().write_parquet(
        root / "silver" / "universe.parquet"
    )


def _cfg(tmp_path: Path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.validation = ValidationConfig(
        scheme="expanding", train_bars=16, val_bars=4, test_bars=4, seed=42
    )
    return cfg


def _identity(frame: pl.DataFrame, cfg, label: str = "future_return_5"):
    return identity_for_training(
        frame,
        config=cfg,
        label=label,
        features=["mom_20", "vol_20", "reversal_1"],
        label_horizon_bars=5,
    )


def _patch_train(monkeypatch: pytest.MonkeyPatch, frame: pl.DataFrame) -> None:
    for module in (
        "quant_fund.pipeline.train",
        "quant_fund.pipeline.train.families",
        "quant_fund.pipeline.train.ranking",
        "quant_fund.pipeline.train.distribution",
        "quant_fund.pipeline.train.volatility",
    ):
        monkeypatch.setattr(f"{module}.panel", lambda *a, **k: frame, raising=False)
        monkeypatch.setattr(f"{module}.configure_tracking", lambda: None, raising=False)
        monkeypatch.setattr(
            f"{module}.log_run", lambda **kwargs: "run_test_0001", raising=False
        )
        # Never touch shared MLflow sqlite state from correctness tests.
        monkeypatch.setattr(
            f"{module}.attach_artifact_identity", lambda *a, **k: None, raising=False
        )


def test_save_training_artifact_requires_identity_keyword(tmp_path: Path) -> None:
    frame = _frame()
    cfg = _cfg(tmp_path)
    _materialize_sources(tmp_path, frame)
    identity = _identity(frame, cfg)
    with pytest.raises(TypeError):
        save_training_artifact({"model": 1}, tmp_path / "a.joblib", identity)  # type: ignore[misc]
    assert not (tmp_path / "a.joblib").exists()
    assert not (tmp_path / "a.manifest.json").exists()


def test_manifest_carries_complete_dataset_identity(tmp_path: Path) -> None:
    frame = _frame()
    cfg = _cfg(tmp_path)
    _materialize_sources(tmp_path, frame)
    identity = _identity(frame, cfg)
    artifact = tmp_path / "model.joblib"
    save_training_artifact({"model": 1}, artifact, identity=identity)

    manifest = json.loads(manifest_path(artifact).read_text(encoding="utf-8"))
    assert manifest["schema"] == "model_artifact.v1"
    block = manifest["identity"]
    assert block["identity_schema"] == "artifact_identity.v1"
    dataset = block["dataset"]
    assert dataset["materialized_panel_sha256"] == canonical_frame_fingerprint(frame)
    assert dataset["row_count"] == frame.height
    assert dataset["column_count"] == frame.width
    assert dataset["label"] == "future_return_5"
    assert dataset["label_horizon_bars"] == 5
    assert dataset["data_source"] == "synthetic"
    assert dataset["time_start"] and dataset["time_end"]
    assert len(dataset["source_manifest_sha256"]) == 64
    assert block["features"]["features"] == ["mom_20", "vol_20", "reversal_1"]
    assert block["label"] == {"name": "future_return_5", "horizon_bars": 5}
    assert block["horizon"]["bars"] == [int(bar) for bar in cfg.horizons.bars]
    assert block["horizon"]["names"] == [str(name) for name in cfg.horizons.names]
    assert len(block["config_sha256"]) == 64
    assert block["git_revision"]
    assert block["git_worktree_sha256"]

    verified = verify_artifact_manifest(artifact)
    assert verified.dataset.materialized_panel_sha256 == dataset["materialized_panel_sha256"]
    assert verified == identity
    assert dataset_identity_from_artifact(artifact) == verified.dataset


def test_verify_artifact_manifest_fails_closed_without_dataset_identity(tmp_path: Path) -> None:
    frame = _frame()
    cfg = _cfg(tmp_path)
    _materialize_sources(tmp_path, frame)
    artifact = tmp_path / "legacy.joblib"
    save_joblib_artifact({"model": 1}, artifact)  # no identity block at all
    with pytest.raises(ArtifactManifestError):
        verify_artifact_manifest(artifact)
    with pytest.raises(ArtifactManifestError):
        dataset_identity_from_artifact(artifact)

    bind_artifact_identity(artifact, _identity(frame, cfg))
    verified = verify_artifact_manifest(artifact)
    assert verified.dataset.row_count == frame.height


def test_verify_artifact_manifest_rejects_partial_dataset_identity(tmp_path: Path) -> None:
    frame = _frame()
    cfg = _cfg(tmp_path)
    _materialize_sources(tmp_path, frame)
    artifact = tmp_path / "model.joblib"
    save_training_artifact({"model": 1}, artifact, identity=_identity(frame, cfg))
    manifest_file = manifest_path(artifact)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    del manifest["identity"]["dataset"]["materialized_panel_sha256"]
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactManifestError):
        verify_artifact_manifest(artifact)


def test_dataset_identity_fails_closed_on_missing_sources(tmp_path: Path) -> None:
    frame = _frame()
    _cfg(tmp_path)  # gold/silver never written; side effects only
    with pytest.raises(DatasetIdentityError):
        build_dataset_identity(
            frame,
            data_root=tmp_path,
            data_source="synthetic",
            label="future_return_5",
            label_horizon_bars=5,
        )


def test_dataset_identity_fails_closed_on_empty_panel(tmp_path: Path) -> None:
    frame = _frame().head(0)
    cfg = _cfg(tmp_path)
    _materialize_sources(tmp_path, _frame())
    with pytest.raises(DatasetIdentityError):
        build_dataset_identity(
            frame,
            data_root=tmp_path,
            data_source="synthetic",
            label="future_return_5",
            label_horizon_bars=5,
        )
    del cfg  # config unused; kept for symmetry of the fixture


def test_no_training_dispatch_save_bypasses_identity_binding() -> None:
    """Static guard: every save in pipeline/train goes through identity binding."""
    train_dir = Path("src/quant_fund/pipeline/train")
    offenders: list[str] = []
    bound_saves = 0
    for path in sorted(train_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "save_joblib_artifact":
                offenders.append(f"{path}:{node.lineno} raw save_joblib_artifact")
            elif isinstance(node.func, ast.Name) and node.func.id == "save_training_artifact":
                bound_saves += 1
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "save":
                offenders.append(f"{path}:{node.lineno} unbound .save()")
    assert offenders == []
    assert bound_saves >= 16, f"expected every dispatch save site bound, found {bound_saves}"


DISPATCHES = (
    ("alpha", "mean"),
    ("regime", "single"),
    ("tail", "historical"),
    ("ranking", "ridge"),
    ("calibration", "isotonic"),
    ("distribution", "gaussian"),
    ("volatility", "rolling"),
    ("reinforcement", "linucb"),
    ("robinhood_plus", "hierarchical_markov"),
)


@pytest.mark.parametrize(("family", "model"), DISPATCHES)
def test_every_training_dispatch_emits_identity_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, family: str, model: str
) -> None:
    frame = _frame()
    cfg = _cfg(tmp_path)
    _materialize_sources(tmp_path, frame)
    _patch_train(monkeypatch, frame)
    result = train_family(cfg, family, model)
    artifact = Path(str(result["path"]))
    assert artifact.is_file()
    assert result.get("manifest_valid") is True
    assert result.get("dataset_identity_valid") is True
    manifest = json.loads(
        artifact.with_name(f"{artifact.name}.manifest.json").read_text(encoding="utf-8")
    )
    dataset = manifest["identity"]["dataset"]
    assert dataset["materialized_panel_sha256"] == canonical_frame_fingerprint(frame)
    verified = verify_artifact_manifest(artifact)
    assert verified.dataset.materialized_panel_sha256 == canonical_frame_fingerprint(frame)


def test_auto_dispatch_rebinds_selected_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = _frame()
    cfg = _cfg(tmp_path)
    _materialize_sources(tmp_path, frame)
    _patch_train(monkeypatch, frame)
    result = train_calibration_auto(cfg)
    auto_path = Path(str(result["path"]))
    selected = auto_path.parent / f"calibrator_{result['selected_model']}.joblib"
    assert auto_path.name == "calibrator_auto.joblib"
    auto_identity = verify_artifact_manifest(auto_path)
    selected_identity = verify_artifact_manifest(selected)
    assert auto_identity == selected_identity
