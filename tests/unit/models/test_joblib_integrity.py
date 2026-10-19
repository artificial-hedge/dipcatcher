"""joblib (pickle) artifacts are unpickled only after their bytes verify.

``load_joblib_artifact`` and ``JoblibMixin.load`` require the ``.sha256``
sidecar every in-repo writer publishes, and the forecast paper-challenger stamp
loads through the same gate. Each refusal is proven with a payload whose
deserialization writes a marker file: the marker must never appear.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.models.base import JoblibMixin, load_joblib_artifact, save_joblib_artifact
from quant_fund.pipeline.forecast.artifacts import _paper_challenger_stamp


class _MarkerGadget:
    """Pickle whose deserialization writes ``marker``: proof the bytes were unpickled."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[Path, str]]:
        return (Path.write_text, (self.marker, "unpickled"))


class _ColumnRanker:
    """Picklable paper-ranker stand-in that scores by its first feature column."""

    def predict(self, x: Any, dates: Any = None, ids: Any = None) -> Any:
        return np.asarray(x, dtype=float)[:, 0]


def _gadget(path: Path, marker: Path) -> Path:
    joblib.dump(_MarkerGadget(marker), path)
    return path


def _sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def _assert_integrity_refusal(excinfo: pytest.ExceptionInfo[ValueError], marker: Path) -> None:
    """The refusal is the typed pre-deserialization error and nothing was unpickled."""
    from quant_fund.models.base import ArtifactIntegrityError

    assert isinstance(excinfo.value, ArtifactIntegrityError)
    assert not marker.exists()


def test_load_joblib_artifact_refuses_missing_sidecar_without_unpickling(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    artifact = _gadget(tmp_path / "payload.joblib", marker)

    with pytest.raises(ValueError, match="checksum sidecar is missing") as excinfo:
        load_joblib_artifact(artifact)
    _assert_integrity_refusal(excinfo, marker)


def test_model_load_refuses_missing_sidecar_without_unpickling(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    artifact = _gadget(tmp_path / "model.joblib", marker)

    with pytest.raises(ValueError, match="model artifact checksum sidecar is missing") as excinfo:
        JoblibMixin.load(artifact)
    _assert_integrity_refusal(excinfo, marker)


_TRUE_DIGEST = b"<sha256 of the artifact bytes>"


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not-hex\n",
        b"0x" + b"a" * 62 + b"\n",
        b"A" * 64 + b"\n",
        # Non-ASCII bytes around the artifact's *true* digest: a lenient decode
        # (utf-8-sig, errors="ignore") would accept the pin and unpickle.
        pytest.param(b"\xef\xbb\xbf" + _TRUE_DIGEST + b"\n", id="utf8-bom-before-true-digest"),
    ],
)
def test_malformed_sidecar_is_refused_without_unpickling(tmp_path: Path, content: bytes) -> None:
    marker = tmp_path / "ran"
    artifact = _gadget(tmp_path / "payload.joblib", marker)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest().encode("ascii")
    _sidecar(artifact).write_bytes(content.replace(_TRUE_DIGEST, digest))

    with pytest.raises(ValueError, match="checksum sidecar is malformed") as excinfo:
        load_joblib_artifact(artifact)
    _assert_integrity_refusal(excinfo, marker)


def test_mismatched_sidecar_is_refused_without_unpickling(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    artifact = _gadget(tmp_path / "payload.joblib", marker)
    _sidecar(artifact).write_text("0" * 64 + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="checksum mismatch") as excinfo:
        load_joblib_artifact(artifact)
    _assert_integrity_refusal(excinfo, marker)
    with pytest.raises(ValueError, match="model artifact checksum mismatch") as excinfo:
        JoblibMixin.load(artifact)
    _assert_integrity_refusal(excinfo, marker)


def test_writer_output_round_trips_through_the_required_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.joblib"
    payload = {"features": ["close"], "weights": [1.0]}

    save_joblib_artifact(payload, artifact)

    assert _sidecar(artifact).read_text(encoding="ascii").strip() == (
        hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    assert load_joblib_artifact(artifact) == payload


def test_load_trusted_joblib_requires_a_well_formed_matching_pin(tmp_path: Path) -> None:
    from quant_fund.models.base import ArtifactIntegrityError, load_trusted_joblib

    marker = tmp_path / "ran"
    artifact = _gadget(tmp_path / "head.joblib", marker)

    with pytest.raises(ArtifactIntegrityError, match="64 lowercase hex"):
        load_trusted_joblib(artifact, "deadbeef")
    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        load_trusted_joblib(artifact, "f" * 64)
    assert not marker.exists()

    trusted = tmp_path / "trusted.joblib"
    joblib.dump({"value": 1}, trusted)
    digest = hashlib.sha256(trusted.read_bytes()).hexdigest()
    assert load_trusted_joblib(trusted, digest) == {"value": 1}


def _paper_config(tmp_path: Path) -> tuple[AppConfig, Path]:
    cfg = AppConfig()
    cfg.data.root = tmp_path
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    return cfg, metadata / f"ranker_{cfg.train.paper_rankers[0]}.joblib"


@pytest.mark.parametrize("sidecar", [None, "0" * 64 + "\n"], ids=["missing", "mismatched"])
def test_paper_challenger_stamp_refuses_unverified_ranker_without_unpickling(
    tmp_path: Path, sidecar: str | None
) -> None:
    cfg, artifact = _paper_config(tmp_path)
    marker = tmp_path / "ran"
    _gadget(artifact, marker)
    if sidecar is not None:
        _sidecar(artifact).write_text(sidecar, encoding="ascii")

    with pytest.raises(ValueError, match="checksum") as excinfo:
        _paper_challenger_stamp(cfg, np.zeros((4, 1)), np.zeros(4))
    _assert_integrity_refusal(excinfo, marker)


def test_paper_challenger_stamp_lists_a_verified_ranker(tmp_path: Path) -> None:
    cfg, artifact = _paper_config(tmp_path)
    save_joblib_artifact(_ColumnRanker(), artifact)
    x = np.arange(8, dtype=float).reshape(-1, 1)

    note, extra = _paper_challenger_stamp(cfg, x, np.arange(8, dtype=float))

    name = cfg.train.paper_rankers[0]
    assert note == f"paper_challengers={name}"
    assert extra[f"paper_{name}_spearman_vs_ridge"] == pytest.approx(1.0)
    assert extra["paper_challengers_blend"] == 0.0
