"""Universal ``model_artifact.v1`` manifests with fail-closed dataset identity.

Every training-dispatch path persists its artifact through
:func:`save_training_artifact` (or re-binds an already persisted artifact with
:func:`bind_artifact_identity`). The base ``model_artifact.v1`` manifest
written by ``quant_fund.models.base`` binds the payload hash, artifact class,
feature list and payload-provided provenance. This module adds the identity
block that manifest previously could not carry:

* ``dataset`` — materialized-panel hash, source-manifest hash, row/column
  counts, observed time range, label + label horizon;
* ``features`` — the exact feature set with its version and set hash;
* ``label`` / ``horizon`` — label identity and the declared horizon grid;
* ``config_sha256`` — canonical hash of the training config;
* ``git_revision`` / ``git_worktree_sha256`` — code identity including a
  dirty-worktree hash.

The identity block is additive: ``models.base._verify_manifest`` checks only
the fields it wrote, so artifacts stay loadable through the existing
``load_joblib_artifact`` / ``JoblibMixin.load`` paths.

Missing dataset identity is impossible to omit silently:

1. :func:`save_training_artifact` takes ``identity`` as a required
   keyword-only argument (no default), and :func:`bind_artifact_identity`
   refuses to write a partial block;
2. :func:`verify_artifact_manifest` fails closed on any absent, malformed or
   mismatched identity field;
3. promotion composition (``pipeline.promotion_receipt``) calls
   :func:`verify_artifact_manifest` and refuses to proceed without it.

Research-only; no live-trading claim is made or implied here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from quant_fund.features.metadata import FEATURE_SET_VERSION
from quant_fund.models.base import save_joblib_artifact
from quant_fund.utils.hashing import (
    canonical_frame_fingerprint,
    canonical_json_bytes,
    hash_bytes,
    hash_file,
)
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

MODEL_ARTIFACT_SCHEMA: Final = "model_artifact.v1"
ARTIFACT_IDENTITY_SCHEMA: Final = "artifact_identity.v1"
MANIFEST_SUFFIX: Final = ".manifest.json"
IDENTITY_BLOCK_KEY: Final = "identity"
# Label identity for artifacts fitted without a supervised target.
UNSUPERVISED_LABEL: Final = "unsupervised"
# Materialized inputs of a gold panel; the source manifest binds their bytes.
SOURCE_PANEL_PARTS: Final[tuple[str, ...]] = (
    "gold/features.parquet",
    "gold/labels.parquet",
    "silver/universe.parquet",
)

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ArtifactManifestError(RuntimeError):
    """Raised when an artifact manifest is absent, malformed or unbound."""


class DatasetIdentityError(RuntimeError):
    """Raised when dataset identity cannot be derived fail-closed."""


class DatasetIdentity(BaseModel):
    """What data produced the artifact — never optional, never implicit."""

    model_config = ConfigDict(extra="forbid")

    materialized_panel_sha256: Sha256Hex
    source_manifest_sha256: Sha256Hex
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    time_start: str = Field(min_length=1)
    time_end: str = Field(min_length=1)
    label: str = Field(min_length=1)
    label_horizon_bars: int = Field(ge=1)
    data_source: str = Field(min_length=1)

    @field_validator("time_start", "time_end")
    @classmethod
    def _parseable_timestamp(cls, value: str) -> str:
        from datetime import datetime

        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"not an ISO-8601 timestamp: {value!r}") from exc
        return value


class FeatureIdentity(BaseModel):
    """The exact feature set the artifact was fitted on."""

    model_config = ConfigDict(extra="forbid")

    features: list[str] = Field(min_length=1)
    feature_set_version: str = Field(min_length=1)
    feature_set_sha256: Sha256Hex


class LabelIdentity(BaseModel):
    """Supervised target identity (label column + trailing horizon bars)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    horizon_bars: int = Field(ge=1)


class HorizonIdentity(BaseModel):
    """Declared forecast horizon grid from the training config."""

    model_config = ConfigDict(extra="forbid")

    bars: list[int] = Field(min_length=1)
    names: list[str] = Field(min_length=1)

    @field_validator("bars")
    @classmethod
    def _positive_bars(cls, value: list[int]) -> list[int]:
        if any(isinstance(bar, bool) or bar < 1 for bar in value):
            raise ValueError("horizon bars must be positive integers")
        return value


class ArtifactIdentity(BaseModel):
    """The full identity block bound into a ``model_artifact.v1`` manifest."""

    model_config = ConfigDict(extra="forbid")

    identity_schema: Literal["artifact_identity.v1"] = ARTIFACT_IDENTITY_SCHEMA
    dataset: DatasetIdentity
    features: FeatureIdentity
    label: LabelIdentity
    horizon: HorizonIdentity
    config_sha256: Sha256Hex
    git_revision: str = Field(min_length=1)
    git_worktree_sha256: str = Field(min_length=1)


def manifest_path(artifact: Path) -> Path:
    """Return the ``model_artifact.v1`` manifest path for an artifact file."""
    path = Path(artifact)
    return path.with_name(f"{path.name}{MANIFEST_SUFFIX}")


def _source_manifest_hash(data_root: Path) -> str:
    """Hash the materialized source artifacts a gold panel is built from.

    Fails closed when any source artifact is absent: a dataset identity built
    from a partial source set would be a fabricated lineage.
    """
    entries: dict[str, str] = {}
    for relative in SOURCE_PANEL_PARTS:
        source = Path(data_root) / relative
        if not source.is_file():
            raise DatasetIdentityError(f"dataset source artifact missing: {source}")
        entries[relative] = hash_file(source)
    return hash_bytes(canonical_json_bytes(entries))


def _require_column(frame: Any, name: str) -> None:
    if name not in list(frame.columns):
        raise DatasetIdentityError(f"dataset panel is missing required column {name!r}")


def build_dataset_identity(
    frame: Any,
    *,
    data_root: Path,
    data_source: str,
    label: str,
    label_horizon_bars: int,
) -> DatasetIdentity:
    """Derive dataset identity from the materialized panel and its sources."""
    if getattr(frame, "height", 0) < 1:
        raise DatasetIdentityError("dataset panel is empty")
    _require_column(frame, "event_time")
    _require_column(frame, "security_id")
    if not str(data_source).strip():
        raise DatasetIdentityError("dataset identity requires a nonempty data source")
    if not str(label).strip():
        raise DatasetIdentityError("dataset identity requires a nonempty label")
    if isinstance(label_horizon_bars, bool) or int(label_horizon_bars) < 1:
        raise DatasetIdentityError("dataset identity requires a positive label horizon")
    times = frame.select(["event_time"]).sort("event_time").to_dicts()
    return DatasetIdentity(
        materialized_panel_sha256=canonical_frame_fingerprint(frame),
        source_manifest_sha256=_source_manifest_hash(Path(data_root)),
        row_count=int(frame.height),
        column_count=len(list(frame.columns)),
        time_start=str(times[0]["event_time"]),
        time_end=str(times[-1]["event_time"]),
        label=str(label),
        label_horizon_bars=int(label_horizon_bars),
        data_source=str(data_source),
    )


def build_artifact_identity(
    *,
    dataset: DatasetIdentity,
    features: list[str],
    label: str,
    label_horizon_bars: int,
    horizon_bars: list[int],
    horizon_names: list[str],
    config: BaseModel,
) -> ArtifactIdentity:
    """Assemble the complete identity block for an artifact about to be saved."""
    names = [str(feature) for feature in features if str(feature).strip()]
    if not names:
        raise DatasetIdentityError("artifact identity requires a nonempty feature set")
    if len(set(names)) != len(names):
        raise DatasetIdentityError("artifact feature set contains duplicate names")
    return ArtifactIdentity(
        dataset=dataset,
        features=FeatureIdentity(
            features=names,
            feature_set_version=str(FEATURE_SET_VERSION),
            feature_set_sha256=hash_bytes(canonical_json_bytes(sorted(names))),
        ),
        label=LabelIdentity(name=str(label), horizon_bars=int(label_horizon_bars)),
        horizon=HorizonIdentity(bars=[int(bar) for bar in horizon_bars], names=list(horizon_names)),
        config_sha256=hash_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        git_revision=git_revision(),
        git_worktree_sha256=git_worktree_sha256(),
    )


def _read_manifest(artifact: Path) -> dict[str, Any]:
    path = manifest_path(artifact)
    if not path.is_file():
        raise ArtifactManifestError(f"model artifact manifest is missing: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"model artifact manifest is unreadable: {path}") from exc
    if not isinstance(record, dict) or record.get("schema") != MODEL_ARTIFACT_SCHEMA:
        raise ArtifactManifestError(
            f"model artifact manifest is not {MODEL_ARTIFACT_SCHEMA}: {path}"
        )
    return record


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def bind_artifact_identity(artifact: Path, identity: ArtifactIdentity) -> dict[str, Any]:
    """Attach (or re-attach) the identity block to a written manifest.

    Fails closed when the artifact has no valid base manifest or when the base
    manifest does not hash-bind the artifact bytes: identity is never attached
    to an unbound payload.
    """
    path = Path(artifact)
    record = _read_manifest(path)
    if record.get("artifact") != path.name or record.get("sha256") != hash_file(path):
        raise ArtifactManifestError(f"model artifact manifest is unbound to payload: {path}")
    record[IDENTITY_BLOCK_KEY] = identity.model_dump(mode="json")
    _atomic_write_json(manifest_path(path), record)
    return record


def verify_artifact_manifest(artifact: Path) -> ArtifactIdentity:
    """Verify and return the identity block; fail closed on any gap."""
    path = Path(artifact)
    if not path.is_file():
        raise ArtifactManifestError(f"model artifact is missing: {path}")
    record = _read_manifest(path)
    if record.get("artifact") != path.name:
        raise ArtifactManifestError(f"model artifact manifest name mismatch: {path}")
    if record.get("sha256") != hash_file(path):
        raise ArtifactManifestError(f"model artifact manifest hash mismatch: {path}")
    block = record.get(IDENTITY_BLOCK_KEY)
    if not isinstance(block, dict):
        raise ArtifactManifestError(f"model artifact manifest has no dataset identity: {path}")
    try:
        identity = ArtifactIdentity.model_validate(block)
    except ValueError as exc:
        raise ArtifactManifestError(f"model artifact identity is malformed: {path}") from exc
    return identity


def save_training_artifact(payload: Any, path: Path, *, identity: ArtifactIdentity) -> Path:
    """Persist an artifact with a full ``model_artifact.v1`` identity manifest.

    ``identity`` is required by construction — a training dispatch cannot save
    an artifact without declaring what data, features, label and horizon
    produced it.
    """
    destination = Path(path)
    save_joblib_artifact(payload, destination)
    bind_artifact_identity(destination, identity)
    verify_artifact_manifest(destination)
    return destination


def identity_for_training(
    frame: Any,
    *,
    config: Any,
    label: str,
    features: list[str],
    label_horizon_bars: int,
) -> ArtifactIdentity:
    """Build the complete identity block for one training dispatch.

    ``config`` is an ``AppConfig``. ``label`` is the supervised target column,
    or :data:`UNSUPERVISED_LABEL` for artifacts fitted without one (the
    horizon then records the declared forecast grid only).
    """
    dataset = build_dataset_identity(
        frame,
        data_root=Path(config.data.root),
        data_source=str(config.data.source),
        label=label,
        label_horizon_bars=label_horizon_bars,
    )
    return build_artifact_identity(
        dataset=dataset,
        features=features,
        label=label,
        label_horizon_bars=label_horizon_bars,
        horizon_bars=[int(bar) for bar in config.horizon.bars],
        horizon_names=[str(name) for name in config.horizon.names],
        config=config,
    )


def identity_from_artifact(artifact: Path) -> ArtifactIdentity:
    """Return the verified identity of a persisted artifact (alias)."""
    return verify_artifact_manifest(artifact)


def dataset_identity_from_artifact(artifact: Path) -> DatasetIdentity:
    """Return just the verified dataset identity of a persisted artifact."""
    return verify_artifact_manifest(artifact).dataset


__all__ = [
    "ARTIFACT_IDENTITY_SCHEMA",
    "ArtifactIdentity",
    "ArtifactManifestError",
    "DatasetIdentity",
    "DatasetIdentityError",
    "FeatureIdentity",
    "HorizonIdentity",
    "LabelIdentity",
    "MANIFEST_SUFFIX",
    "MODEL_ARTIFACT_SCHEMA",
    "UNSUPERVISED_LABEL",
    "bind_artifact_identity",
    "build_artifact_identity",
    "build_dataset_identity",
    "dataset_identity_from_artifact",
    "identity_for_training",
    "identity_from_artifact",
    "manifest_path",
    "save_training_artifact",
    "verify_artifact_manifest",
]
