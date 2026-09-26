"""Format-agnostic local checkpoint loader.

Backends (pickle, joblib, torch, onnx, json) are imported only when that
format is requested, so a missing optional dependency does not break import
or tests. The loader never fetches a remote path. Pickle and torch loads
execute operator-supplied bytes; point them only at checkpoints you trust.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_file

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".joblib": "joblib",
    ".pt": "torch",
    ".pth": "torch",
    ".onnx": "onnx",
    ".json": "json",
}


class ArtifactBackendUnavailable(ImportError):
    """The requested checkpoint backend is not installed."""

    def __init__(self, backend: str) -> None:
        super().__init__(
            f"checkpoint backend {backend!r} is not installed; "
            "the harness stays importable without this optional dependency"
        )
        self.backend = backend


@dataclass(frozen=True)
class LoadedArtifact:
    """A local checkpoint plus the stamp recorded in run metadata."""

    path: str
    format: str
    sha256: str
    version: str | None
    payload: object


def import_optional(module: str) -> Any:
    """Import ``module`` or raise :class:`ArtifactBackendUnavailable`."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ArtifactBackendUnavailable(module) from exc


def resolve_format(path: Path, fmt: str | None) -> str:
    if fmt and fmt != "auto":
        return fmt
    resolved = _FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if resolved is None:
        raise ValueError(
            f"cannot infer checkpoint format from {path.name!r}; "
            f"set model.checkpoint_format to one of {sorted(set(_FORMAT_BY_SUFFIX.values()))}"
        )
    return resolved


def _sidecar_version(path: Path) -> str | None:
    sidecar = Path(str(path) + ".version")
    if not sidecar.is_file():
        return None
    text = sidecar.read_text(encoding="utf-8").strip()
    return text or None


def _payload_version(payload: object) -> str | None:
    if isinstance(payload, dict) and payload.get("version") is not None:
        return str(payload["version"])
    version = getattr(payload, "version", None)
    if isinstance(version, str) and version:
        return version
    return None


def probe_artifact(path: str | Path, fmt: str | None = None) -> dict[str, str | None]:
    """Hash and version-stamp a checkpoint without deserializing it.

    Version comes from a sibling ``<path>.version`` file when present.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {file_path}")
    return {
        "path": str(file_path),
        "format": resolve_format(file_path, fmt),
        "sha256": hash_file(file_path),
        "version": _sidecar_version(file_path),
    }


def _load_pickle(path: Path) -> object:
    pickle = import_optional("pickle")
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301  # trusted local checkpoint


def _load_joblib(path: Path) -> object:
    joblib = import_optional("joblib")
    return joblib.load(path)


def _load_torch(path: Path) -> object:
    torch = import_optional("torch")
    # Operator-local checkpoint. weights_only=False because an external fx-1
    # artifact may be a full module, not a tensor dict. Do not point this at
    # untrusted bytes.
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_onnx(path: Path) -> object:
    onnx = import_optional("onnx")
    return onnx.load(str(path))


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


_BACKENDS = {
    "pickle": _load_pickle,
    "joblib": _load_joblib,
    "torch": _load_torch,
    "onnx": _load_onnx,
    "json": _load_json,
}


def load_artifact(path: str | Path, fmt: str | None = None) -> LoadedArtifact:
    """Load a local checkpoint and stamp its sha256 and version."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {file_path}")
    resolved = resolve_format(file_path, fmt)
    backend = _BACKENDS.get(resolved)
    if backend is None:
        raise ValueError(f"unsupported checkpoint format {resolved!r}")
    payload = backend(file_path)
    version = _sidecar_version(file_path) or _payload_version(payload)
    return LoadedArtifact(
        path=str(file_path),
        format=resolved,
        sha256=hash_file(file_path),
        version=version,
        payload=payload,
    )
