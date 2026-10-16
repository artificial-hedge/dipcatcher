"""Serving gates for fixture and signed public checkpoints."""

from __future__ import annotations

import os
import stat
import threading
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from fx1.modelcard import ModelCard
from fx1.serve.backends import BackendNotConfiguredError, _checkpoint_weights_sha
from fx1.serve.signing import (
    MANIFEST_FILENAME,
    PUBLIC_SIGNING_KEY_ENV,
    PUBLIC_VERIFY_KEY_ENV,
    ReleaseManifest,
    verify_release_ed25519,
)

ALLOW_TEST_FIXTURE_ENV = "FX1_ALLOW_TEST_FIXTURE"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
PUBLIC_MODEL_IDS = ("fx1", "fx1-lite")
IMMUTABLE_IMAGE_ENV = "FX1_RELEASE_IMMUTABLE_IMAGE"
_VERIFIED_PUBLIC_RELEASES: dict[
    tuple[str, str], tuple[tuple[str, int, int, int, int, int, int], ...]
] = {}
_VERIFIED_LOCK = threading.Lock()


def _read_only_mount(root: Path) -> bool:
    """Require an OS read-only volume, not a self-attested env flag."""
    if os.name != "nt":
        statvfs = cast(Any, os).statvfs
        return bool(statvfs(root).f_flag & getattr(os, "ST_RDONLY", 1))
    import ctypes

    volume = ctypes.create_unicode_buffer(32768)
    # Linux typeshed omits WinDLL even inside this Windows-only branch.
    kernel32 = cast(Any, ctypes).WinDLL("kernel32", use_last_error=True)
    get_path = kernel32.GetVolumePathNameW
    get_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    get_path.restype = ctypes.c_int
    if not get_path(str(root), volume, len(volume)):
        return False
    flags = ctypes.c_uint32()
    get_info = kernel32.GetVolumeInformationW
    get_info.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_info.restype = ctypes.c_int
    if not get_info(volume.value, None, 0, None, None, ctypes.byref(flags), None, 0):
        return False
    return bool(flags.value & 0x00080000)  # FILE_READ_ONLY_VOLUME


def _inventory_stat(root: Path) -> tuple[tuple[str, int, int, int, int, int, int], ...]:
    """Cheap closed-inventory drift check after full signed verification."""
    entries: list[tuple[str, int, int, int, int, int, int]] = []
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        parent = Path(directory)
        paths = [root] if parent == root else []
        paths.extend(parent / name for name in sorted([*dirs, *files]))
        for path in paths:
            info = path.lstat()
            if (
                not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))
                or getattr(info, "st_file_attributes", 0) & 0x400  # FILE_ATTRIBUTE_REPARSE_POINT
                or not _read_only_mount(path)
            ):
                raise ValueError(f"public release contains a mutable or nonregular path: {path}")
            entries.append(
                (
                    str(path.relative_to(root)),
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
            )
    return tuple(entries)


def _no_link_ancestors(path: Path) -> bool:
    """An alias path cannot redirect through a symlink or Windows junction."""
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            return False
    return True


def _verified_public_release(root: Path) -> bool:
    """Hash once on an OS read-only volume; refuse any later metadata drift."""
    try:
        if os.environ.get(IMMUTABLE_IMAGE_ENV) != "1":
            return False
        root = Path(os.path.abspath(root))
        if not _no_link_ancestors(root):
            return False
        root = root.resolve(strict=True)
        if not _read_only_mount(root):
            return False
        key = (str(root), os.environ.get(PUBLIC_VERIFY_KEY_ENV, ""))
        before = _inventory_stat(root)
        with _VERIFIED_LOCK:
            cached = _VERIFIED_PUBLIC_RELEASES.get(key)
        if cached is not None:
            return cached == before
        if not verify_release_ed25519(root):
            return False
        after = _inventory_stat(root)
        if after != before:
            return False
        with _VERIFIED_LOCK:
            existing = _VERIFIED_PUBLIC_RELEASES.setdefault(key, after)
        return existing == after
    except (OSError, ValueError):
        return False


def public_checkpoint_env(model_id: str) -> str:
    """Explicit per-alias checkpoint binding for a public model name."""
    if model_id not in PUBLIC_MODEL_IDS:
        raise ValueError(f"unknown public model id: {model_id!r}")
    return f"FX1_CHECKPOINT_DIR_{model_id.upper().replace('-', '_')}"


def public_serve_url_env(model_id: str) -> str:
    """A public alias needs its own persistent engine endpoint."""
    if model_id not in PUBLIC_MODEL_IDS:
        raise ValueError(f"unknown public model id: {model_id!r}")
    return f"FX1_LOCAL_SERVE_URL_{model_id.upper().replace('-', '_')}"


def _matches_public_checkpoint_pin(root: Path, alias: str) -> bool:
    """Accept only the operator's immutable, per-alias checkpoint path."""
    configured = os.environ.get(public_checkpoint_env(alias))
    if not configured:
        return False
    try:
        requested = Path(os.path.abspath(root))
        pinned = Path(os.path.abspath(configured))
        return (
            _no_link_ancestors(requested)
            and _no_link_ancestors(pinned)
            and requested.resolve(strict=True) == pinned.resolve(strict=True)
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _trusted_public_serve_url(url: str) -> bool:
    """Public weight probes trust only a same-host engine endpoint."""
    try:
        parsed = urllib.parse.urlsplit(url)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "::1"}
            and parsed.port is not None
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/v1", "/v1/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def require_public_runtime_env(enabled: bool) -> None:
    """A public inference process must never hold the release private seed."""
    if enabled and os.environ.get(PUBLIC_SIGNING_KEY_ENV):
        raise RuntimeError(
            f"public inference refuses {PUBLIC_SIGNING_KEY_ENV}; "
            "keep the private signing seed on the release build host"
        )


def _public_checkpoint_alias(root: Path) -> str | None:
    """Read the signed inventory's alias and pin; signature is checked by caller."""
    try:
        card = ModelCard.load(root / "modelcard.json")
        if (
            card.fixture_only
            or card.public_model_id is None
            or not card.eval_delta.ship_eligible
            or card.license_tier not in {"maas", "commercial"}
        ):
            return None
        release = ReleaseManifest.model_validate_json((root / MANIFEST_FILENAME).read_bytes())
        pin = _checkpoint_weights_sha(root)
        if pin is None or release.artifacts.get("weights.safetensors") != pin:
            return None
        return card.public_model_id
    except (OSError, ValueError, RuntimeError):
        return None


def configured_public_models() -> tuple[str, ...]:
    """Only advertise aliases with explicit paths and valid signed weights."""
    if not os.environ.get(PUBLIC_VERIFY_KEY_ENV):
        return ()
    available: list[str] = []
    for alias in PUBLIC_MODEL_IDS:
        path = os.environ.get(public_checkpoint_env(alias))
        url = os.environ.get(public_serve_url_env(alias))
        if (
            path
            and url
            and _trusted_public_serve_url(url)
            and _public_checkpoint_alias(Path(path)) == alias
            and _verified_public_release(Path(path))
        ):
            available.append(alias)
    return tuple(available)


def require_nonfixture_or_test_opt_in(card: ModelCard) -> None:
    """Keep synthetic correctness fixtures out of ordinary model serving."""
    if card.fixture_only and os.environ.get(ALLOW_TEST_FIXTURE_ENV) != "1":
        raise RuntimeError(
            "synthetic fixture checkpoint is not a release candidate; "
            f"{ALLOW_TEST_FIXTURE_ENV}=1 is for isolated internal tests only"
        )


def require_safe_fixture_bind(card: ModelCard, host: str) -> None:
    """Even a test opt-in never permits a fixture engine on a public bind."""
    if card.fixture_only and host not in _LOOPBACK_HOSTS:
        raise RuntimeError("synthetic fixture checkpoint may bind only to loopback")


def require_loopback_engine_bind(host: str) -> None:
    """The thin weights engine has no caller authentication; keep it local."""
    if host not in _LOOPBACK_HOSTS:
        raise RuntimeError("local weights engine may bind only to loopback")


def public_release_resolver(
    resolver: Callable[..., Any], *, enabled: bool = True
) -> Callable[..., Any]:
    """Require signed local weights on every public API resolution path."""
    if not enabled:
        return resolver

    def resolve(kind: str, **kwargs: Any) -> Any:
        require_public_runtime_env(True)
        if kind == "local_fx1":
            kwargs = _public_local_kwargs(kwargs)
        return resolver(kind, **kwargs)

    return resolve


def _public_local_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get(ALLOW_TEST_FIXTURE_ENV) == "1":
        raise BackendNotConfiguredError(
            f"public local_fx1 serving refuses {ALLOW_TEST_FIXTURE_ENV}=1"
        )
    if not os.environ.get(PUBLIC_VERIFY_KEY_ENV):
        raise BackendNotConfiguredError(
            f"public local_fx1 serving requires {PUBLIC_VERIFY_KEY_ENV} "
            "and an Ed25519-signed checkpoint"
        )
    checkpoint_dir = kwargs.get("checkpoint_dir")
    if not isinstance(checkpoint_dir, (str, Path)) or not checkpoint_dir:
        raise BackendNotConfiguredError("public local_fx1 requires a configured checkpoint_dir")
    root = Path(checkpoint_dir)
    alias = _public_checkpoint_alias(root)
    if alias is None:
        raise BackendNotConfiguredError(
            "public local_fx1 requires an Ed25519-signed release with a "
            "model card binding fx1 or fx1-lite, a public-serving license "
            "tier, and a weights SHA-256 pin"
        )
    if not _matches_public_checkpoint_pin(root, alias):
        raise BackendNotConfiguredError(
            f"public {alias} requires the checkpoint bound by {public_checkpoint_env(alias)}"
        )
    serve_url = os.environ.get(public_serve_url_env(alias))
    if not serve_url or not _trusted_public_serve_url(serve_url):
        raise BackendNotConfiguredError(
            f"public {alias} requires {public_serve_url_env(alias)} pointing to "
            "a persistent 127.0.0.1 or [::1] /v1 engine"
        )
    requested = kwargs.get("model")
    if requested is not None and requested != alias:
        raise BackendNotConfiguredError(
            f"signed checkpoint binds {alias!r}; requested model {requested!r} refused"
        )
    if not _verified_public_release(root):
        raise BackendNotConfiguredError(
            "public local_fx1 requires a valid Ed25519 signature on an OS read-only "
            "checkpoint volume; changed inventory is refused until process restart"
        )
    # The release was verified against a read-only volume above. The backend
    # still probes the live engine's weights SHA on every call; avoid hashing
    # K3-sized artifacts a second time in its constructor.
    return {
        **kwargs,
        "model": alias,
        "serve_url": serve_url,
        "serve_cmd": "",
        "require_signature": False,
    }
