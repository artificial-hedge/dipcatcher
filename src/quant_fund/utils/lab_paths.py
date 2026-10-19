"""Checkout-rooted paths so lab commands do not depend on the process cwd."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

LAB_SCRUB_KEYS: frozenset[str] = frozenset({"MOONSHOT_API_KEY"})
_REPO_SRC = ("src/quant_fund", "src/fx1")


def _looks_like_repo(path: Path) -> bool:
    if not (path / "pyproject.toml").is_file():
        return False
    return any((path / marker).exists() for marker in _REPO_SRC)


def discover_repo_root() -> Path:
    """Resolve the dipcatcher checkout.

    Order: ``DIPCATCHER_ROOT`` / ``QUANT_REPO_ROOT``, cwd parents, this
    package's parents. Falls back to cwd when no checkout is found.
    """
    for key in ("DIPCATCHER_ROOT", "QUANT_REPO_ROOT"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if _looks_like_repo(candidate):
            return candidate
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        if _looks_like_repo(parent):
            return parent
    for parent in Path(__file__).resolve().parents:
        if _looks_like_repo(parent):
            return parent
    return here


def configs_dir() -> Path:
    return discover_repo_root() / "configs"


def resolve_config_path(config: str | Path) -> Path:
    """Resolve a config path against cwd first, then the checkout root."""
    raw = Path(config).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    return (discover_repo_root() / raw).resolve()


def resolve_data_root(configured: str | Path | None = None) -> Path:
    """Data lake root: ``QUANT_DATA_ROOT``, else *configured*, else ``<repo>/data``."""
    env = os.environ.get("QUANT_DATA_ROOT", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.is_absolute():
            return path.resolve()
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.exists() or _looks_like_repo(Path.cwd()):
            return cwd_candidate
        return (discover_repo_root() / path).resolve()
    if configured is not None:
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path.resolve()
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.exists() or _looks_like_repo(Path.cwd()):
            return cwd_candidate
        return (discover_repo_root() / path).resolve()
    return (discover_repo_root() / "data").resolve()


def data_dir_status(path: Path, *, create: bool = True) -> str:
    """``ok`` if the directory already existed, ``created`` if we made it."""
    if path.is_dir():
        return "ok"
    if not create:
        return "missing"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return "missing"
    return "created" if path.is_dir() else "missing"


def lab_child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy *base* (default ``os.environ``) with the model API key removed."""
    env = dict(os.environ if base is None else base)
    for key in LAB_SCRUB_KEYS:
        env.pop(key, None)
    return env


__all__ = [
    "LAB_SCRUB_KEYS",
    "configs_dir",
    "data_dir_status",
    "discover_repo_root",
    "lab_child_env",
    "resolve_config_path",
    "resolve_data_root",
]
