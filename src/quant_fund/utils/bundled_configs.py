"""Resolve lab config defaults shipped with the package.

``<cwd>/configs/<file>`` wins when that file exists. A relative
``configs/<file>`` that is absent falls back to ``quant_fund/_bundled/configs``
(the wheel) or, in a source checkout, the repository ``configs/`` directory.
``load_config`` then resolves ``inherit:`` beside the chosen file.
"""

from __future__ import annotations

import importlib.util
from importlib.resources import files
from pathlib import Path
from typing import TypeGuard

# Cwd defaults in src/quant_fund/cli/*_cmds.py, plus their inherit closure.
# base.yaml <- research.yaml, backtest.yaml, paper.yaml; paper.yaml <- sim_live.yaml.
BUNDLED_CONFIGS: tuple[str, ...] = (
    "base.yaml",
    "research.yaml",
    "backtest.yaml",
    "paper.yaml",
    "sim_live.yaml",
    "net_tournament.json",
)


def resolve_lab_config(path: Path | str) -> Path:
    """Prefer an existing path. Fall back only for a missing ``configs/<name>``."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    bundled = bundled_file(_configs_name(candidate))
    if bundled is None:
        return candidate
    return bundled


def bundled_file(name: str | None) -> Path | None:
    """Return one shipped config file, or None when it is not a plain name."""
    if not _plain_name(name):
        return None
    packaged = _real_file(_packaged_dir(), name)
    if packaged is not None:
        return packaged
    return _real_file(_checkout_configs_dir(), name)


def bundled_configs_dir() -> Path | None:
    """Directory of shipped configs, rejecting a symlinked directory."""
    packaged = _packaged_dir()
    if packaged is not None:
        return packaged
    return _checkout_configs_dir()


def _configs_name(candidate: Path) -> str | None:
    if candidate.is_absolute():
        return None
    parts = candidate.parts
    if len(parts) != 2 or parts[0] != "configs":
        return None
    return parts[1]


def _plain_name(name: str | None) -> TypeGuard[str]:
    if name is None or name in {"", ".", ".."}:
        return False
    return "/" not in name and "\\" not in name and ".." not in name


def _packaged_dir() -> Path | None:
    resource = files("quant_fund").joinpath("_bundled", "configs")
    if not resource.is_dir():
        return None
    path = Path(str(resource))
    if path.is_symlink() or not path.is_dir():
        return None
    return path.resolve()


def _checkout_configs_dir() -> Path | None:
    origin = _package_origin()
    if origin is None:
        return None
    for candidate in (origin.parent, *origin.parent.parents):
        found = _marked_configs(candidate)
        if found is not None:
            return found
    return None


def _marked_configs(candidate: Path) -> Path | None:
    configs = candidate / "configs"
    if not (candidate / "pyproject.toml").is_file():
        return None
    if not configs.is_dir() or configs.is_symlink():
        return None
    return configs.resolve()


def _package_origin() -> Path | None:
    spec = importlib.util.find_spec("quant_fund")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def _real_file(root: Path | None, name: str) -> Path | None:
    if root is None:
        return None
    path = root / name
    if path.is_symlink() or not path.is_file():
        return None
    return path.resolve()
