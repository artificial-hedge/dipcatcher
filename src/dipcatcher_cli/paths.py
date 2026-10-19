"""Checkout-root resolution for the dipcatcher launcher and lab CLI."""

from __future__ import annotations

from pathlib import Path

from quant_fund.utils.lab_paths import discover_repo_root, resolve_config_path


def repo_root() -> Path:
    """Absolute path to the dipcatcher checkout (directory with ``pyproject.toml``)."""
    return discover_repo_root()


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve *path* against :func:`repo_root` when it is relative."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root() / candidate).resolve()


__all__ = ["discover_repo_root", "repo_root", "resolve_config_path", "resolve_repo_path"]
