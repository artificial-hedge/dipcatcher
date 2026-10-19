"""Checkout-root resolution for the dipcatcher launcher and lab CLI."""

from __future__ import annotations

from pathlib import Path

from quant_fund.audit.kernel_lineage import default_repo_root


def repo_root() -> Path:
    """Absolute path to the dipcatcher checkout (directory with ``pyproject.toml``)."""
    return default_repo_root()


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve *path* against :func:`repo_root` when it is relative."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root() / candidate).resolve()


__all__ = ["repo_root", "resolve_repo_path"]
