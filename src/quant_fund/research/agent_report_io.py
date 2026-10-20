"""Research notebook artifact I/O helpers for ``quant_fund.research.agent``."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from quant_fund.reporting.report import write_report
from quant_fund.utils.atomicio import publish_text_once


def atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete text artifact without exposing partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="wb",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content.encode("utf-8"))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def render_research_report(
    directory: Path, title: str, sections: dict[str, Any], *, synthetic: bool
) -> str:
    """Render with the shared report writer before publishing immutable bytes."""
    with NamedTemporaryFile(
        dir=directory, prefix=".research-report.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        write_report(temporary_path, title, sections, synthetic=synthetic)
        return temporary_path.read_text(encoding="utf-8")
    finally:
        temporary_path.unlink(missing_ok=True)


def publish_research_run_once(
    json_path: Path, serialized: str, markdown_path: Path, markdown: str
) -> None:
    """Keep a repeated run ID from replacing either immutable artifact."""
    artifacts = ((markdown_path, markdown), (json_path, serialized))
    for path, content in artifacts:
        if path.is_symlink() or (path.exists() and path.read_bytes() != content.encode("utf-8")):
            raise FileExistsError(
                f"research run artifact already exists with different bytes: {path}"
            )
    for path, content in artifacts:
        publish_text_once(path, content)
