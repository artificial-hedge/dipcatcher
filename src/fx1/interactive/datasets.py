"""Operator-attached datasets for ``/data``.

A path must be a CSV or Parquet file inside ``<cwd>/data``. The command does
not invent a file, a column, or a data label.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from fx1.interactive.planner import DatasetRef
from fx1.interactive.policy import PolicyViolation, contain_path

_KINDS: dict[str, str] = {".csv": "csv", ".parquet": "parquet"}
_LABEL = "file"


def input_dir(cwd: Path) -> Path:
    """The only directory ``/data attach`` will read."""

    return cwd / "data"


def attach_dataset(raw: str, *, cwd: Path) -> DatasetRef:
    """Resolve one CSV or Parquet file under :func:`input_dir`."""

    text = _clean_request(raw)
    resolved = _resolve_under_data(text, cwd)
    if not resolved.is_file():
        raise ValueError("dataset file was not found under data/")
    return DatasetRef(str(resolved), _KINDS[resolved.suffix.lower()], _LABEL)


def format_datasets(rows: Sequence[DatasetRef]) -> str:
    """Presence-only list. An empty session says so."""

    if not rows:
        return "no datasets attached"
    return "\n".join(f"{item.kind}  {item.path}" for item in rows)


def dataset_rows(rows: Sequence[DatasetRef]) -> list[dict[str, str]]:
    """JSON-safe rows stored on the session record."""

    return [{"path": item.path, "kind": item.kind, "label": item.label} for item in rows]


def restore_datasets(raw: object, *, cwd: Path) -> tuple[DatasetRef, ...]:
    """Drop rows that are not files inside the data directory."""

    if not isinstance(raw, list):
        return ()
    kept = [_one_restored(item, cwd) for item in raw]
    return tuple(item for item in kept if item is not None)


def _clean_request(raw: str) -> str:
    text = raw.strip()
    if not text or len(text) > 512:
        raise ValueError("usage: /data attach <csv-or-parquet>")
    if Path(text).suffix.lower() not in _KINDS:
        raise ValueError("dataset must be a .csv or .parquet file")
    return text


def _resolve_under_data(text: str, cwd: Path) -> Path:
    root = input_dir(cwd)
    found = _contained(text, cwd, root)
    if found is not None:
        return found
    # ``data/name.csv`` is also tried relative to ``data/``. Do not do that
    # when the cwd-relative path already exists: a symlink that leaves data/
    # would otherwise attach ``data/data/name.csv`` instead.
    if _named_candidate_exists(text, cwd):
        raise ValueError("dataset path must stay inside data/")
    found = _contained(text, root, root)
    if found is None:
        raise ValueError("dataset path must stay inside data/")
    return found


def _named_candidate_exists(text: str, cwd: Path) -> bool:
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        return candidate.is_symlink() or candidate.exists()
    except OSError:
        return False


def _contained(text: str, base: Path, data_root: Path) -> Path | None:
    try:
        found = contain_path(text, [base], must_exist=False)
    except PolicyViolation:
        return None
    if not _inside(found, data_root):
        return None
    return found


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _one_restored(item: object, cwd: Path) -> DatasetRef | None:
    parsed = _row(item)
    if parsed is None:
        return None
    path, kind, label = parsed
    resolved = _contained(path, cwd, input_dir(cwd))
    if resolved is None:
        resolved = _contained(path, input_dir(cwd), input_dir(cwd))
    if resolved is None or resolved.suffix.lower() not in _KINDS:
        return None
    return DatasetRef(str(resolved), kind, label)


def _row(item: object) -> tuple[str, str, str] | None:
    if not isinstance(item, dict):
        return None
    path = item.get("path")
    kind = item.get("kind")
    label = item.get("label")
    if not isinstance(path, str) or not path:
        return None
    if kind not in {"csv", "parquet"} or not isinstance(label, str) or not label.strip():
        return None
    return path, kind, label
