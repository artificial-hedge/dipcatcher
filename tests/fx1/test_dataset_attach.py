"""``/data attach`` keeps a CSV or Parquet inside the session data directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.interactive.datasets import attach_dataset, restore_datasets
from fx1.interactive.sessions import SessionStore
from tests.fx1.test_concierge import joined, make_concierge


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    return tmp_path


def _csv(root: Path, name: str = "variance.csv") -> Path:
    path = root / "data" / name
    path.write_text("timestamp,variance\n2020-01-01T00:00:00+00:00,0.2\n", encoding="utf-8")
    return path


def test_attach_accepts_a_file_under_data_and_rejects_escapes(env: Path) -> None:
    csv_path = _csv(env)
    via_root = attach_dataset("variance.csv", cwd=env)
    via_prefix = attach_dataset("data/variance.csv", cwd=env)
    assert via_root.path == str(csv_path.resolve())
    assert via_prefix.path == via_root.path
    assert via_root.kind == "csv"
    assert via_root.label == "file"
    with pytest.raises(ValueError, match="stay inside data"):
        attach_dataset("../outside.csv", cwd=env)
    with pytest.raises(ValueError, match="not found"):
        attach_dataset("missing.csv", cwd=env)
    with pytest.raises(ValueError, match="csv or .parquet"):
        attach_dataset("notes.txt", cwd=env)


def test_attach_rejects_a_symlink_that_leaves_data(env: Path) -> None:
    outside = env / "outside.csv"
    outside.write_text("timestamp,variance\n", encoding="utf-8")
    link = env / "data" / "link.csv"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="stay inside data"):
        attach_dataset("link.csv", cwd=env)


def test_restore_drops_a_path_outside_data(env: Path) -> None:
    csv_path = _csv(env)
    rows = [
        {"path": str(csv_path.resolve()), "kind": "csv", "label": "file"},
        {"path": str((env / "outside.csv").resolve()), "kind": "csv", "label": "file"},
        {"path": str(csv_path.resolve()), "kind": "exe", "label": "file"},
    ]
    restored = restore_datasets(rows, cwd=env)
    assert [item.path for item in restored] == [str(csv_path.resolve())]


def test_concierge_attach_persists_and_resumes(env: Path) -> None:
    csv_path = _csv(env)
    store = SessionStore(env / "sessions")
    concierge, out = make_concierge(env)
    concierge._session_store = store  # noqa: SLF001
    concierge.handle_line("/data")
    assert "no datasets attached" in joined(out)
    concierge.handle_line("/data attach variance.csv")
    assert f"attached dataset {csv_path.resolve()}" in joined(out)
    record = concierge._session  # noqa: SLF001
    assert record is not None
    assert record["datasets"] == [{"path": str(csv_path.resolve()), "kind": "csv", "label": "file"}]
    concierge.handle_line("/data attach variance.csv")
    assert len(concierge._datasets) == 1  # noqa: SLF001
    other, out_again = make_concierge(env)
    other._session_store = store  # noqa: SLF001
    other.handle_line(f"/resume {record['id'][:8]}")
    assert "resumed session" in joined(out_again)
    assert [item.path for item in other._datasets] == [str(csv_path.resolve())]  # noqa: SLF001
