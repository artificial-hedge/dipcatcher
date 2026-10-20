"""Path refusals, profile parsing, and stale detection for ``/data``."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fx1.interactive.contracts import _inside_root
from fx1.interactive.data_ux import (
    DATA_MAX_BYTES,
    Attachment,
    dataset_refs,
    handle_data,
    model_context,
    rehash_rows,
)
from fx1.interactive.superpower_glue import qualified_roots

_SECRET = "sk-secret-column-value"


class ScriptHarness:
    def __init__(self, stdout: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.exit_code = exit_code
        self.calls: list[tuple[str, list[str]]] = []

    def run(
        self, name: str, extra_args: list[str] | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del kwargs
        self.calls.append((name, list(extra_args or [])))
        return SimpleNamespace(exit_code=self.exit_code, stdout=self.stdout, stderr="")


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.chdir(tmp_path)


def _csv(path: Path, text: str | None = None) -> Path:
    body = text or "timestamp,realized\n2020-01-01T00:00:00+00:00,1.0\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _inside(tmp_path: Path, name: str, text: str | None = None) -> Path:
    return _csv(tmp_path / "data" / name, text)


def _profile(path: Path, **overrides: object) -> str:
    payload: dict[str, object] = {
        "schema": "dataset_profile.v1",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "n_rows": 2,
        "columns": [
            {"name": "timestamp", "dtype": "String", "nulls": 0},
            {"name": "realized", "dtype": "Float64", "nulls": 1},
        ],
        "time": {
            "column": None,
            "start": None,
            "end": None,
            "parsed": False,
            "strictly_increasing": False,
        },
        "suggested_roles": {"available_time": {"column": None}},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_path_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = ScriptHarness()
    rows, text = handle_data("", [], harness)
    assert rows == []
    assert text == "no datasets attached"
    _rows_none, text = handle_data("list", [], harness)
    assert text == "no datasets attached"
    missing = tmp_path / "data" / "nope.csv"
    rows, text = handle_data(f"attach {missing}", [], harness)
    assert rows == []
    assert text == "dataset file was not found under data/"
    outside = _csv(tmp_path / "loose.csv")
    _rows_none, text = handle_data(f"attach {outside}", [], harness)
    assert "stay inside data" in text
    notes = _csv(tmp_path / "notes.txt", "nope")
    _rows_none, text = handle_data(f"attach {notes}", [], harness)
    assert "csv or .parquet" in text
    directory = tmp_path / "data" / "folder.csv"
    directory.mkdir(parents=True)
    _rows_none, text = handle_data(f"attach {directory}", [], harness)
    assert text == "refusing a directory"
    huge = tmp_path / "big.csv"
    with huge.open("wb") as handle:
        handle.truncate(DATA_MAX_BYTES + 1)
    _rows_none, text = handle_data(f"attach {huge}", [], harness)
    assert "exceeds" in text
    assert harness.calls == []
    fifo = tmp_path / "pipe.csv"
    os.mkfifo(fifo)
    _rows_none, text = handle_data(f"attach {fifo}", [], harness)
    assert text == "refusing a FIFO"
    _rows_none, text = handle_data("attach /dev/null", [], harness)
    assert text == "refusing a non-regular file"
    _rows_none, text = handle_data("attach bad\x00name.csv", [], harness)
    assert text == "path contains NUL"
    target = _csv(tmp_path / "notes-target.txt", "x")
    link = tmp_path / "alias.csv"
    link.symlink_to(target)
    _rows_none, text = handle_data(f"attach {link}", [], harness)
    assert "stay inside data" in text
    blocked = _inside(tmp_path, "blocked.csv")
    blocked.chmod(0)
    _rows_none, text = handle_data(f"attach {blocked}", [], harness)
    blocked.chmod(0o644)
    assert text == "dataset is not readable"
    locked = _inside(tmp_path, "locked.csv")
    real_access = os.access

    def denied(path: object, mode: int, *, follow_symlinks: bool = True) -> bool:
        if str(path).endswith("locked.csv"):
            return False
        return real_access(path, mode, follow_symlinks=follow_symlinks)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "access", denied)
    _rows_none, text = handle_data(f"attach {locked}", [], harness)
    assert text == "dataset is not readable"
    with pytest.raises(ValueError, match="usage: /data"):
        handle_data("attach", [], harness)


def test_in_data_symlink_resolves_and_escapes_are_refused(tmp_path: Path) -> None:
    real = _inside(tmp_path, "real.csv")
    inside = tmp_path / "data" / "link.csv"
    inside.symlink_to(real)
    harness = ScriptHarness(_profile(real))
    rows, text = handle_data("attach link.csv", [], harness)
    assert rows
    assert rows[0].path == str(real.resolve())
    assert text.startswith(f"attached dataset {real.resolve()}")
    outside = _csv(tmp_path / "outside.csv")
    escape = tmp_path / "data" / "escape.csv"
    escape.symlink_to(outside)
    kept, message = handle_data("attach escape.csv", rows, harness)
    assert kept == rows
    assert "stay inside data" in message
    dangling = tmp_path / "data" / "gone.csv"
    dangling.symlink_to(tmp_path / "data" / "missing.csv")
    kept, message = handle_data("attach gone.csv", rows, harness)
    assert kept == rows
    assert "not found" in message
    hostile = tmp_path / "data" / "evil.csv"
    hostile.symlink_to(tmp_path / "bad\x1b[31m.csv")
    kept, message = handle_data("attach evil.csv", rows, harness)
    assert kept == rows
    assert "\x1b" not in message
    assert "stay inside data" in message


def test_profile_warnings_hostile_names_and_malformed(tmp_path: Path) -> None:
    path = _inside(tmp_path, "panel.csv")
    hostile = "\x1b[31mred\x1b[0m"
    osc = "osc\x1b]0;owned\x07"
    hidden_key = "sk-\u200b" + ("b" * 24)
    clipboard = "a\x1b]52;c;CLIPBOARD\x07"
    harness = ScriptHarness(
        _profile(
            path,
            columns=[
                {"name": hostile, "dtype": "String", "nulls": 0},
                {"name": osc, "dtype": "String", "nulls": 2},
                {"name": _SECRET, "dtype": "String", "nulls": 0},
                {"name": hidden_key, "dtype": "String", "nulls": 0},
                {"name": clipboard, "dtype": "String", "nulls": 0},
            ],
        )
    )
    rows, text = handle_data(f"attach {path}", [], harness)
    assert rows
    assert text.startswith(f"attached dataset {path.resolve()}")
    assert "profile rows=2 columns=" in text
    assert "warning: no time column" in text
    assert "warning: columns contain nulls" in text
    assert "point-in-time validity will NOT be verified" in text
    assert "\x1b" not in text
    assert _SECRET not in text
    assert "CLIPBOARD" not in text
    assert "\u200b" not in text
    assert "\x1b" not in "".join(rows[0].columns)
    context = model_context(rows)
    assert "UNTRUSTED" in context
    assert rows[0].path not in context
    assert _SECRET not in context
    assert "CLIPBOARD" not in context
    assert "\u200b" not in context
    bad = ScriptHarness("not json")
    attached, message = handle_data(f"attach {path}", [], bad)
    assert attached
    assert attached[0].sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert attached[0].rows is None
    assert message.startswith(f"attached dataset {path.resolve()}")
    assert "dataset profile unavailable; attached without a profile" in message
    wrong = ScriptHarness(_profile(path, schema="nope"))
    attached, message = handle_data(f"attach {path}", [], wrong)
    assert attached
    assert "dataset profile unavailable; attached without a profile" in message
    array = ScriptHarness("[1, 2, 3]")
    attached, message = handle_data(f"attach {path}", [], array)
    assert attached
    assert "dataset profile unavailable; attached without a profile" in message


def test_monotonic_warning_and_eight_dataset_cap(tmp_path: Path) -> None:
    path = _inside(tmp_path, "times.csv")
    harness = ScriptHarness(
        _profile(
            path,
            columns=[{"name": "timestamp", "dtype": "String", "nulls": 0}],
            time={
                "column": "timestamp",
                "parsed": True,
                "strictly_increasing": False,
                "start": "a",
                "end": "b",
            },
        )
    )
    _rows, text = handle_data(f"attach {path}", [], harness)
    assert "warning: time is not monotonic" in text
    assert "no time column" not in text
    rows: list[Attachment] = []
    for index in range(8):
        item = _inside(tmp_path, f"f{index}.csv")
        harness.stdout = _profile(item)
        rows, text = handle_data(f"attach {item}", rows, harness)
        assert "profile rows=" in text
    assert len(rows) == 8
    extra = _inside(tmp_path, "ninth.csv")
    harness.stdout = _profile(extra)
    kept, text = handle_data(f"attach {extra}", rows, harness)
    assert kept == rows
    assert "dataset limit is 8" in text
    removed, text = handle_data("remove 1", rows, harness)
    assert len(removed) == 7
    assert text.startswith("removed ")
    _unknown, text = handle_data("remove 9", removed, harness)
    assert text == "unknown dataset index"
    _unknown, text = handle_data("inspect 0", removed, harness)
    assert text == "unknown dataset index"
    listed, text = handle_data("list", removed, harness)
    assert listed == removed
    assert text.startswith("1 ")


def test_stale_digest_and_file_root(tmp_path: Path) -> None:
    path = _inside(tmp_path, "nested/panel.csv")
    harness = ScriptHarness(_profile(path))
    rows, _text = handle_data(f"attach {path}", [], harness)
    original = rows[0].sha256
    path.write_text(
        path.read_text(encoding="utf-8") + "2020-01-02T00:00:00+00:00,2.0\n", encoding="utf-8"
    )
    harness.stdout = _profile(path)
    inspected, text = handle_data("inspect 1", rows, harness)
    assert "file changed since attach" in text
    assert inspected[0].stale is True
    assert inspected[0].sha256 == original
    updated, reason = rehash_rows(inspected)
    assert updated[0].stale is True
    assert reason is not None
    assert "dataset changed since attach" in reason
    assert "/data inspect 1" in reason
    sibling = _inside(tmp_path, "other.csv")
    roots = qualified_roots(extra_inputs=(path.resolve(),))
    assert path.resolve() in roots["input"]
    assert path.parent not in roots["input"]
    assert _inside_root(path.resolve(), path.resolve()) == path.resolve()
    assert _inside_root(sibling.resolve(), path.resolve()) is None


def test_real_inspect_profile(tmp_path: Path) -> None:
    from fx1.harness import Harness

    path = _csv(
        tmp_path / "data" / "vol.csv",
        "timestamp,realized,forecast_var\n"
        "2020-01-01T00:00:00+00:00,1.0,1.1\n"
        "2020-01-01T01:00:00+00:00,1.0,1.1\n",
    )
    rows, text = handle_data(f"attach {path}", [], Harness())
    assert rows
    assert "profile rows=2 columns=timestamp, realized, forecast_var" in text
    assert "point-in-time validity will NOT be verified" in text
    assert rows[0].sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    listed, listing = handle_data("list", rows, Harness())
    assert listed == rows
    assert "vol" in listing


def test_parquet_suffix_and_ansi_filename(tmp_path: Path) -> None:
    parquet = tmp_path / "data" / "panel.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"PAR1")
    harness = ScriptHarness(_profile(parquet))
    rows, text = handle_data(f"attach {parquet}", [], harness)
    assert rows[0].kind == "parquet"
    assert rows[0].label == "file"
    assert "profile rows=" in text
    nasty = tmp_path / "data" / "evil\x1b[31mname.csv"
    _csv(nasty)
    harness.stdout = _profile(nasty)
    named, shown = handle_data(f"attach {nasty}", [], harness)
    assert named == []
    assert "\x1b" not in shown
    assert "stay inside data" in shown


def test_issue_time_binds_only_for_an_exact_name(tmp_path: Path) -> None:
    path = _inside(tmp_path, "issued.csv")
    exact = {
        "available_time": {"column": "timestamp", "confidence": "exact_name"},
    }
    rows, _text = handle_data(
        f"attach {path}", [], ScriptHarness(_profile(path, suggested_roles=exact))
    )
    assert dataset_refs(rows)[0].available_time_col == "timestamp"
    alias = {"available_time": {"column": "timestamp", "confidence": "alias"}}
    rows, _text = handle_data(
        f"attach {path}", [], ScriptHarness(_profile(path, suggested_roles=alias))
    )
    assert dataset_refs(rows)[0].available_time_col == ""
