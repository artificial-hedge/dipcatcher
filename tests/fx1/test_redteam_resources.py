"""Bounds on tool payloads, harness output, slash batches, and dataset files."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fx1.interactive.data_ux import (
    DATA_MAX_BYTES,
    handle_data,
    model_context,
    resolve_dataset,
)
from tests.fx1.redteam_support import (
    SECRET,
    assert_canary,
    completion,
    isolate,
    open_chat,
    tool,
    visible,
)
from tests.fx1.test_concierge import FakeHarnessResult, FakeResearcher


@pytest.fixture()
def root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    isolate(monkeypatch, tmp_path)
    return tmp_path


def test_oversized_and_deep_tool_arguments_are_refused(root: Path) -> None:
    researcher = FakeResearcher()
    huge = json.dumps({"query": "A" * 40_000})
    deep = "[" * 40 + "]" * 40
    script = [
        completion(None, (tool("w", "web_lookup", huge),)),
        completion(None, (tool("h", "harness_run", deep),)),
        completion("stopped", None),
    ]
    concierge, out, harness = open_chat(root, script, researcher=researcher)
    concierge.handle_line("too big")
    blob = visible(concierge, out)
    assert blob.count("too_large") >= 2
    assert researcher.goals == []
    assert harness.calls == []
    assert_canary(harness.canary)


def test_tool_call_budget_stops_a_burst(root: Path) -> None:
    calls = tuple(
        tool(f"c{index}", "harness_run", {"name": "doctor", "args": []}) for index in range(40)
    )
    script = [completion(None, calls), completion("done", None)]
    concierge, _out, harness = open_chat(root, script)
    concierge.handle_line("burst")
    assert len(harness.calls) == 32
    messages = [message["content"] for message in concierge.messages() if message["role"] == "tool"]
    assert messages.count("too_many_calls") == 8


def test_huge_harness_stdout_is_clipped_and_redacted(root: Path) -> None:
    payload = f"prefix {SECRET} \x1b]52;c;CLIPBOARD\x07" + ("x" * 80_000)

    class Loud:
        def run(self, name: str, extra_args: list[str] | None = None, **_kwargs: object) -> object:
            del name, extra_args
            return FakeHarnessResult(0, payload, "")

    script = [
        completion(None, (tool("d", "harness_run", {"name": "doctor", "args": []}),)),
        completion("done", None),
    ]
    concierge, out, _harness = open_chat(root, script, harness=Loud())
    concierge.handle_line("dump")
    shown = "\n".join(out)
    assert SECRET not in shown
    assert "CLIPBOARD" not in shown
    assert "\x1b" not in shown
    assert "truncated:" in shown
    stored = "\n".join(str(message.get("content", "")) for message in concierge.messages())
    assert SECRET not in stored
    assert len(stored) < 20_000


def test_local_slash_batch_does_not_execute_the_canary(root: Path) -> None:
    concierge, _out, harness = open_chat(root, [])
    for _ in range(200):
        assert concierge.handle_line("/help") is False
    assert harness.calls == []
    assert_canary(harness.canary)


def test_fifo_sparse_file_and_dev_zero_symlink_are_refused(root: Path) -> None:
    fifo = root / "pipe.csv"
    os.mkfifo(fifo)
    assert resolve_dataset(str(fifo), cwd=root)[0] is None

    big = root / "sparse.csv"
    with big.open("wb") as handle:
        handle.truncate(10 * 1024**3)
    assert big.stat().st_size > DATA_MAX_BYTES
    refused = resolve_dataset(str(big), cwd=root)
    assert refused[0] is None
    assert refused[1] is not None and "exceeds" in refused[1]

    zero = root / "zero.csv"
    zero.symlink_to("/dev/zero")
    assert resolve_dataset(str(zero), cwd=root)[0] is None


def test_hostile_column_names_are_redacted_in_model_context(root: Path) -> None:
    import hashlib

    path = root / "scores.csv"
    path.write_text("t,y\n1,2\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    hostile = "sk-\u200b" + ("b" * 24)
    payload = {
        "schema": "dataset_profile.v1",
        "sha256": digest,
        "size_bytes": size,
        "n_rows": 1,
        "columns": [
            {"name": hostile, "nulls": 0},
            {"name": "a\x1b]52;c;CLIPBOARD\x07", "nulls": 0},
        ],
        "time": {"column": "", "monotonic": False},
        "suggested_roles": {},
    }

    class Inspect:
        def run(self, name: str, args: list[str], timeout_s: int = 30) -> FakeHarnessResult:
            del name, args, timeout_s
            return FakeHarnessResult(0, json.dumps(payload), "")

    rows, text = handle_data(f"attach {path}", [], Inspect(), cwd=root)
    assert rows
    blob = text + "\n" + model_context(rows)
    assert SECRET not in blob
    assert "CLIPBOARD" not in blob
    assert "\x1b" not in blob
    assert "\u200b" not in blob
