"""Approval scope, path escape, and non-interactive fail-closed execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.interactive.approvals import ApprovalDecision, ApprovalGate, ApprovalRequest
from fx1.interactive.headless import ExecRequest, run_exec
from tests.fx1.redteam_support import (
    assert_canary,
    completion,
    isolate,
    joined,
    open_chat,
    plant_canary,
    tool,
    visible,
)
from tests.fx1.test_concierge import FakeHarnessResult


@pytest.fixture()
def root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    isolate(monkeypatch, tmp_path)
    return tmp_path


def test_session_approval_does_not_cover_a_broader_argv(root: Path) -> None:
    prompts: list[str] = []

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        prompts.append(request.key)
        if len(prompts) == 1:
            return ApprovalDecision.SESSION
        return ApprovalDecision.DENY

    script = [
        completion(None, (tool("t1", "harness_run", {"name": "train", "args": []}),)),
        completion("first", None),
        completion(None, (tool("t2", "harness_run", {"name": "train", "args": ["--wider"]}),)),
        completion("second", None),
    ]
    concierge, out, harness = open_chat(root, script, approvals=ApprovalGate(decide_fn=decide))
    concierge.handle_line("train narrow")
    concierge.handle_line("train wider")
    assert prompts == ["harness:train", "harness:train"]
    assert harness.calls == [("train", [])]
    assert "not approved" in joined(out)
    assert harness.canary.read_text(encoding="utf-8") == "executed"


def test_identical_argv_reuses_the_session_grant(root: Path) -> None:
    prompts: list[str] = []

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        prompts.append(request.key)
        return ApprovalDecision.SESSION

    script = [
        completion(None, (tool("t1", "harness_run", {"name": "train", "args": []}),)),
        completion("first", None),
        completion(None, (tool("t2", "harness_run", {"name": "train", "args": []}),)),
        completion("second", None),
    ]
    concierge, _out, harness = open_chat(root, script, approvals=ApprovalGate(decide_fn=decide))
    concierge.handle_line("one")
    concierge.handle_line("two")
    assert prompts == ["harness:train"]
    assert harness.calls == [("train", []), ("train", [])]


def test_a_model_supplied_approval_key_does_not_authorize_another_command(root: Path) -> None:
    def decide(request: ApprovalRequest) -> ApprovalDecision:
        if request.key == "harness:train":
            return ApprovalDecision.SESSION
        return ApprovalDecision.DENY

    raw = '{"name": "optimize", "args": [], "approval_key": "harness:train"}'
    script = [
        completion(None, (tool("t", "harness_run", {"name": "train", "args": []}),)),
        completion("trained", None),
        completion(None, (tool("o", "harness_run", raw),)),
        completion("no", None),
    ]
    concierge, out, harness = open_chat(root, script, approvals=ApprovalGate(decide_fn=decide))
    concierge.handle_line("train")
    concierge.handle_line("optimize")
    assert [name for name, _args in harness.calls] == ["train"]
    assert "not approved" in joined(out)


def test_exec_without_approve_returns_and_does_not_run(root: Path) -> None:
    del root

    class Harness:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(self, name: str, extra_args: list[str] | None = None, **_kwargs: object) -> object:
            del extra_args
            self.calls.append(name)
            return FakeHarnessResult(0, "synthetic harness marker\n")

    denied = Harness()
    blocked = run_exec(
        ExecRequest(prompt="/harness run train", timeout_s=15, session="none"),
        harness=denied,
        install_signals=False,
    )
    assert denied.calls == []
    assert blocked.exit_code != 0

    allowed = Harness()
    ran = run_exec(
        ExecRequest(
            prompt="/harness run train",
            timeout_s=15,
            approve_keys=("harness:train",),
            session="none",
        ),
        harness=allowed,
        install_signals=False,
    )
    assert allowed.calls == ["train"]
    assert ran.exit_code == 0

    other = Harness()
    refused = run_exec(
        ExecRequest(
            prompt="/harness run optimize",
            timeout_s=15,
            approve_keys=("harness:train",),
            session="none",
        ),
        harness=other,
        install_signals=False,
    )
    assert other.calls == []
    assert refused.exit_code != 0


def test_path_escapes_and_symlink_targets_do_not_run(root: Path) -> None:
    outside = root / "outside"
    outside.mkdir()
    secret = outside / "secret.csv"
    secret.write_text("a,b\n", encoding="utf-8")
    receipts = root / "receipts"
    receipts.mkdir()
    link = receipts / "escape"
    link.symlink_to(outside)
    canary = plant_canary(root)
    script = [
        completion(
            None, (tool("a", "harness_run", {"name": "doctor", "args": ["--out-dir", "/"]}),)
        ),
        completion(
            None,
            (tool("b", "harness_run", {"name": "doctor", "args": ["--out-dir=/etc"]}),),
        ),
        completion(
            None,
            (tool("c", "harness_run", {"name": "doctor", "args": ["--out", "../outside"]}),),
        ),
        completion(
            None,
            (tool("d", "harness_run", {"name": "doctor", "args": ["--out-dir", str(link)]}),),
        ),
        completion(
            None,
            (tool("e", "harness_run", {"name": "doctor", "args": ["--verifier", "off"]}),),
        ),
        completion(
            None,
            (tool("f", "harness_run", {"name": "doctor", "args": ["--no-verify"]}),),
        ),
        completion("done", None),
    ]
    concierge, out, harness = open_chat(root, script)
    concierge.handle_line("escape")
    blob = visible(concierge, out)
    assert harness.calls == []
    assert blob.count("path_escape") >= 4
    assert "forbidden_flag" in blob
    assert_canary(canary)
    assert secret.read_text(encoding="utf-8") == "a,b\n"
    assert "--out-dir" not in blob
    assert "/etc" not in blob
