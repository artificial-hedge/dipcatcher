"""In-process chat journey: attach a CSV and score it with the real lab."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fx1.harness import Harness
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import Concierge
from fx1.interactive.evidence_ledger import Verdict
from fx1.interactive.sessions import SessionStore
from tests.fx1.stub_model_server import StubModelServer

_KEY = "sk-test-fx1-secret-value-aaa111"
_GOAL = "Check whether my volatility forecasts are calibrated using QLIKE --sync"
_FORBIDDEN = ("sharpe", "sortino", "calmar", "p&l", " pnl", "nav")


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def _vol_csv(path: Path, rows: int = 40) -> None:
    lines = ["timestamp,realized,forecast_var"]
    for index in range(rows):
        day = index // 24 + 1
        hour = index % 24
        lines.append(f"2020-01-{day:02d}T{hour:02d}:00:00+00:00,1.0,1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_attached_csv_scores_and_verifies(tmp_path: Path) -> None:
    csv_path = tmp_path / "vol.csv"
    _vol_csv(csv_path)
    out: list[str] = []
    with StubModelServer({"fx1": _KEY, "fx1-lite": _KEY}) as server:
        from fx1.interactive import profiles

        profiles.set_endpoint("fx1", _KEY, server.url)
        concierge = Concierge(
            harness=Harness(),
            approvals=ApprovalGate(),
            session_store=SessionStore(tmp_path / "sessions"),
            output_fn=out.append,
            animated=False,
            input_fn=lambda _prompt: "n",
        )
        assert concierge.handle_line(f"/data attach {csv_path.name}") is False
        assert concierge.handle_line(f"/superpower {_GOAL}") is False
        scored = "\n".join(out)
        out.clear()
        concierge.handle_line("/evidence")
        evidence = "\n".join(out)
        out.clear()
        concierge.handle_line("/runs")
        runs = "\n".join(out)
    text = "\n".join((scored, evidence, runs))
    assert _KEY not in text
    lowered = scored.casefold()
    for token in _FORBIDDEN:
        assert token not in lowered
    assert "point-in-time validity" in lowered
    assert "not verified" in lowered
    result = concierge.last_result
    assert result is not None
    steps = [step for step in result.plan.typed_steps if step.capability_id == "score-forecasts"]
    assert steps
    assert Path(str(steps[0].args["--data"])) == csv_path.resolve()
    assert steps[0].args["--task"] == "volatility"
    assert "--out-dir" in steps[0].args
    ran = [run for run in result.runs if run.id == "score-forecasts"]
    assert ran and ran[0].status == "ran", ran[0].detail if ran else text[-1500:]
    jobs = concierge._job_ledger().list()  # noqa: SLF001
    assert jobs
    argv = jobs[0].steps[0].argv_summary
    assert "score-forecasts" in argv
    assert str(csv_path.resolve()) in argv
    receipts = list((tmp_path / "cfg").rglob("forecast_scoring_*.json"))
    assert receipts
    body = json.loads(receipts[0].read_text(encoding="utf-8"))
    blob = json.dumps(body)
    assert "forecast_scoring.v1" in blob
    verified = Harness().run("verify-receipt", [str(receipts[0])])
    assert verified.exit_code == 0, verified.stderr[-1500:]
    items = concierge._evidence.items()  # noqa: SLF001
    verified_items = [item for item in items if item.verdict is Verdict.VERIFIED]
    assert len(verified_items) == 1
    assert verified_items[0].data_label.casefold() == "file"
    assert "VERIFIED" in evidence
    assert "FILE" in evidence
    assert re.search(r"j-\d{8}T", runs)
    assert "succeeded" in runs
