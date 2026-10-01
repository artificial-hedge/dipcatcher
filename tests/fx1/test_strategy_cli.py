"""CLI end-to-end smoke uses a labeled injected model, never a real-model result."""

import json
from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from fx1.cli import app

runner = CliRunner()
CODE = "def strategy(history): return {'probability_up':0.6,'target_weight':0.25}"


def test_generate_and_replay_cli_immutable_and_unjudged(tmp_path, monkeypatch):
    import fx1.serve

    calls = []

    class Model:
        def complete(self, messages):
            calls.append(messages)
            return CODE

    monkeypatch.setattr(fx1.serve, "get_backend", lambda *_args, **_kwargs: Model())
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {"id": "x", "title": "x", "description": "constant probability", "parameters": {}}
        )
    )
    code = tmp_path / "generated.json"
    command = ["strategy", "generate", "--spec", str(spec), "--out", str(code)]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.exception
    assert len(calls) == 1
    assert json.loads(code.read_text())["code_text"] == CODE
    assert runner.invoke(app, command).exit_code != 0
    assert len(calls) == 1  # overwrite refused before inference

    start = datetime(2020, 1, 1, tzinfo=UTC)
    times = [(start + timedelta(days=i)).isoformat() for i in range(4)]
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(
            {
                "bars": [
                    {"event_time": t, "available_time": t, "close": 100 + i}
                    for i, t in enumerate(times)
                ],
                "decision_times": times[:-1],
                "data_source": "SYNTHETIC_CLI_fixture",
                "synthetic": True,
            }
        )
    )
    receipt = tmp_path / "receipt.json"
    replay = [
        "strategy",
        "replay",
        "--spec",
        str(spec),
        "--generated",
        str(code),
        "--data",
        str(data),
        "--out",
        str(receipt),
        "--cost-bps",
        "2",
    ]
    report = runner.invoke(app, replay)
    assert report.exit_code == 0, report.exception
    summary = json.loads(report.stdout)
    assert summary["specification_judge_passed"] is False
    assert summary["failures"] == ["specification_judge_missing"]
    assert summary["synthetic"] and not summary["live_pnl_claim"]
    assert json.loads(receipt.read_text())["replay"]["transaction_cost_bps"] == 0.5
    assert runner.invoke(app, replay).exit_code != 0


def test_generation_error_creates_no_success_output(tmp_path, monkeypatch):
    import fx1.serve

    class Model:
        def complete(self, _messages):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(fx1.serve, "get_backend", lambda *_args, **_kwargs: Model())
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"id": "x", "title": "x", "description": "x", "parameters": {}}))
    out = tmp_path / "code.json"
    result = runner.invoke(app, ["strategy", "generate", "--spec", str(spec), "--out", str(out)])
    assert result.exit_code != 0
    assert not out.exists()
