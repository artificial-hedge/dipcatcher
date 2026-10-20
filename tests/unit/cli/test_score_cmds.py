"""CLI contract for ``score-forecasts``: stdout lines, inspect, and verify-receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_fx1_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))


def _ts(index: int) -> str:
    return f"2020-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00+00:00"


def _volatility_file(path: Path, n: int = 30) -> None:
    rows = [f"{_ts(i)},1.0,2.0" for i in range(n)]
    path.write_text("timestamp,realized,forecast_var\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.strip()]


def test_help_lists_the_task_and_inspect_flags() -> None:
    """``score-forecasts --help`` documents the contract flags and exits 0."""
    result = runner.invoke(app, ["score-forecasts", "--help"])
    assert result.exit_code == 0
    assert "--task" in result.output
    assert "--data" in result.output
    assert "--inspect" in result.output
    assert "--quantile-cols" in result.output
    assert "--available-time-col" in result.output


def test_success_stdout_puts_the_data_label_first_and_the_receipt_last(tmp_path: Path) -> None:
    """A scored file prints DATA_LABEL, a PIT warning, score lines, then receipt=."""
    data = tmp_path / "v.csv"
    _volatility_file(data)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["score-forecasts", "--task", "volatility", "--data", str(data), "--out-dir", str(out)],
    )
    assert result.exit_code == 0, result.output
    lines = _lines(result.output)
    assert lines[0] == "DATA_LABEL=file"
    assert lines[1].startswith("WARNING:")
    assert "point-in-time validity is NOT verified" in lines[1]
    assert lines[-1].startswith("receipt=")
    assert all(line.startswith("score.") for line in lines[2:-1])
    assert any(line.startswith("score.qlike=") for line in lines)
    receipt = Path(lines[-1].split("=", 1)[1])
    assert receipt.is_file()
    assert receipt.parent == out
    verify = runner.invoke(app, ["verify-receipt", str(receipt)])
    assert verify.exit_code == 0, verify.output
    assert json.loads(verify.output)["valid"] is True


def test_available_time_suppresses_the_warning(tmp_path: Path) -> None:
    """Passing an earlier available_time column omits the PIT warning."""
    rows = [f"{_ts(i)},2019-12-31T00:00:00+00:00,1,1" for i in range(30)]
    data = tmp_path / "pit.csv"
    data.write_text(
        "timestamp,available_time,realized,forecast_var\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "score-forecasts",
            "--task",
            "volatility",
            "--data",
            str(data),
            "--available-time-col",
            "available_time",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "WARNING:" not in result.output
    assert _lines(result.output)[0] == "DATA_LABEL=file"


def test_synthetic_preamble_labels_stdout_synthetic(tmp_path: Path) -> None:
    """A SYNTHETIC preamble is the first stdout label."""
    data = tmp_path / "syn.csv"
    body = "\n".join(f"{_ts(i)},1,1" for i in range(30))
    data.write_text(
        "# DATA_LABEL=SYNTHETIC\ntimestamp,realized,forecast_var\n" + body + "\n", encoding="utf-8"
    )
    result = runner.invoke(
        app,
        [
            "score-forecasts",
            "--task",
            "volatility",
            "--data",
            str(data),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert _lines(result.output)[0] == "DATA_LABEL=SYNTHETIC"


def test_inspect_prints_only_a_profile_and_writes_nothing(tmp_path: Path) -> None:
    """``--inspect`` is read-only JSON and does not create the output directory."""
    data = tmp_path / "v.csv"
    _volatility_file(data, n=4)
    out = tmp_path / "unused"
    result = runner.invoke(
        app,
        [
            "score-forecasts",
            "--task",
            "volatility",
            "--data",
            str(data),
            "--inspect",
            "--out-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    profile = json.loads(result.output)
    assert profile["schema"] == "dataset_profile.v1"
    assert profile["n_rows"] == 4
    assert "DATA_LABEL" not in result.output
    assert "receipt=" not in result.output
    assert "score." not in result.output
    assert not out.exists()


def test_user_data_errors_exit_nonzero_without_a_traceback(tmp_path: Path) -> None:
    """A missing file and a short file fail closed with an actionable message."""
    missing = runner.invoke(
        app,
        ["score-forecasts", "--task", "volatility", "--data", str(tmp_path / "nope.csv")],
    )
    assert missing.exit_code != 0
    combined = missing.output + str(missing.exception or "")
    assert "Traceback" not in combined
    assert "not found" in combined

    short = tmp_path / "short.csv"
    _volatility_file(short, n=5)
    refused = runner.invoke(
        app,
        [
            "score-forecasts",
            "--task",
            "volatility",
            "--data",
            str(short),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert refused.exit_code != 0
    assert "Traceback" not in refused.output + str(refused.exception or "")
    assert "at least 30" in refused.output + str(refused.exception or "")


def test_verify_receipt_rejects_a_tampered_score_payload(tmp_path: Path) -> None:
    """``verify-receipt`` accepts the sealed file and rejects an edited score."""
    data = tmp_path / "v.csv"
    _volatility_file(data)
    out = tmp_path / "out"
    scored = runner.invoke(
        app,
        ["score-forecasts", "--task", "volatility", "--data", str(data), "--out-dir", str(out)],
    )
    assert scored.exit_code == 0, scored.output
    receipt = Path(_lines(scored.output)[-1].split("=", 1)[1])
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["payload"]["scores"]["qlike"] = 99.0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(app, ["verify-receipt", str(tampered)])
    assert result.exit_code != 0
    assert json.loads(result.output)["valid"] is False
