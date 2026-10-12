"""Receipt paths stay intact on their own explain line."""

from __future__ import annotations

from pathlib import Path

from fx1.interactive.orchestration import _stdout_detail
from fx1.interactive.superpower import CapabilityRun, SuperpowerPlan, SuperpowerResult


def test_receipt_path_is_not_clipped_to_160_chars(tmp_path: Path) -> None:
    folder = tmp_path / ("nested_" + "d" * 80)
    folder.mkdir()
    path = folder / f"forecast_scoring_{'ab' * 32}.json"
    path.write_text("{}\n", encoding="utf-8")
    assert len(f"receipt={path}") > 160
    stdout = f"DATA_LABEL=SYNTHETIC\nSYNTHETIC\n{'x' * 400}\nreceipt={path}\n"
    detail = _stdout_detail(stdout, "")
    lines = detail.splitlines()
    assert lines[-1] == f"receipt={path}"
    assert "x" * 161 not in detail
    assert "DATA_LABEL=SYNTHETIC" in lines[0]
    plan = SuperpowerPlan(goal="score")
    explained = SuperpowerResult(
        goal="score",
        plan=plan,
        runs=[CapabilityRun("score-forecasts", "score-forecasts", "ran", detail)],
    ).explain()
    receipt_lines = [line for line in explained.splitlines() if line.startswith("receipt=")]
    assert receipt_lines == [f"receipt={path}"]


def test_non_receipt_stderr_stays_bounded() -> None:
    detail = _stdout_detail("", "y" * 500)
    assert detail == "y" * 160
    assert "receipt=" not in detail
