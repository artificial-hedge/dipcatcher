from __future__ import annotations

import os
from pathlib import Path

import pytest

from quant_fund.proof.runner import _atomic_write
from quant_fund.proofcore.contracts import ProofError
from quant_fund.proofcore.scheduler import parse_step


def test_atomic_write_replaces_a_complete_file(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text("old", encoding="utf-8")

    _atomic_write(destination, b"new-complete-content")

    assert destination.read_bytes() == b"new-complete-content"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_write_io_error_preserves_destination_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text("known-good", encoding="utf-8")

    def fail_replace(src: object, dst: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        _atomic_write(destination, b"partial-replacement")

    assert destination.read_text(encoding="utf-8") == "known-good"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("bad", ["", "0d", "-1h", "1.5m", "1w", "1D"])
def test_scheduler_rejects_invalid_steps_before_orchestration(bad: str) -> None:
    with pytest.raises(ProofError, match="step"):
        parse_step(bad)
