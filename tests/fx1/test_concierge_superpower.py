"""Qualified /superpower wiring: catalog, evidence, and cancellation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fx1.harness import Harness
from fx1.interactive.concierge import Concierge
from tests.fx1.test_concierge import FakeHarnessResult, FakeResearcher, make_concierge


class _CatalogHarness:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def run(
        self, name: str, extra_args: list[str] | None = None, **_kwargs: object
    ) -> FakeHarnessResult:
        self.calls.append(name)
        if name == "capabilities" and self.fail:
            return FakeHarnessResult(1, "", "catalog down")
        return FakeHarnessResult(0, "DATA_LABEL=SYNTHETIC\nok\n", "")


def _joined(lines: list[str]) -> str:
    return "\n".join(lines)


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return tmp_path


def test_unavailable_catalog_fails_closed(env: Path, tmp_path: Path) -> None:
    del env
    harness = _CatalogHarness(fail=True)
    concierge, out = make_concierge(tmp_path, harness=harness)
    concierge.handle_line("/superpower score volatility with qlike --sync")
    text = _joined(out)
    assert "catalog unavailable" in text
    assert "capabilities" in harness.calls
    assert "vol-bench" not in harness.calls
    assert "research" not in harness.calls
    assert concierge.last_status == "ok"
    assert concierge.last_result is not None
    assert concierge.last_result.plan.catalog_unavailable


def test_cancel_reports_pending_until_the_worker_stops(env: Path, tmp_path: Path) -> None:
    del env
    researcher = FakeResearcher(block=True)
    concierge, out = make_concierge(
        tmp_path, researcher=researcher, harness=_CatalogHarness(fail=True)
    )
    concierge.handle_line("/superpower latest public news on rates")
    assert concierge.job_running
    concierge.handle_line("/cancel")
    text = _joined(out)
    assert "cancel requested" in text
    assert "cancellation pending" in text
    concierge.join_job(timeout_s=5)
    assert concierge.job_running is False


def test_evidence_lists_flash_and_shows_one_item(env: Path, tmp_path: Path) -> None:
    del env
    concierge, out = make_concierge(tmp_path, harness=_CatalogHarness(fail=True))
    concierge.handle_line("/superpower remember a conformal note --sync")
    concierge.handle_line("/evidence")
    text = _joined(out)
    assert "flash entry is not a sealed receipt" in text
    assert "E1" in text
    out.clear()
    concierge.handle_line("/evidence E1")
    detail = _joined(out)
    assert "E1" in detail
    assert "flash" in detail.casefold()


def test_real_vol_bench_verifies_a_receipt_outside_the_repo(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del env
    monkeypatch.chdir(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy(Path("/workspace/configs/base.yaml"), configs / "base.yaml")
    shutil.copy(Path("/workspace/configs/research.yaml"), configs / "research.yaml")
    (tmp_path / "data").mkdir()
    repo_receipts = Path("/workspace/receipts")
    before = {path.name for path in repo_receipts.iterdir()} if repo_receipts.is_dir() else set()
    out: list[str] = []

    class _SmallBench(Harness):
        """Keep the real command, but the catalog default grid is too large for this test."""

        def run(self, name: str, extra_args: list[str] | None = None, **kwargs: object) -> object:
            args = list(extra_args or [])
            if name == "vol-bench":
                args.extend(["--models", "rv_roll", "--n-origins", "10", "--horizons", "1"])
            return super().run(name, args, **kwargs)

    concierge = Concierge(output_fn=out.append, animated=False, harness=_SmallBench())
    concierge.handle_line(
        "/superpower score a volatility forecast with QLIKE on synthetic shards --sync"
    )
    text = _joined(out)
    assert "vol-bench" in text
    assert "VERIFIED" in text
    assert "SYNTHETIC" in text
    assert "evidence status:" in text
    assert concierge.last_result is not None
    items = concierge.last_result.evidence.get("items") or []
    assert any(
        item.get("verdict") == "VERIFIED" and item.get("data_label") == "SYNTHETIC"
        for item in items
    )
    run_dir = tmp_path / "cfg" / "runs"
    assert any(path.is_dir() for path in run_dir.iterdir())
    after = {path.name for path in repo_receipts.iterdir()} if repo_receipts.is_dir() else set()
    assert after == before
