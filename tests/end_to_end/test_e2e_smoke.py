"""Deterministic SYNTHETIC ingest-to-report smoke."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pytest

from quant_fund.pipeline.e2e_smoke import (
    E2ESmokeError,
    main,
    require_finite,
    require_rows,
    run_e2e_smoke,
)
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent

pytestmark = pytest.mark.e2e_smoke


def test_require_finite_rejects_nan_inf_and_empty_rows() -> None:
    with pytest.raises(E2ESmokeError, match="non-finite"):
        require_finite("demo", {"x": float("nan")})
    with pytest.raises(E2ESmokeError, match="non-finite"):
        require_finite("demo", [float("inf")])
    with pytest.raises(E2ESmokeError, match="empty"):
        require_rows("demo", pl.DataFrame({"close": []}))


def test_smoke_refuses_non_synthetic_hosted_and_oversized(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  source: file\n  root: data\n")
    with pytest.raises(E2ESmokeError, match="non-synthetic"):
        run_e2e_smoke(tmp_path / "lake", config_path=bad)
    hosted = tmp_path / "hosted.yaml"
    hosted.write_text(
        "data:\n  source: synthetic\n  synthetic_n_assets: 2\n  synthetic_n_days: 16\n"
        "robinhood_plus:\n  enabled: true\n"
    )
    with pytest.raises(E2ESmokeError, match="robinhood_plus"):
        run_e2e_smoke(tmp_path / "lake", config_path=hosted)
    wide = tmp_path / "wide.yaml"
    wide.write_text(
        "data:\n  source: synthetic\n  synthetic_n_assets: 13\n  synthetic_n_days: 16\n"
        "robinhood_plus:\n  enabled: false\n"
    )
    with pytest.raises(E2ESmokeError, match="12 assets"):
        run_e2e_smoke(tmp_path / "lake", config_path=wide)


def test_main_rejects_a_non_synthetic_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  source: file\n  root: data\n")
    code = main(["--root", str(tmp_path / "lake"), "--config", str(bad)])
    assert code == 1
    assert "non-synthetic" in capsys.readouterr().err


@pytest.mark.synthetic
def test_two_runs_share_a_digest(tmp_path: Path) -> None:
    started = time.perf_counter()
    first = run_e2e_smoke(tmp_path / "a")
    second = run_e2e_smoke(tmp_path / "b")
    elapsed = time.perf_counter() - started
    assert elapsed < 300.0, f"e2e smoke took {elapsed:.1f}s"
    assert first == second
    assert len(first["digest"]) == 64
    assert first["data_label"] == "SYNTHETIC"
    assert first["live_pnl_claim"] is False
    assert family_blob_forbidden_metrics_absent(first)
    assert first["stages"]["backtest"]["listing_exits"] >= 1
    assert first["stages"]["backtest"]["fill_rows"] > 0
    assert first["stages"]["forecast"]["n"] > 0
    assert first["stages"]["risk"]["pretrade"] == "accepted"
    report = (tmp_path / "a" / "metadata" / "reports" / "e2e_smoke.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in report
    tear = (tmp_path / "a" / "metadata" / "reports" / "e2e_tearsheet.md").read_text(
        encoding="utf-8"
    )
    assert "SYNTHETIC" in tear
