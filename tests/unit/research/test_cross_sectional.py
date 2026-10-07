"""Cross-sectional rank-IC lane (P3.4): known-answer + contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.research.cross_sectional import (
    CHALLENGERS,
    PANEL_GENERATORS,
    RANKIC_SCHEMA,
    format_rankic_table,
    resolve_panels,
    run_cross_sectional_bench,
    write_rankic_receipt,
)


def _frame(receipt: bool = False):
    frame, receipt_obj = run_cross_sectional_bench(seed=11)
    return (frame, receipt_obj) if receipt else frame


def _cell(frame: pl.DataFrame, shard: str, challenger: str, horizon: int) -> dict:
    row = frame.filter(
        (pl.col("shard") == shard)
        & (pl.col("challenger") == challenger)
        & (pl.col("horizon") == horizon)
    )
    assert row.height == 1
    return row.row(0, named=True)


def test_planted_linear_signal_recovers_high_rank_ic() -> None:
    frame = _frame()
    for horizon in (1, 5, 20):
        row = _cell(frame, "linear_signal", "identity", horizon)
        assert row["status"] == "ok"
        assert row["mean_spearman"] > 0.25
        assert row["t_spearman"] > 5.0
        assert row["n_dates"] >= 70


def test_rank_preserving_cubic_scores_spearman_over_pearson_shape() -> None:
    frame = _frame()
    row = _cell(frame, "monotone_cubic", "identity", 1)
    assert row["status"] == "ok"
    assert row["mean_spearman"] > 0.08


def test_shuffled_challenger_is_a_zero_ic_null() -> None:
    frame = _frame()
    hits = 0
    total = 0
    for row in frame.filter(
        (pl.col("challenger") == "shuffled") & (pl.col("status") == "ok")
    ).iter_rows(named=True):
        total += 1
        assert abs(row["mean_spearman"]) < 0.15
        if abs(row["t_spearman"]) > 1.96:
            hits += 1
    # Across shards x horizons a true null should reject ~5% of cells.
    assert hits <= max(2, int(0.4 * total))


def test_inverted_challenger_flips_the_sign() -> None:
    frame = _frame()
    fwd = _cell(frame, "linear_signal", "identity", 1)
    inv = _cell(frame, "linear_signal", "inverted", 1)
    assert inv["mean_spearman"] == pytest.approx(-fwd["mean_spearman"], abs=1e-12)
    assert inv["t_spearman"] == pytest.approx(-fwd["t_spearman"], abs=1e-12)


def test_pure_noise_panel_stays_null_for_identity() -> None:
    frame = _frame()
    row = _cell(frame, "pure_noise", "identity", 1)
    assert abs(row["mean_spearman"]) < 0.1
    assert abs(row["t_spearman"]) < 2.0


def test_lagged_signal_loses_the_edge() -> None:
    frame = _frame()
    lag = _cell(frame, "linear_signal", "lagged", 1)
    now = _cell(frame, "linear_signal", "identity", 1)
    assert abs(lag["mean_spearman"]) < now["mean_spearman"] * 0.5


def test_regime_flip_has_small_mean_ic_despite_large_absolute_ic() -> None:
    frame = _frame()
    row = _cell(frame, "regime_flip", "identity", 20)
    assert abs(row["mean_spearman"]) < 0.25


def test_bench_is_deterministic() -> None:
    _, receipt_a = run_cross_sectional_bench(seed=42)
    _, receipt_b = run_cross_sectional_bench(seed=42)
    payload_a = {k: v for k, v in receipt_a.items() if k not in ("generated_at", "meta")}
    payload_b = {k: v for k, v in receipt_b.items() if k not in ("generated_at", "meta")}
    assert payload_a == payload_b


def test_bench_fails_closed_on_tiny_panels() -> None:
    with pytest.raises(ValueError, match="at least"):
        run_cross_sectional_bench(n_assets=3)
    with pytest.raises(ValueError, match="at least"):
        run_cross_sectional_bench(n_dates=10)
    with pytest.raises(ValueError, match="horizons"):
        run_cross_sectional_bench(horizons=(0,))


def test_resolve_panels_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown panel"):
        resolve_panels(["does_not_exist"])
    assert set(resolve_panels()) == set(PANEL_GENERATORS)
    assert set(resolve_panels(["pure_noise"])) == {"pure_noise"}


def test_receipt_contract_and_immutability(tmp_path: Path) -> None:
    frame, receipt = run_cross_sectional_bench(seed=5)
    assert receipt["schema"] == RANKIC_SCHEMA
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert receipt["n_rows"] == frame.height
    path = write_rankic_receipt(receipt, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["receipt_sha256"][:16] == path.stem.removeprefix("rankic_eval_")
    # Same payload again -> same file, no error.
    assert write_rankic_receipt(receipt, tmp_path) == path
    # A changed payload hashes to a different filename; the original is untouched.
    tampered = dict(receipt)
    tampered["seed"] = int(receipt["seed"]) + 1
    other = write_rankic_receipt(tampered, tmp_path)
    assert other != path
    assert json.loads(path.read_text())["seed"] == receipt["seed"]


def test_receipt_writer_rejects_dishonest_payloads(tmp_path: Path) -> None:
    _, receipt = run_cross_sectional_bench(seed=5)
    bad = dict(receipt, data_label="REAL")
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_rankic_receipt(bad, tmp_path)
    bad2 = dict(receipt, live_pnl_claim=True)
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_rankic_receipt(bad2, tmp_path)


def test_no_forbidden_metric_keys_in_receipt() -> None:
    from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

    _, receipt = run_cross_sectional_bench(seed=7)
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                found.add(str(key).lower())
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(receipt)
    assert found.isdisjoint({k.lower() for k in FORBIDDEN_RESEARCH_METRIC_KEYS})


def test_format_table_lists_all_ok_rows() -> None:
    frame = _frame()
    text = format_rankic_table(frame)
    assert "mean_rankIC" in text.splitlines()[0]
    ok = frame.filter(pl.col("status") == "ok")
    assert len(text.splitlines()) == ok.height + 1


def test_all_challengers_run_on_all_panels() -> None:
    frame = _frame()
    combos = frame.select(["shard", "challenger"]).unique()
    expected = {(shard, chal) for shard in PANEL_GENERATORS for chal in CHALLENGERS}
    got = {(r["shard"], r["challenger"]) for r in combos.iter_rows(named=True)}
    assert got == expected


def test_rankic_v1_audit_clean_and_tampered(tmp_path: Path) -> None:
    """The audit recounts the grid, enforces IC bounds, catches tampering."""
    from quant_fund.research.cross_sectional import rankic_v1_audit_errors

    _, receipt = run_cross_sectional_bench(seed=11)
    assert rankic_v1_audit_errors(receipt) == []

    tampered = json.loads(json.dumps(receipt))
    tampered["results"].pop()
    assert "results_grid_incomplete" in rankic_v1_audit_errors(tampered)
    assert "n_rows_mismatch" in rankic_v1_audit_errors(tampered)

    tampered = json.loads(json.dumps(receipt))
    tampered["results"][0]["mean_spearman"] = 1.5
    assert any(e.startswith("row_mean_spearman_invalid") for e in rankic_v1_audit_errors(tampered))

    tampered = json.loads(json.dumps(receipt))
    tampered["results"][0]["p_spearman"] = 1.2
    assert any(e.startswith("row_p_spearman_invalid") for e in rankic_v1_audit_errors(tampered))

    tampered = json.loads(json.dumps(receipt))
    tampered["panels"]["linear_signal"]["signal_sha256"] = "zz"
    assert any(e.startswith("panel_digest_invalid") for e in rankic_v1_audit_errors(tampered))

    tampered = json.loads(json.dumps(receipt))
    tampered["n_error_rows"] = 7
    assert "n_error_rows_mismatch" in rankic_v1_audit_errors(tampered)


def test_rankic_v1_audit_committed_receipt_clean() -> None:
    """The sealed rank-IC receipt committed to main must audit clean."""
    from quant_fund.research.receipt_v2 import verify_receipt_file

    receipt_path = (
        Path(__file__).resolve().parents[3] / "receipts" / "rankic_eval_9ebdad7da83e7348.json"
    )
    if not receipt_path.exists():
        pytest.skip("committed rankic receipt not present")
    result = verify_receipt_file(receipt_path)
    assert result["valid"], result["errors"]


def test_data_label_derived_from_panels() -> None:
    def real_panel(n_dates: int, n_assets: int, seed: int, horizons):
        p = PANEL_GENERATORS["linear_signal"](n_dates, n_assets, seed, horizons)
        return type(p)(
            dates=p.dates,
            asset_ids=p.asset_ids,
            signal=p.signal,
            forward=p.forward,
            description=p.description,
            data_label="yahoo_eod",
        )

    _, receipt = run_cross_sectional_bench(
        panels={"yahoo_x": real_panel}, n_dates=52, n_assets=8, horizons=(1,)
    )
    assert receipt["data_label"] == "yahoo_eod"
    assert receipt["panels"]["yahoo_x"]["data_label"] == "yahoo_eod"


def test_mixed_data_labels_fail_closed() -> None:
    def real_panel(n_dates: int, n_assets: int, seed: int, horizons):
        p = PANEL_GENERATORS["linear_signal"](n_dates, n_assets, seed, horizons)
        return type(p)(
            dates=p.dates,
            asset_ids=p.asset_ids,
            signal=p.signal,
            forward=p.forward,
            description=p.description,
            data_label="yahoo_eod",
        )

    with pytest.raises(ValueError, match="mixed data_label"):
        run_cross_sectional_bench(
            panels={"synth": PANEL_GENERATORS["pure_noise"], "real": real_panel},
            n_dates=52,
            n_assets=8,
            horizons=(1,),
        )


def test_empty_label_rejected() -> None:
    p = PANEL_GENERATORS["linear_signal"](52, 8, 0, (1,))
    with pytest.raises(ValueError, match="data_label"):
        type(p)(
            dates=p.dates,
            asset_ids=p.asset_ids,
            signal=p.signal,
            forward=p.forward,
            description=p.description,
            data_label=" ",
        )


def test_dataset_sha256_tracks_panels_not_run_params() -> None:
    """Same panels under a different challenger set share dataset_sha256;
    a different seed regenerates the panels and changes it."""
    chal = list(CHALLENGERS)[:1]
    _, r1 = run_cross_sectional_bench(seed=42, challengers=chal)
    _, r2 = run_cross_sectional_bench(seed=42)
    _, r3 = run_cross_sectional_bench(seed=43, challengers=chal)
    d1, d2, d3 = (r["dataset_sha256"] for r in (r1, r2, r3))
    assert len(d1) == 64 and all(c in "0123456789abcdef" for c in d1)
    assert d1 == d2  # challenger set is a run param, not data
    assert r1["inputs_sha256"] != r2["inputs_sha256"]
    assert d1 != d3


def test_forward_returns_are_true_forward_windows() -> None:
    """fwd[t] must be the sum of daily[t+1..t+h] — realized AFTER the signal,
    per the panel's no-lookahead contract. Pins direction and the telescoping
    identity fwd5[t] = fwd1[t] + fwd4[t+1]; a trailing window fails both."""
    from quant_fund.research.cross_sectional import _forward_returns

    n_dates, n_assets = 80, 16
    fw = _forward_returns(np.random.default_rng(11), n_dates, n_assets, (1, 4, 5))
    f1, f4, f5 = fw[1], fw[4], fw[5]
    # NaN block sits at the END (dates without a full h-window), not the start.
    assert np.isfinite(f5[0]).all()
    assert np.isnan(f5[-5:]).all() and np.isfinite(f5[:-5]).all()
    # Telescoping: fwd1[t] + fwd4[t+1] == fwd5[t] for all valid t.
    np.testing.assert_allclose(f5[:-5], f1[:-5] + f4[1:-4], rtol=1e-12, atol=1e-12)
