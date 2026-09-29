"""cost_calibration receipts: the dataset_sha256 convention pins."""

from __future__ import annotations

from quant_fund.research.cost_calibration import run_cost_calibration_trials


def test_dataset_sha256_tracks_panel_not_run_params() -> None:
    """Same synthetic OHLC book under different estimator sets shares
    dataset_sha256; a different seed regenerates the panel and changes it."""
    _, r1 = run_cost_calibration_trials(seed=7, n_dates=16, n_names=4, estimators=("flat",))
    _, r2 = run_cost_calibration_trials(
        seed=7, n_dates=16, n_names=4, estimators=("flat", "corwin_schultz")
    )
    _, r3 = run_cost_calibration_trials(seed=8, n_dates=16, n_names=4, estimators=("flat",))
    d1, d2, d3 = (r["dataset_sha256"] for r in (r1, r2, r3))
    assert len(d1) == 64 and all(c in "0123456789abcdef" for c in d1)
    assert d1 == d2  # estimator set is a run param, not data
    assert r1["inputs_sha256"] != r2["inputs_sha256"]
    assert d1 != d3
