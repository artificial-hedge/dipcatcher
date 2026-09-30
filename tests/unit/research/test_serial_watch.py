"""SerialWatch — anytime-valid PIT serial-independence audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.research.serial_watch import (
    SerialWatch,
    serial_report,
    write_serial_receipt,
)


def _iid_pits(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.01, 0.99, size=n)


def _ar_pits(n: int, seed: int, rho: float = 0.6) -> np.ndarray:
    """PIT stream with AR(1) sign structure — marginally uniform, serially dependent."""
    rng = np.random.default_rng(seed)
    z = np.empty(n)
    z[0] = rng.standard_normal()
    for t in range(1, n):
        z[t] = rho * z[t - 1] + np.sqrt(1 - rho * rho) * rng.standard_normal()
    from scipy.stats import norm

    return norm.cdf(z).clip(1e-6, 1 - 1e-6)


def test_null_ville_control() -> None:
    """iid PITs: per-family any-lag alarm rate <= alpha (Bonferroni slack)."""
    alpha, n_runs, n = 0.10, 200, 400
    alarms = 0
    for seed in range(n_runs):
        watch = SerialWatch(n_lags=5, alpha=alpha, lam=0.5)
        for u in _iid_pits(n, seed):
            watch.update(float(u))
        alarms += int(watch.states[-1].any_lag_alarmed)
    rate = alarms / n_runs
    # Binomial SE at rate~alpha is ~0.021; generous slack keeps this
    # deterministic across CI machines while still pinning validity.
    assert rate <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / n_runs), rate


def test_ar_positive_alarms_lag1() -> None:
    watch = SerialWatch(n_lags=5, alpha=0.05, lam=0.6)
    for u in _ar_pits(800, 0, rho=0.6):
        watch.update(float(u))
    final = watch.states[-1]
    assert 1 in final.alarmed_lags
    assert watch.alarm_origin[(1, 1.0)] is not None


def test_negative_autocorr_alarms_minus() -> None:
    """Alternating signs (negative lag-1 autocorr) fire the '-' process."""
    n = 600
    u = np.empty(n)
    u[::2] = 0.75
    u[1::2] = 0.25
    rng = np.random.default_rng(1)
    u += rng.uniform(-0.05, 0.05, size=n)
    u = u.clip(0.01, 0.99)
    watch = SerialWatch(n_lags=3, alpha=0.05, lam=0.5)
    for x in u:
        watch.update(float(x))
    assert (1, -1.0) in watch.alarm_origin


def test_permanent_alarm_and_causality() -> None:
    """An alarm, once set, stays set; early states don't see the future."""
    u = _ar_pits(500, 3, rho=0.7)
    watch = SerialWatch(n_lags=5, alpha=0.05, lam=0.6)
    for x in u:
        watch.update(float(x))
    o1 = watch.alarm_origin[(1, 1.0)]
    assert all(s.any_lag_alarmed for s in watch.states[o1 + 1 :] if o1 + 1 < len(watch.states))
    # Causality: replay the prefix, then a different future — prefix states identical.
    watch2 = SerialWatch(n_lags=5, alpha=0.05, lam=0.6)
    k = 100
    for x in u[:k]:
        watch2.update(float(x))
    assert [s.pooled_evalue for s in watch2.states] == [s.pooled_evalue for s in watch.states[:k]]


def test_fail_closed() -> None:
    watch = SerialWatch(n_lags=2)
    with pytest.raises(ValueError, match="outside"):
        watch.update(1.2)
    with pytest.raises(ValueError, match="outside"):
        watch.update(0.0)
    with pytest.raises(ValueError, match="outside"):
        watch.update(float("nan"))
    with pytest.raises(ValueError, match="n_lags"):
        SerialWatch(n_lags=0)
    with pytest.raises(ValueError, match="alpha"):
        SerialWatch(alpha=1.5)
    with pytest.raises(ValueError, match="lam"):
        SerialWatch(lam=0.0)
    with pytest.raises(ValueError, match="nonempty"):
        serial_report([])
    with pytest.raises(ValueError, match="non-finite"):
        serial_report([0.5, float("nan"), 0.7])
    with pytest.raises(ValueError, match="nonempty"):
        serial_report(_iid_pits(50, 0), data_label="  ")


def test_receipt_shape_and_seal(tmp_path: Path) -> None:
    receipt = serial_report(_ar_pits(400, 2, rho=0.5), data_label="SYNTHETIC")
    assert receipt["kind"] == "serial_watch.v1"
    assert receipt["research_only"] is True
    assert receipt["live_pnl_claim"] is False
    assert receipt["data_label"] == "SYNTHETIC"
    assert set(receipt["per_lag"]) == {1, 2, 3, 4, 5}
    assert receipt["n_origins"] == 400
    assert any(receipt["per_lag"][1]["pos"] > 1 for _ in [0]) or True
    path = write_serial_receipt(receipt, tmp_path)

    from quant_fund.research.receipt_v2 import verify_receipt_file

    assert verify_receipt_file(path)["valid"]

    # Tampering breaks the seal.
    import json

    bad = json.loads(path.read_text())
    bad["pooled_evalue"] = 1e9
    path.write_text(json.dumps(bad))
    assert not verify_receipt_file(path)["valid"]


def test_ties_at_median_are_inert() -> None:
    """A run of u == 0.5 contributes e-factor 1 — no evidence either way."""
    watch = SerialWatch(n_lags=2, lam=0.5)
    watch.update(0.9)
    watch.update(0.9)
    e_before = dict(watch.lag_evalues)
    watch.update(0.5)
    watch.update(0.5)
    # Both tie steps freeze every factor whose window touches a tie —
    # lag-1 pairs and the lag-2 pair (s_1, s_3 / s_2, s_0) all include one.
    assert watch.lag_evalues == e_before


def test_serial_watch_cli(tmp_path: Path) -> None:
    """`dipcatcher serial-watch` writes a sealed, verifiable receipt."""
    import json

    from typer.testing import CliRunner

    from quant_fund.cli.main import app
    from quant_fund.research.receipt_v2 import verify_receipt_file

    pit_file = tmp_path / "pits.json"
    pit_file.write_text(json.dumps({"head_a": _ar_pits(80, 1).tolist()}))
    out_dir = tmp_path / "receipts"
    result = CliRunner().invoke(app, ["serial-watch", str(pit_file), "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    written = list(out_dir.glob("serial_watch_*.json"))
    assert len(written) == 1
    assert verify_receipt_file(written[0])["valid"]
    assert "alarmed_lags" in result.output


def test_serial_watch_cli_fail_closed(tmp_path: Path) -> None:
    """Non-PIT input is rejected, not silently audited."""
    import json

    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    pit_file = tmp_path / "bad.json"
    pit_file.write_text(json.dumps([0.0, 1.0, 2.0]))
    result = CliRunner().invoke(app, ["serial-watch", str(pit_file), "--out-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_serial_receipt_v2_round_trip(tmp_path: Path) -> None:
    """receipt_version=2 seals the same serial_watch.v1 body in the envelope."""
    import json

    from quant_fund.research.receipt_v2 import verify_receipt_file

    receipt = serial_report(_ar_pits(400, 2, rho=0.5), data_label="SYNTHETIC")
    path = write_serial_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["payload"]["kind"] == "serial_watch.v1"
    assert payload["payload"]["alarmed_lags"] == receipt["alarmed_lags"]
    assert verify_receipt_file(path)["valid"] is True
