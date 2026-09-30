"""quantile_ladder — merged e-process audit across the quantile vector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.research.quantile_ladder import (
    DEFAULT_LEVELS,
    QuantileLadder,
    _quantile_ladder_errors,
    quantile_ladder_report,
    write_ladder_receipt,
)

LEVELS = DEFAULT_LEVELS


def _rows(y: np.ndarray, q_of_tau: dict[float, float]):
    """Constant-quantile forecaster stream: same vector at every origin."""
    return [(float(v), dict(q_of_tau)) for v in y]


def _calibrated_rows(n: int, seed: int):
    """y ~ N(0,1) with exact normal quantiles — the calibration null."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    return _rows(y, {tau: float(norm.ppf(tau)) for tau in LEVELS})


# Wrong-shape forecaster: breach-rate map r_tau != tau at most levels,
# assembled so that the marginal audits a single-level watch runs stay
# nominal — the median is exact and the 90% band covers exactly 90%
# (0.01 + (1 - 0.91) = 0.10) while the mid is over-covered
# (r_0.75 - r_0.25 = 0.65 > 0.50) and the tails are mis-set
# (fat left: r_0.05 = 0.01; thin right: r_0.95 = 0.91).
WRONG_SHAPE_RATES = {
    0.05: 0.01,
    0.10: 0.06,
    0.15: 0.10,
    0.20: 0.14,
    0.25: 0.17,
    0.30: 0.21,
    0.35: 0.27,
    0.40: 0.34,
    0.45: 0.42,
    0.50: 0.50,
    0.55: 0.58,
    0.60: 0.66,
    0.65: 0.73,
    0.70: 0.78,
    0.75: 0.82,
    0.80: 0.86,
    0.85: 0.88,
    0.90: 0.895,
    0.95: 0.91,
}


def _wrong_shape_rows(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    return _rows(y, {tau: float(norm.ppf(WRONG_SHAPE_RATES[tau])) for tau in LEVELS})


class _IntervalRateEProcess:
    """Minimal Bernoulli LR-mixture watch on the 90% band (coverage_watch shape).

    Included here so the wrong-shape test pins that a marginal single-band
    audit stays silent on this stream — the band covers exactly 90%.
    """

    def __init__(self, p0: float = 0.10, alt_grid=(0.5, 0.7, 1.3, 2.0)) -> None:
        self.p0 = p0
        self.wealths = [1.0] * len(alt_grid)
        self.grid = list(alt_grid)

    def update(self, breach: bool) -> float:
        for i, g in enumerate(self.grid):
            p1 = min(0.999999, g * self.p0)
            num = p1 if breach else 1.0 - p1
            den = self.p0 if breach else 1.0 - self.p0
            self.wealths[i] *= num / den
        return self.evalue

    @property
    def evalue(self) -> float:
        return sum(self.wealths) / len(self.wealths)


def test_null_family_alarm_rate() -> None:
    """Exact quantiles: the Bonferroni family claim stays at level alpha."""
    alpha, n_runs, n_steps = 0.10, 100, 260
    family_alarms = 0
    merged_alarms = 0
    for seed in range(n_runs):
        ladder = QuantileLadder(alpha=alpha)
        for realized, quantiles in _calibrated_rows(n_steps, seed):
            ladder.update(realized, quantiles)
        family_alarms += int(bool(ladder.alarmed_levels))
        merged_alarms += int(ladder.merged_alarmed)
    slack = 3.0 * np.sqrt(alpha * (1.0 - alpha) / n_runs)
    assert family_alarms / n_runs <= alpha + slack
    assert merged_alarms / n_runs <= alpha + slack


def test_per_step_validity_and_causality() -> None:
    """E[e_t | F_{t-1}] = 1 under the null for any predictable stake."""
    ladder = QuantileLadder()
    for realized, quantiles in _calibrated_rows(300, 0):
        ladder.update(realized, quantiles)
    # Causality: replaying a prefix must reproduce identical states.
    prefix = QuantileLadder()
    rows = _calibrated_rows(300, 0)
    for realized, quantiles in rows[:100]:
        prefix.update(realized, quantiles)
    assert [s.merged_evalue for s in prefix.states] == [
        s.merged_evalue for s in ladder.states[:100]
    ]


def test_wrong_shape_caught_where_single_band_silent() -> None:
    """The pinned failure mode: exact median + nominal 90% band, wrong shape."""
    n = 600
    rows = _wrong_shape_rows(n, 0)
    ladder = QuantileLadder(alpha=0.05)
    band = _IntervalRateEProcess(p0=0.10)
    n_band_breach = 0
    for realized, quantiles in rows:
        ladder.update(realized, quantiles)
        band_breach = realized < quantiles[0.05] or realized > quantiles[0.95]
        n_band_breach += int(band_breach)
        band.update(band_breach)
    # The marginal audits see exactly what the forecaster claims.
    assert n_band_breach / n == pytest.approx(0.10, abs=0.04)
    assert band.evalue < 20.0  # 1/alpha: a single-level watch stays quiet
    assert 0.5 not in ladder.alarmed_levels  # median is exact by construction
    # The ladder sees the shape error: both tail directions alarm.
    assert 0.05 in ladder.alarmed_levels  # fat left tail: breach 0.02 < 0.05
    assert 0.95 in ladder.alarmed_levels  # thin right tail: breach 0.92 < 0.95
    assert ladder.merged_alarmed


def test_undercoverage_alarms_low_levels() -> None:
    """Persistently thin lower tail: low-tau levels breach above tau."""
    rng = np.random.default_rng(1)
    y = rng.standard_normal(400)
    rates = {tau: min(0.98, tau + 0.10) for tau in LEVELS}
    ladder = QuantileLadder(alpha=0.05)
    for realized, quantiles in _rows(y, {t: float(norm.ppf(r)) for t, r in rates.items()}):
        ladder.update(realized, quantiles)
    assert 0.05 in ladder.alarmed_levels
    assert ladder.merged_alarmed


def _deterministic_stream(rates: list[float], n: int) -> tuple[list[float], dict[float, float]]:
    """Emit realized values between adjacent quantiles so level i breaches
    at exactly rate_i (must be nondecreasing in i). Emitting y = j + 0.5
    breaches every level with index >= j, so the emitted-index frequencies
    are the prefix differences f_j = r_j - r_{j-1}, f_19 = 1 - r_18.
    Deterministic: no RNG — the merged-vs-family separation is exact."""
    assert all(b >= a for a, b in zip(rates, rates[1:], strict=False))
    freqs = [rates[0]] + [rates[i] - rates[i - 1] for i in range(1, len(rates))] + [1.0 - rates[-1]]
    remaining = [int(round(f * n)) for f in freqs]
    emitted = []
    while sum(remaining) > 0 and len(emitted) < n:
        for j in range(len(remaining)):
            if remaining[j] > 0 and len(emitted) < n:
                emitted.append(j)
                remaining[j] -= 1
    quantiles = {tau: float(i + 1) for i, tau in enumerate(LEVELS)}
    return [j + 0.5 for j in emitted], quantiles


def test_merged_claim_separate_from_family() -> None:
    """Diffuse mild miscalibration: merged alarms with no single level alarmed.

    Every level undercovers by +0.16 in breach rate — each level's e grows
    but stays under the Bonferroni bound n_levels/alpha = 380, while the
    dependence-robust mean crosses 1/alpha = 20."""
    rates = [min(0.999, tau + 0.16) for tau in LEVELS]
    y, quantiles = _deterministic_stream(rates, 60)
    ladder = QuantileLadder(alpha=0.05)
    for realized in y:
        ladder.update(realized, quantiles)
    final = ladder.states[-1]
    assert final.merged_alarmed
    assert final.alarmed_levels == ()
    # ...and the two claims are reported as separate fields.
    assert final.merged_alarmed is not bool(final.alarmed_levels)


def test_tie_counts_as_breach() -> None:
    """realized == q_tau hits the <= convention: a degenerate vector breaches all."""
    ladder = QuantileLadder(levels=(0.25, 0.5, 0.75))
    for _ in range(5):
        ladder.update(1.0, {0.25: 1.0, 0.5: 1.0, 0.75: 1.0})
    assert ladder._watches[0.5].n_breach == 5


def test_fail_closed_edges() -> None:
    ladder = QuantileLadder()
    good = {tau: float(norm.ppf(tau)) for tau in LEVELS}
    with pytest.raises(ValueError, match="finite"):
        ladder.update(float("nan"), good)
    with pytest.raises(ValueError, match="finite"):
        ladder.update(0.0, {**good, 0.5: float("inf")})
    with pytest.raises(ValueError, match="missing declared level"):
        ladder.update(0.0, {k: v for k, v in good.items() if k != 0.5})
    misordered = dict(good)
    misordered[0.75] = misordered[0.25]
    with pytest.raises(ValueError, match="nondecreasing"):
        ladder.update(0.0, misordered)
    with pytest.raises(ValueError, match="mapping"):
        ladder.update(0.0, [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numeric"):
        ladder.update(0.0, {**good, "bad": 1.0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="nonempty"):
        QuantileLadder(levels=())
    with pytest.raises(ValueError, match="strictly increasing"):
        QuantileLadder(levels=(0.5, 0.5))
    with pytest.raises(ValueError, match="0, 1"):
        QuantileLadder(levels=(0.0, 0.5))
    with pytest.raises(ValueError, match="alpha"):
        QuantileLadder(alpha=1.0)
    with pytest.raises(ValueError, match="lam"):
        QuantileLadder(lam=0.0)
    with pytest.raises(ValueError, match="nonempty"):
        quantile_ladder_report([])
    with pytest.raises(ValueError, match="nonempty string"):
        quantile_ladder_report(_calibrated_rows(50, 0), data_label="  ")


def test_failed_update_leaves_state_unchanged() -> None:
    """A rejected row must not consume an origin or tilt the bets."""
    ladder = QuantileLadder()
    rows = _calibrated_rows(120, 3)
    for realized, quantiles in rows:
        ladder.update(realized, quantiles)
    before = ladder.merged_evalue
    with pytest.raises(ValueError):
        ladder.update(float("inf"), {tau: 0.0 for tau in LEVELS})
    assert ladder.merged_evalue == before
    assert ladder.n_origins == 120


def test_receipt_shape_seal_and_tamper(tmp_path: Path) -> None:
    receipt = quantile_ladder_report(_wrong_shape_rows(400, 0), data_label="SYNTHETIC")
    assert receipt["kind"] == "quantile_ladder.v1"
    assert receipt["research_only"] is True
    assert receipt["live_pnl_claim"] is False
    assert receipt["data_label"] == "SYNTHETIC"
    assert set(receipt["per_level"]) == {repr(t) for t in LEVELS}
    assert receipt["n_origins"] == 400
    assert receipt["n_levels"] == len(LEVELS)
    assert "anytime_valid" in receipt["evidence"]
    for entry in receipt["per_level"].values():
        assert set(entry) == {"e", "p_hat", "n", "alarmed"}
        assert entry["e"] > 0.0
        assert 0.0 <= entry["p_hat"] <= 1.0
    mean_e = sum(e["e"] for e in receipt["per_level"].values()) / receipt["n_levels"]
    assert receipt["merged_evalue"] == pytest.approx(mean_e)

    path = write_ladder_receipt(receipt, tmp_path)
    assert path.name.startswith("quantile_ladder_")

    from quant_fund.research.receipt_v2 import verify_receipt_file

    result = verify_receipt_file(path)
    assert result["valid"], result["errors"]

    # Second write of identical content is a no-op, not an error.
    assert write_ladder_receipt(receipt, tmp_path) == path

    # Tampering breaks the seal AND the contract catches forged claims.
    bad = json.loads(path.read_text())
    bad["merged_evalue"] = mean_e * 1.5  # under the merger bound, but not the mean
    path.write_text(json.dumps(bad))
    result = verify_receipt_file(path)
    assert not result["valid"]
    assert "receipt_sha256_mismatch" in result["errors"]
    contract_errors = _quantile_ladder_errors(bad)
    assert "merged_evalue_not_mean_of_levels" in contract_errors

    # A merged claim above the merger bound is rejected outright.
    worse = json.loads(path.read_text())
    worse["merged_evalue"] = 1e9
    assert "merged_evalue_exceeds_merger_bound" in _quantile_ladder_errors(worse)


def test_contract_rejects_bad_claims() -> None:
    receipt = quantile_ladder_report(_calibrated_rows(80, 0), data_label="SYNTHETIC")
    assert _quantile_ladder_errors(receipt) == []

    for mutation, tag in (
        ({"alpha": 2.0}, "alpha_out_of_unit_interval"),
        ({"lam": 0.0}, "lam_out_of_unit_interval"),
        ({"data_label": ""}, "data_label_not_nonempty_str"),
        ({"research_only": False}, "research_only_not_true"),
        ({"live_pnl_claim": True}, "live_pnl_claim_not_false"),
        ({"merged_alarmed": True}, "merged_alarmed_below_threshold"),
        ({"alarmed_levels": [0.37]}, "alarmed_level_not_declared:0.37"),
        ({"merged_evalue": -1.0}, "merged_evalue_not_positive_finite"),
        ({"n_origins": 0}, "n_origins_not_positive_int"),
    ):
        bad = dict(receipt, **mutation)
        errors = _quantile_ladder_errors(bad)
        assert tag in errors, (tag, errors)

    bad_key = dict(receipt)
    bad_key["per_level"] = dict(receipt["per_level"], **{"0.33": receipt["per_level"]["0.5"]})
    assert "per_level_key_not_declared:0.33" in _quantile_ladder_errors(bad_key)

    # The unversioned kind alias dispatches the same contract.
    from quant_fund.research.receipt_v2 import verify_receipt_payload

    sealed = dict(receipt, kind="quantile_ladder")
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    sealed["receipt_sha256"] = hash_bytes(canonical_json_bytes(sealed))
    result = verify_receipt_payload(sealed)
    assert result["valid"], result["errors"]


def test_missing_levels_subset_still_valid_and_alarmable() -> None:
    """A 3-level ladder on a narrow grid alarms on shape error too."""
    ladder = QuantileLadder(levels=(0.25, 0.5, 0.75), alpha=0.05)
    rng = np.random.default_rng(5)
    y = rng.standard_normal(300)
    rates = {0.25: 0.17, 0.5: 0.5, 0.75: 0.82}
    for realized, quantiles in _rows(y, {t: float(norm.ppf(r)) for t, r in rates.items()}):
        ladder.update(realized, quantiles)
    assert ladder.merged_alarmed
    assert ladder.alarmed_levels != ()
