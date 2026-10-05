"""Unit tests for research/spa_test.py — the spa_test bench core (wave 18).

Wired through a *lazy* import inside
:func:`quant_fund.research.benches_w18.bench_spa_test`, so a module-level import
scan reports it as unreachable. This file is the compact wave-18 bench core; the
fuller canon multiplicity battery lives in ``quant_fund.metrics.snooping`` and
``quant_fund.metrics.inference`` and is covered by its own tests.

What is pinned here: the stationary bootstrap's index contract, White's Reality
Check p-value under a global null versus a planted edge, Romano–Wolf stepdown
FWER control at a fixed seed, and the sealed SYNTHETIC bench payload. All data
is synthetic Gaussian noise — a correctness check on the estimators, never
market evidence and never a headline performance ratio.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.spa_test import (
    SPA_SCHEMA,
    StepdownResult,
    reality_check_pvalue,
    romano_wolf_stepdown,
    spa_bench,
    stationary_bootstrap_indices,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

#: FWER slack over alpha. The bench itself ships ``alpha + 0.08`` at 12 reps;
#: 40 reps at 0.15 keeps the seeded run stable while still failing loudly if
#: the stepdown loses multiplicity control (an uncorrected max-t would sit
#: near 1.0 here).
_ALPHA = 0.10
_FWER_TOL = 0.15


def _null_losses(n: int, k: int, seed: int) -> NDArray[np.float64]:
    """Every column iid N(0, 1): no head beats the benchmark (global null)."""
    return np.random.default_rng(seed).standard_normal((n, k))


def _planted_losses(
    n: int, k: int, seed: int, *, delta: float, winner: int = 1
) -> NDArray[np.float64]:
    """Column ``winner`` loses ``delta`` per period — a real edge to detect."""
    losses = _null_losses(n, k, seed)
    losses[:, winner] -= delta
    return losses


# ---------------------------------------------------------------------------
# stationary_bootstrap_indices
# ---------------------------------------------------------------------------


def test_stationary_bootstrap_indices_shape_dtype_and_range() -> None:
    idx = stationary_bootstrap_indices(50, 0.2, np.random.default_rng(0))
    assert idx.shape == (50,)
    assert idx.dtype == np.int64
    assert int(idx.min()) >= 0 and int(idx.max()) < 50


def test_stationary_bootstrap_indices_are_seed_deterministic() -> None:
    a = stationary_bootstrap_indices(64, 0.1, np.random.default_rng(7))
    b = stationary_bootstrap_indices(64, 0.1, np.random.default_rng(7))
    c = stationary_bootstrap_indices(64, 0.1, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stationary_bootstrap_block_structure_follows_p() -> None:
    """Small p → long contiguous runs (with wraparound); p = 1 → fresh draws.

    This is the defining property of the Politis–Romano scheme: mean block
    length 1/p, continued with probability 1 − p and wrapped modulo n.
    """
    n = 400

    def contiguous_fraction(p: float, seed: int) -> float:
        idx = stationary_bootstrap_indices(n, p, np.random.default_rng(seed))
        return float(np.mean(idx[1:] == (idx[:-1] + 1) % n))

    sticky = contiguous_fraction(0.02, 3)
    fresh = contiguous_fraction(1.0, 3)
    assert sticky > 0.9
    assert fresh < 0.1
    assert sticky > fresh


def test_stationary_bootstrap_p_one_draws_independent_positions() -> None:
    """p = 1 resets every step, so the index vector covers many distinct slots."""
    idx = stationary_bootstrap_indices(200, 1.0, np.random.default_rng(5))
    assert len(set(idx.tolist())) > 100


def test_stationary_bootstrap_rejects_invalid_p() -> None:
    rng = np.random.default_rng(0)
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=r"p must be in \(0,1\]"):
            stationary_bootstrap_indices(20, bad, rng)
    # The upper edge is legal: p = 1 means "always resample the position".
    assert stationary_bootstrap_indices(20, 1.0, rng).shape == (20,)


# ---------------------------------------------------------------------------
# reality_check_pvalue
# ---------------------------------------------------------------------------


def test_reality_check_pvalue_is_large_under_the_global_null() -> None:
    losses = _null_losses(200, 4, seed=11)
    p = reality_check_pvalue(losses, benchmark_col=0, n_boot=200, seed=11)
    assert 0.0 <= p <= 1.0
    assert p > _ALPHA, f"a null arm must not look significant, got p={p}"


def test_reality_check_pvalue_is_small_for_a_planted_edge() -> None:
    losses = _planted_losses(200, 4, seed=12, delta=1.0)
    p = reality_check_pvalue(losses, benchmark_col=0, n_boot=200, seed=12)
    assert p < 0.01


def test_reality_check_pvalue_orders_null_above_edge() -> None:
    p_null = reality_check_pvalue(_null_losses(200, 4, seed=13), n_boot=150, seed=13)
    p_edge = reality_check_pvalue(_planted_losses(200, 4, seed=13, delta=0.75), n_boot=150, seed=13)
    assert p_null > p_edge


def test_reality_check_pvalue_is_seed_deterministic() -> None:
    losses = _null_losses(120, 3, seed=14)
    a = reality_check_pvalue(losses, n_boot=80, seed=21)
    b = reality_check_pvalue(losses, n_boot=80, seed=21)
    c = reality_check_pvalue(losses, n_boot=80, seed=22)
    assert a == b
    assert 0.0 <= c <= 1.0


def test_reality_check_pvalue_uses_the_requested_benchmark_column() -> None:
    """Column 1 is the strong head: benchmarking against it must *not* look
    significant, while benchmarking the same data against column 0 must."""
    losses = _planted_losses(200, 3, seed=15, delta=1.0, winner=1)
    assert reality_check_pvalue(losses, benchmark_col=0, n_boot=150, seed=15) < 0.01
    assert reality_check_pvalue(losses, benchmark_col=1, n_boot=150, seed=15) > _ALPHA


# ---------------------------------------------------------------------------
# Fail-closed input validation
# ---------------------------------------------------------------------------


def test_loss_matrix_shape_is_enforced() -> None:
    with pytest.raises(ValueError, match=r"\(n, k>=2\)"):
        reality_check_pvalue(np.zeros(10), n_boot=10)
    with pytest.raises(ValueError, match=r"\(n, k>=2\)"):
        reality_check_pvalue(np.zeros((10, 1)), n_boot=10)
    with pytest.raises(ValueError, match=r"\(n, k>=2\)"):
        romano_wolf_stepdown(np.zeros((10, 1)), n_boot=10)


def test_benchmark_column_range_is_enforced() -> None:
    losses = _null_losses(40, 3, seed=16)
    for bad in (-1, 3, 7):
        with pytest.raises(ValueError, match="benchmark_col out of range"):
            reality_check_pvalue(losses, benchmark_col=bad, n_boot=10)
        with pytest.raises(ValueError, match="benchmark_col out of range"):
            romano_wolf_stepdown(losses, benchmark_col=bad, n_boot=10)


def test_invalid_p_propagates_from_the_bootstrap() -> None:
    losses = _null_losses(40, 3, seed=17)
    with pytest.raises(ValueError, match=r"p must be in \(0,1\]"):
        reality_check_pvalue(losses, n_boot=10, p=0.0)
    with pytest.raises(ValueError, match=r"p must be in \(0,1\]"):
        romano_wolf_stepdown(losses, n_boot=10, p=1.5)


# ---------------------------------------------------------------------------
# romano_wolf_stepdown
# ---------------------------------------------------------------------------


def test_stepdown_adjusted_pvalues_are_ordered_and_in_range() -> None:
    losses = _null_losses(150, 6, seed=18)
    out = romano_wolf_stepdown(losses, n_boot=120, alpha=_ALPHA, seed=18)
    assert isinstance(out, StepdownResult)
    # The benchmark column is excluded: adj_p indexes the head set.
    assert out.adj_p.shape == (5,)
    assert np.all(out.adj_p >= 0.0) and np.all(out.adj_p <= 1.0)
    assert all(0 <= head < 5 for head in out.rejected)
    assert all(out.adj_p[head] < _ALPHA for head in out.rejected)
    # Stepdown enforcement: adjusted p must be non-decreasing along the
    # descending observed-t order (recomputed here from the loss matrix).
    d = losses[:, 0][:, None] - losses[:, 1:]
    n = d.shape[0]
    se = np.where(d.std(axis=0, ddof=1) < 1e-12, 1e-12, d.std(axis=0, ddof=1))
    t_obs = np.sqrt(n) * d.mean(axis=0) / se
    along_order = out.adj_p[np.argsort(-t_obs)]
    assert np.all(np.diff(along_order) >= -1e-12)


def test_stepdown_indices_are_over_the_head_set_not_the_raw_columns() -> None:
    """With benchmark_col=1 the head set is columns (0, 2, 3, …) in that order,
    so positional index 0 is loss column 0 — pinned because the dataclass
    comment says "loss_matrix columns"."""
    losses = _planted_losses(200, 4, seed=19, delta=1.5, winner=0)
    out = romano_wolf_stepdown(losses, benchmark_col=1, n_boot=120, alpha=_ALPHA, seed=19)
    assert out.adj_p.shape == (3,)
    assert out.rejected[0] == 0
    assert out.adj_p[0] < _ALPHA


def test_stepdown_detects_a_planted_winner() -> None:
    losses = _planted_losses(200, 5, seed=20, delta=1.0, winner=3)
    out = romano_wolf_stepdown(losses, n_boot=150, alpha=_ALPHA, seed=20)
    # Head positions are loss columns minus the benchmark (column 0), so the
    # planted winner at column 3 is head position 2.
    assert 2 in out.rejected
    assert out.adj_p[2] < _ALPHA
    assert all(out.adj_p[j] >= out.adj_p[2] for j in range(out.adj_p.size))


def test_stepdown_fwer_under_global_null_stays_near_alpha() -> None:
    """FWER ≤ alpha + tolerance across seeded null replications.

    Every column is iid N(0, 1), so *any* rejection is a familywise error.
    """
    rejections = 0
    n_reps = 40
    for rep in range(n_reps):
        losses = _null_losses(100, 5, seed=1000 + rep)
        out = romano_wolf_stepdown(losses, n_boot=100, alpha=_ALPHA, seed=rep)
        rejections += int(len(out.rejected) > 0)
    fwer = rejections / n_reps
    assert fwer <= _ALPHA + _FWER_TOL, f"FWER {fwer:.3f} exceeds alpha + tol"


def test_stepdown_power_rises_with_the_planted_edge() -> None:
    """Same seeds, same shapes: a bigger delta rejects more often."""

    def reject_rate(delta: float) -> float:
        hits = 0
        reps = 12
        for rep in range(reps):
            losses = _planted_losses(120, 4, seed=500 + rep, delta=delta)
            hits += int(len(romano_wolf_stepdown(losses, n_boot=100, seed=rep).rejected) > 0)
        return hits / reps

    assert reject_rate(0.0) <= reject_rate(1.0)
    assert reject_rate(1.0) > 0.5


def test_stepdown_zero_variance_diffs_do_not_divide_by_zero() -> None:
    """A head identical to the benchmark gives d ≡ 0 → se = 0; the floor must
    keep the statistic finite and must not manufacture a rejection."""
    ramp = np.arange(60, dtype=np.float64)
    losses = np.column_stack([ramp, ramp, ramp + 5.0])
    out = romano_wolf_stepdown(losses, n_boot=50, alpha=_ALPHA, seed=3)
    assert np.all(np.isfinite(out.adj_p))
    assert np.all(out.adj_p >= 0.0) and np.all(out.adj_p <= 1.0)
    assert out.rejected == []
    assert float(reality_check_pvalue(losses, n_boot=50, seed=3)) >= 0.0


# ---------------------------------------------------------------------------
# Sealed bench payload
# ---------------------------------------------------------------------------

_BENCH = {"n": 100, "k": 4, "n_boot": 60, "n_reps": 3, "deltas": (0.0, 0.5), "seed": 25}


@pytest.fixture(scope="module")
def bench() -> dict[str, object]:
    return spa_bench(**_BENCH)  # type: ignore[arg-type]


def test_bench_payload_is_sealed_and_synthetic(bench: dict[str, object]) -> None:
    assert bench["schema"] == SPA_SCHEMA
    assert bench["kind"] == "spa_test"
    assert bench["data_label"] == "SYNTHETIC"
    assert bench["research_only"] is True
    assert bench["n"] == _BENCH["n"]
    assert bench["k"] == _BENCH["k"]
    assert bench["n_boot"] == _BENCH["n_boot"]
    assert bench["alpha"] == pytest.approx(0.10)
    assert isinstance(bench["interpretation"], str) and bench["interpretation"]


def test_bench_arms_cover_every_delta(bench: dict[str, object]) -> None:
    arms = bench["arms"]
    assert isinstance(arms, dict)
    assert set(arms) == {f"delta={d}" for d in _BENCH["deltas"]}
    for name, block in arms.items():
        assert isinstance(block, dict), name
        assert block["n_reps"] == float(_BENCH["n_reps"])
        rate = float(block["reject_rate"])
        assert 0.0 <= rate <= 1.0, name


def test_bench_flags_are_booleans(bench: dict[str, object]) -> None:
    assert isinstance(bench["fwer_controlled"], bool)
    assert isinstance(bench["power_monotone"], bool)


def test_bench_payload_hash_is_reproducible(bench: dict[str, object]) -> None:
    digest = bench["payload_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    body = {k: v for k, v in bench.items() if k != "payload_sha256"}
    assert hash_bytes(canonical_json_bytes(body)) == digest


def test_bench_payload_carries_no_forbidden_headline_metric(bench: dict[str, object]) -> None:
    assert family_blob_forbidden_metrics_absent(bench)


def test_bench_is_seed_deterministic(bench: dict[str, object]) -> None:
    """Same seed → byte-identical sealed payload.

    The reverse is deliberately *not* asserted: the payload records only coarse
    per-arm reject rates (and no ``seed`` field), so two seeds at ``n_reps=3``
    can land on the same 1/3 grid and hash identically. Determinism, not
    seed-sensitivity, is the reproducibility contract here.
    """
    again = spa_bench(**_BENCH)  # type: ignore[arg-type]
    assert again["payload_sha256"] == bench["payload_sha256"]
    assert again == bench
    assert "seed" not in bench


def test_bench_null_arm_controls_fwer_at_a_meaningful_size() -> None:
    """The claim the lane ships with, at the reduced size the scorecard uses:
    the delta=0 arm must stay inside alpha + tolerance."""
    payload = spa_bench(n=100, k=4, n_boot=100, n_reps=12, deltas=(0.0,), seed=20261004)
    arms = payload["arms"]
    assert isinstance(arms, dict)
    block = arms["delta=0.0"]
    assert isinstance(block, dict)
    assert float(block["reject_rate"]) <= _ALPHA + 0.08
    assert payload["fwer_controlled"] is True
