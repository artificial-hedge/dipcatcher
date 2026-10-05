"""Unit tests for research/energy_score.py — the copula_specification bench core.

The lane is wired through a *lazy* import inside
:func:`quant_fund.research.benches_w18.bench_copula_specification`, so a
module-level import scan reports it as unreachable. These tests cover the score
functions directly. This is a different API from the canon univariate
``quant_fund.metrics.energy_score``: ``variogram_score`` and
``gaussian_copula_samples`` exist only here.

Every number below is SYNTHETIC — a correctness check on the estimator against
hand-computed values, never market evidence and never a headline performance
ratio. The bench payload is asserted to be sealed, labeled and free of
forbidden research-metric keys.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.energy_score import (
    ENERGY_SCORE_SCHEMA,
    energy_score,
    energy_score_bench,
    gaussian_copula_samples,
    variogram_score,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

#: Tiny rep/sample counts: the bench is a sealed-payload shape check here, the
#: statistics are checked against hand values in the tests above it.
_BENCH = {"n_reps": 3, "m": 32, "d": 3, "n_names_corr": 0.6, "seed": 11}


@pytest.fixture(scope="module")
def bench() -> dict[str, object]:
    return energy_score_bench(**_BENCH)  # type: ignore[arg-type]


def _brute_force_energy_score(y: NDArray[np.float64], s: NDArray[np.float64]) -> float:
    """ES(y, X) = E‖X−y‖ − ½E‖X−X′‖ written out as explicit loops."""
    m = s.shape[0]
    term1 = sum(float(np.linalg.norm(s[i] - y)) for i in range(m)) / m
    term2 = sum(float(np.linalg.norm(s[i] - s[j])) for i in range(m) for j in range(m)) / (m * m)
    return term1 - 0.5 * term2


def _brute_force_variogram(y: NDArray[np.float64], s: NDArray[np.float64], *, p: float) -> float:
    m, d = s.shape
    total = 0.0
    for i in range(d):
        for j in range(d):
            expected = sum(abs(float(s[r, i]) - float(s[r, j])) ** p for r in range(m)) / m
            total += (abs(float(y[i]) - float(y[j])) ** p - expected) ** 2
    return total


# ---------------------------------------------------------------------------
# energy_score — hand-computed values
# ---------------------------------------------------------------------------


def test_energy_score_hand_computed_univariate() -> None:
    """y=0, X ∈ {−1, +1}: E|X−y| = 1, E|X−X′| = (0+2+2+0)/4 = 1 → 1 − ½ = ½."""
    y = np.array([0.0])
    samples = np.array([[-1.0], [1.0]])
    assert energy_score(y, samples) == pytest.approx(0.5)


def test_energy_score_hand_computed_bivariate_single_draw() -> None:
    """One draw at (3, 4) against y = 0: ‖X−y‖ = 5, E‖X−X′‖ = 0 → ES = 5."""
    y = np.zeros(2)
    samples = np.array([[3.0, 4.0]])
    assert energy_score(y, samples) == pytest.approx(5.0)


def test_energy_score_hand_computed_two_point_bivariate() -> None:
    """y = (0,0), X ∈ {(0,0), (0,2)}: term1 = (0+2)/2 = 1,
    term2 = (0+2+2+0)/4 = 1 → ES = 1 − ½ = ½."""
    y = np.zeros(2)
    samples = np.array([[0.0, 0.0], [0.0, 2.0]])
    assert energy_score(y, samples) == pytest.approx(0.5)


def test_energy_score_matches_brute_force_on_seeded_draws() -> None:
    rng = np.random.default_rng(4)
    y = rng.standard_normal(3)
    samples = rng.standard_normal((12, 3))
    assert energy_score(y, samples) == pytest.approx(_brute_force_energy_score(y, samples))


def test_energy_score_is_proper_sharp_beats_dispersed() -> None:
    """Properness payoff (SYNTHETIC): an over-dispersed ensemble scores worse."""
    rng = np.random.default_rng(9)
    ys = rng.standard_normal((40, 3))
    sharp = rng.standard_normal((64, 3))
    wide = rng.standard_normal((64, 3)) * 3.0
    es_sharp = float(np.mean([energy_score(y, sharp) for y in ys]))
    es_wide = float(np.mean([energy_score(y, wide) for y in ys]))
    assert es_sharp < es_wide


def test_energy_score_shifts_with_observation_distance() -> None:
    samples = np.zeros((4, 2))
    assert energy_score(np.zeros(2), samples) == pytest.approx(0.0)
    assert energy_score(np.array([3.0, 4.0]), samples) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# variogram_score — hand-computed values + determinism
# ---------------------------------------------------------------------------


def test_variogram_score_hand_computed_constant_ensemble() -> None:
    """Constant ensemble has E|Xᵢ−Xⱼ|^p = 0, so VS_p = Σᵢⱼ |yᵢ−yⱼ|^{2p}.

    y = (0, 2): p=1 → 2·2² = 8; p=0.5 → 2·(√2)² = 4.
    """
    y = np.array([0.0, 2.0])
    samples = np.zeros((3, 2))
    assert variogram_score(y, samples, p=1.0) == pytest.approx(8.0)
    assert variogram_score(y, samples) == pytest.approx(4.0)  # default p = 0.5


def test_variogram_score_zero_when_pairwise_gaps_match() -> None:
    """y = (0,2) vs ensemble {(0,0),(0,4)}: E|X₁−X₂| = 2 = |y₁−y₂| → VS₁ = 0."""
    y = np.array([0.0, 2.0])
    samples = np.array([[0.0, 0.0], [0.0, 4.0]])
    assert variogram_score(y, samples, p=1.0) == pytest.approx(0.0)


def test_variogram_score_deterministic_value_on_fixed_seed() -> None:
    """Same seed → bit-identical score, and it equals the loop form."""
    y = np.random.default_rng(2).standard_normal(4)
    first = variogram_score(y, np.random.default_rng(5).standard_normal((16, 4)))
    second = variogram_score(y, np.random.default_rng(5).standard_normal((16, 4)))
    assert first == second
    assert math.isfinite(first) and first >= 0.0
    assert first == pytest.approx(
        _brute_force_variogram(y, np.random.default_rng(5).standard_normal((16, 4)), p=0.5)
    )


def test_variogram_score_is_nonnegative_and_penalises_dependence_error() -> None:
    """VS ≥ 0 always; a wrong-ρ copula scores above the planted one."""
    rng = np.random.default_rng(7)
    d, m = 4, 128
    sigma = np.full((d, d), 0.7)
    np.fill_diagonal(sigma, 1.0)
    y = rng.standard_normal(d) @ np.linalg.cholesky(sigma).T
    marginals = [rng.standard_normal(m) for _ in range(d)]
    correct = gaussian_copula_samples(marginals, 0.7, np.random.default_rng(21))
    independent = gaussian_copula_samples(marginals, 0.0, np.random.default_rng(21))
    vs_correct = variogram_score(y, correct)
    vs_independent = variogram_score(y, independent)
    assert vs_correct >= 0.0 and vs_independent >= 0.0
    assert vs_correct < vs_independent


# ---------------------------------------------------------------------------
# gaussian_copula_samples — shape, ranks, seed determinism, fail-closed
# ---------------------------------------------------------------------------


def test_gaussian_copula_samples_shape_and_values_come_from_marginals() -> None:
    rng = np.random.default_rng(3)
    marginals = [rng.standard_normal(20) * (j + 1) for j in range(3)]
    out = gaussian_copula_samples(marginals, 0.5, np.random.default_rng(8))
    assert out.shape == (20, 3)
    for j, x in enumerate(marginals):
        assert set(np.unique(out[:, j]).tolist()) <= set(np.unique(x).tolist())


def test_gaussian_copula_samples_seed_determinism() -> None:
    marginals = [np.arange(16, dtype=np.float64), np.arange(16, dtype=np.float64) * 2.0]
    a = gaussian_copula_samples(marginals, 0.4, np.random.default_rng(13))
    b = gaussian_copula_samples(marginals, 0.4, np.random.default_rng(13))
    c = gaussian_copula_samples(marginals, 0.4, np.random.default_rng(14))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_gaussian_copula_samples_near_one_rho_is_comonotonic() -> None:
    """ρ → 1 → one shared uniform → near-identical rank order across names."""
    rng = np.random.default_rng(17)
    marginals = [rng.standard_normal(64), rng.standard_normal(64) * 5.0 + 1.0]

    def spearman(rho: float) -> float:
        out = gaussian_copula_samples(marginals, rho, np.random.default_rng(19))
        r0 = np.argsort(np.argsort(out[:, 0])).astype(np.float64)
        r1 = np.argsort(np.argsort(out[:, 1])).astype(np.float64)
        return float(np.corrcoef(r0, r1)[0, 1])

    high = spearman(0.9999)
    assert high > 0.98
    assert high > spearman(0.5) > spearman(0.0)


def test_gaussian_copula_samples_rho_one_fails_closed_in_the_cholesky() -> None:
    """ρ = 1 passes the feasibility check but the correlation matrix is
    singular, so the draw fails closed instead of returning a degenerate
    ensemble. Pinned because the guard admits the boundary value."""
    with pytest.raises(np.linalg.LinAlgError):
        gaussian_copula_samples([np.zeros(8), np.zeros(8)], 1.0, np.random.default_rng(0))


def test_gaussian_copula_samples_rank_dependence_tracks_rho() -> None:
    """SYNTHETIC sanity: mean |Spearman| across pairs rises with ρ."""

    def mean_abs_rank_corr(rho: float) -> float:
        marginals = [np.random.default_rng(31 + j).standard_normal(256) for j in range(4)]
        out = gaussian_copula_samples(marginals, rho, np.random.default_rng(23))
        ranks = np.apply_along_axis(lambda col: np.argsort(np.argsort(col)), 0, out)
        corr = np.corrcoef(ranks.T)
        off = corr[~np.eye(4, dtype=bool)]
        return float(np.mean(np.abs(off)))

    assert mean_abs_rank_corr(0.9) > mean_abs_rank_corr(0.0)


def test_gaussian_copula_samples_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="share length"):
        gaussian_copula_samples([np.zeros(8), np.zeros(7)], 0.3, np.random.default_rng(0))


def test_gaussian_copula_samples_rejects_infeasible_equicorrelation() -> None:
    """ρ must satisfy −1/(d−1) < ρ ≤ 1; d=3 → ρ=−0.6 is infeasible."""
    marginals = [np.zeros(8) for _ in range(3)]
    with pytest.raises(ValueError, match="feasible range"):
        gaussian_copula_samples(marginals, -0.6, np.random.default_rng(0))
    with pytest.raises(ValueError, match="feasible range"):
        gaussian_copula_samples(marginals, 1.5, np.random.default_rng(0))
    # The lower edge itself is feasible for d=3 (−1/2 < −0.4).
    assert gaussian_copula_samples(marginals, -0.4, np.random.default_rng(0)).shape == (8, 3)


# ---------------------------------------------------------------------------
# Fail-closed input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y", [np.zeros((2, 2)), np.zeros((3,))])
@pytest.mark.parametrize("samples", [np.zeros(4), np.zeros((4, 3)), np.zeros((2, 3, 1))])
def test_score_shapes_fail_closed(y: NDArray[np.float64], samples: NDArray[np.float64]) -> None:
    """Only (m, d) samples against a matching (d,) observation are accepted."""
    if samples.ndim == 2 and y.ndim == 1 and samples.shape[1] == y.size:
        pytest.skip("shape combination is valid")
    with pytest.raises(ValueError, match=r"\(m, d\)"):
        energy_score(y, samples)
    with pytest.raises(ValueError, match=r"\(m, d\)"):
        variogram_score(y, samples)


def test_empty_sample_vector_fails_closed() -> None:
    """A 1-D empty array is not an (m, d) ensemble → ValueError, not a score."""
    with pytest.raises(ValueError, match=r"\(m, d\)"):
        energy_score(np.zeros(2), np.empty(0))
    with pytest.raises(ValueError, match=r"\(m, d\)"):
        variogram_score(np.zeros(2), np.empty(0))


def test_zero_draw_ensemble_yields_no_finite_score() -> None:
    """(0, d) has no draws: both scores must come back non-finite, never 0.0."""
    empty = np.empty((0, 2))
    y = np.zeros(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es = energy_score(y, empty)
        vs = variogram_score(y, empty)
    assert not math.isfinite(es)
    assert not math.isfinite(vs)


# ---------------------------------------------------------------------------
# Sealed bench payload (the benches_w18 lane this module exists for)
# ---------------------------------------------------------------------------


def test_bench_payload_is_sealed_and_synthetic(bench: dict[str, object]) -> None:
    assert bench["schema"] == ENERGY_SCORE_SCHEMA
    assert bench["kind"] == "energy_score"
    assert bench["data_label"] == "SYNTHETIC"
    assert bench["research_only"] is True
    assert bench["n_reps"] == _BENCH["n_reps"]
    assert bench["m"] == _BENCH["m"]
    assert bench["d"] == _BENCH["d"]
    assert bench["planted_rho"] == pytest.approx(_BENCH["n_names_corr"])
    assert isinstance(bench["interpretation"], str) and bench["interpretation"]


def test_bench_payload_hash_is_reproducible(bench: dict[str, object]) -> None:
    digest = bench["payload_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    body = {k: v for k, v in bench.items() if k != "payload_sha256"}
    assert hash_bytes(canonical_json_bytes(body)) == digest


def test_bench_arms_carry_both_proper_scores(bench: dict[str, object]) -> None:
    arms = bench["arms"]
    assert isinstance(arms, dict)
    assert set(arms) == {"correct", "misspecified", "independent"}
    for name, block in arms.items():
        assert isinstance(block, dict), name
        for score in ("energy_score", "variogram_score"):
            stats = block[score]
            assert isinstance(stats, dict), f"{name}/{score}"
            assert stats["n"] == float(_BENCH["n_reps"])
            assert math.isfinite(float(stats["mean"])), f"{name}/{score}"
            assert float(stats["std"]) >= 0.0


def test_bench_flags_are_booleans_and_payload_is_honest(
    bench: dict[str, object],
) -> None:
    assert isinstance(bench["correct_copula_best_es"], bool)
    assert isinstance(bench["correct_copula_best_vs"], bool)
    # Proper scores only: no headline performance-ratio key anywhere in the blob.
    assert family_blob_forbidden_metrics_absent(bench)
    text = json.dumps(bench, sort_keys=True, default=str).lower()
    for token in ("sharpe", "sortino", "calmar", "pnl", "nav"):
        assert token not in text, token


def test_bench_is_seed_deterministic(bench: dict[str, object]) -> None:
    again = energy_score_bench(**_BENCH)  # type: ignore[arg-type]
    assert again["payload_sha256"] == bench["payload_sha256"]
    other = energy_score_bench(**{**_BENCH, "seed": 12})  # type: ignore[arg-type]
    assert other["payload_sha256"] != bench["payload_sha256"]


def test_bench_ranks_correct_copula_first_at_the_wired_size() -> None:
    """The science claim the lane ships with, at the reduced size
    :mod:`quant_fund.research.benches_w18` runs it at (seed included), so this
    unit test and the scorecard wrapper cannot drift apart."""
    payload = energy_score_bench(n_reps=12, m=128, d=5, n_names_corr=0.6, seed=20261004)
    assert payload["correct_copula_best_es"] is True
    assert payload["correct_copula_best_vs"] is True
    arms = payload["arms"]
    assert isinstance(arms, dict)
    es = {k: float(v["energy_score"]["mean"]) for k, v in arms.items()}  # type: ignore[index]
    vs = {k: float(v["variogram_score"]["mean"]) for k, v in arms.items()}  # type: ignore[index]
    # Lower is better for both proper scores.
    assert es["correct"] < es["independent"]
    assert vs["correct"] < vs["independent"]
    # VS isolates dependence structure, so it separates more sharply than ES.
    assert (vs["independent"] - vs["correct"]) > (es["independent"] - es["correct"]) > 0.0
