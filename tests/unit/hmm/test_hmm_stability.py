"""hmm_stability: refit stability of a learned regime decomposition.

Covers the lane's four contracts: planted-chain recovery across bootstrap
restarts, label-permutation invariance of the Hungarian state matching,
seeded determinism, and seal determinism of the ``hmm_stability.v1``
receipt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from quant_fund.hmm.discrete import DiscreteHMM, baum_welch
from quant_fund.hmm.hmm_stability import (
    _match_states,
    _relabel,
    _simulate,
    hmm_stability_bench,
    refit_stability,
    stability_report,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

pytestmark = pytest.mark.synthetic

PLANTED = DiscreteHMM(
    np.array([[0.93, 0.07], [0.09, 0.91]], dtype=float),
    np.array([[0.60, 0.40, 0.00, 0.00], [0.00, 0.00, 0.40, 0.60]], dtype=float),
    np.array([0.5, 0.5], dtype=float),
)


def _best_of(obs: np.ndarray, seed: int, n_inits: int, n_iter: int) -> DiscreteHMM:
    """Best-of-K seeded Baum–Welch inits by final likelihood."""
    best: DiscreteHMM | None = None
    best_ll = -math.inf
    for child in np.random.SeedSequence(int(seed)).spawn(int(n_inits)):
        model, history = baum_welch(
            obs,
            n_states=2,
            n_obs=4,
            n_iter=int(n_iter),
            seed=int(child.generate_state(1)[0]),
            log_space=True,
        )
        if history[-1] > best_ll:
            best, best_ll = model, history[-1]
    assert best is not None
    return best


def test_label_permuted_fit_matches_with_zero_dev() -> None:
    # Transition matrix is nearly swap-symmetric on purpose: a naive
    # row-wise Frobenius cost on A alone cannot tell the swap apart —
    # matching must be driven by the permutation-invariant signature + B.
    perm = np.array([1, 0], dtype=np.intp)
    swapped = _relabel(PLANTED, perm)
    recovered = _match_states(PLANTED, swapped)
    assert recovered.tolist() == [1, 0]
    matched = _relabel(swapped, recovered)
    assert np.allclose(matched.A, PLANTED.A, atol=1e-12)
    assert np.allclose(matched.B, PLANTED.B, atol=1e-12)
    assert np.allclose(matched.pi, PLANTED.pi, atol=1e-12)


def test_identity_match_on_unpermuted_fit() -> None:
    assert _match_states(PLANTED, PLANTED).tolist() == [0, 1]


def test_match_rejects_shape_mismatch() -> None:
    other = DiscreteHMM(
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[0.5, 0.5], [0.2, 0.8]]),
        np.array([0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="observation symbols"):
        _match_states(PLANTED, other)


def test_planted_chain_recovers_across_restarts() -> None:
    rng = np.random.default_rng(3)
    truth, stream = _simulate(PLANTED, 800, rng)

    def fit(obs: np.ndarray, seed: int) -> DiscreteHMM:
        return _best_of(obs, seed, n_inits=4, n_iter=25)

    stability = refit_stability(fit, stream, n_restarts=3, seed=3)
    report = stability_report(stability, max_dev=0.25)
    ref_perm = _match_states(PLANTED, stability.reference)
    ref_in_truth = _relabel(stability.reference, ref_perm)
    assert np.linalg.norm(ref_in_truth.A - PLANTED.A, "fro") < 0.15
    assert np.linalg.norm(ref_in_truth.B - PLANTED.B, "fro") < 0.15
    assert report["verdict"] == "stable"
    assert all(row["dev"] < 0.25 for row in report["restarts"])
    assert all(s["assignment_consistency"] == 1.0 for s in report["states"])


def test_refit_stability_is_seed_deterministic() -> None:
    rng = np.random.default_rng(5)
    _, stream = _simulate(PLANTED, 300, rng)

    def fit(obs: np.ndarray, seed: int) -> DiscreteHMM:
        model, _ = baum_welch(obs, n_states=2, n_obs=4, n_iter=4, seed=seed, log_space=True)
        return model

    first = refit_stability(fit, stream, n_restarts=3, seed=11)
    second = refit_stability(fit, stream, n_restarts=3, seed=11)
    assert first.assignments == second.assignments
    for a, b in zip(first.matched, second.matched, strict=True):
        assert np.array_equal(a.A, b.A)
        assert np.array_equal(a.B, b.B)
        assert np.array_equal(a.pi, b.pi)


def test_random_fit_fn_reports_unstable() -> None:
    rng = np.random.default_rng(5)
    _, stream = _simulate(PLANTED, 200, rng)

    def wandering(obs: np.ndarray, seed: int) -> DiscreteHMM:
        gen = np.random.default_rng(seed)
        a = gen.dirichlet(np.ones(2), size=2)
        b = gen.dirichlet(np.ones(4), size=2)
        return DiscreteHMM(a, b, np.array([0.5, 0.5]))

    stability = refit_stability(wandering, stream, n_restarts=4, seed=11)
    report = stability_report(stability, max_dev=0.05)
    assert report["verdict"] == "unstable"
    assert report["stable_share"] == 0.0


def test_input_validation() -> None:
    def fit(obs: np.ndarray, seed: int) -> DiscreteHMM:
        return PLANTED

    with pytest.raises(ValueError, match="non-empty 1-D"):
        refit_stability(fit, np.array([], dtype=np.intp), 2, 0)
    with pytest.raises(ValueError, match="n_restarts"):
        refit_stability(fit, np.array([0, 1, 0], dtype=np.intp), 0, 0)


def test_bench_receipt_seal_and_immutable_publish(tmp_path: Path) -> None:
    kwargs = dict(n_steps=1000, n_restarts=4, n_iter=25, n_inits=4)
    first = hmm_stability_bench(13, receipts_dir=tmp_path, **kwargs)
    # Identical re-publish is a no-op under publish_text_once; differing
    # content would raise — so a clean second run proves byte determinism.
    second = hmm_stability_bench(13, receipts_dir=tmp_path, **kwargs)
    assert first["receipt_sha256"] == second["receipt_sha256"]

    body = {k: v for k, v in first.items() if k != "receipt_sha256"}
    assert hash_bytes(canonical_json_bytes(body)) == first["receipt_sha256"]
    text = (tmp_path / "hmm_stability.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["receipt_sha256"] == first["receipt_sha256"]
    assert first["schema"] == "hmm_stability.v1"
    assert first["kind"] == "hmm_stability"
    assert first["data_label"] == "SYNTHETIC"
    assert first["live_pnl_claim"] is False
    assert first["research_only"] is True


def test_bench_recovers_planted_decomposition(tmp_path: Path) -> None:
    receipt = hmm_stability_bench(
        13,
        n_steps=1000,
        n_restarts=4,
        n_iter=25,
        n_inits=4,
        receipts_dir=tmp_path,
    )
    assert receipt["verdict"] == "stable"
    assert receipt["report"]["stable_share"] >= 0.9
    assert receipt["recovery"]["state_path_agreement"] > 0.9
    assert receipt["recovery"]["truth_dev_a"] < 0.15
    assert receipt["recovery"]["truth_dev_b"] < 0.15
