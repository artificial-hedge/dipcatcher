"""Equivalence locks for the McCabe splits.

Each number below was measured on the pre-split functions. A helper that
drops a branch, reorders a draw, or changes a clip fails one of these.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from quant_fund.hmm.discrete import DiscreteHMM, _logsumexp_row, log_forward
from quant_fund.leakage import scan_paths
from quant_fund.leakage.patterns import find_forbidden_headline
from quant_fund.models.graded_irt import grm_jml
from quant_fund.models.pair_vine_copula import gas_copula_filter, gas_copula_fit
from quant_fund.models.pmcmc_sv import simulate_sv, sv_pf_loglik
from quant_fund.models.quest import quest_forward
from quant_fund.research.capability_value import make_synthetic_trajectory


def _reference_log_forward(
    log_a: np.ndarray, log_b: np.ndarray, log_pi: np.ndarray, obs: list[int]
) -> tuple[np.ndarray, float]:
    """SciPy log-sum-exp forward, independent of the numba kernel."""
    alpha = np.empty((len(obs), log_a.shape[0]))
    alpha[0] = log_pi + log_b[:, obs[0]]
    for t in range(1, len(obs)):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + log_a, axis=0) + log_b[:, obs[t]]
    return alpha, float(logsumexp(alpha[-1]))


def test_log_forward_matches_scipy_and_the_pre_split_path() -> None:
    model = DiscreteHMM(
        A=np.array([[0.7, 0.3], [0.4, 0.6]]),
        B=np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]),
        pi=np.array([0.6, 0.4]),
    )
    obs = [0, 1, 2, 1, 0, 2, 2, 1]
    alpha, ll = log_forward(model, obs)
    eps = 1e-300
    ref_alpha, ref_ll = _reference_log_forward(
        np.log(np.clip(model.A, eps, None)),
        np.log(np.clip(model.B, eps, None)),
        np.log(np.clip(model.pi, eps, None)),
        obs,
    )
    assert alpha == pytest.approx(ref_alpha, rel=1e-12, abs=1e-12)
    assert ll == pytest.approx(ref_ll, rel=1e-12)
    assert ll == pytest.approx(-8.890717308305, abs=1e-12)
    assert alpha[-1].tolist() == pytest.approx([-9.5627389511, -9.6054459643], abs=1e-9)
    assert _logsumexp_row(np.array([0.0, 0.0])) == pytest.approx(np.log(2.0))


def test_forbidden_headline_keeps_both_stages_sorted() -> None:
    # Proximity hit on sharpe, paragraph-density hit on sortino.
    text = "sharpe 1.2\n\nsortino is discussed here" + "x" * 120 + "see table 4 and figure 7"
    assert find_forbidden_headline(text) == ["sharpe", "sortino"]
    distant = "sharpe is discussed here" + "x" * 120 + "see table 4 at the end"
    assert find_forbidden_headline(distant) == []


def test_lh009_branches_and_lh014_still_fire(tmp_path: Path) -> None:
    source = (
        "def leaky(close):\n"
        "    return close.shift(-1)\n"
        "\n"
        "def caller(close):\n"
        "    return leaky(close)\n"
        "\n"
        "def load(path):\n"
        "    import polars as pl\n"
        "    direct = pl.read_parquet(path)\n"
        "    audit = pl.history()\n"
        "    dynamic = getattr(pl, 'read_parquet')(path)\n"
        "    query = pl.sql(\"select * from read_parquet('x')\")\n"
        "    return direct, audit, dynamic, query\n"
    )
    path = tmp_path / "leak.py"
    path.write_text(source, encoding="utf-8")
    findings = scan_paths([path]).findings
    lh009 = [f.message for f in findings if f.rule_id == "LH009"]
    assert any("direct parquet read" in message for message in lh009)
    assert any("history()" in message for message in lh009)
    assert any("dynamic getattr" in message for message in lh009)
    assert any("embedded in a sql" in message for message in lh009)
    lh014 = [f for f in findings if f.rule_id == "LH014"]
    assert lh014 and "leaky" in lh014[0].message and "LH001" in lh014[0].message


def test_gas_filter_and_fit_match_the_pre_split_path() -> None:
    rng = np.random.default_rng(0)
    u = norm.cdf(rng.normal(size=(40, 2)))
    rho, _kappa, scores = gas_copula_filter(u, 0.0, 0.1, 0.8)
    assert float(rho[0]) == pytest.approx(0.1732521048, abs=1e-9)
    assert float(rho[-1]) == pytest.approx(-0.2840163409, abs=1e-9)
    assert float(scores[5]) == pytest.approx(0.1021902682, abs=1e-9)
    fit = gas_copula_fit(u)
    assert float(fit["omega"]) == pytest.approx(0.0003151088, abs=1e-9)
    assert float(fit["alpha"]) == pytest.approx(0.0, abs=1e-12)
    assert float(fit["beta"]) == pytest.approx(0.9699999964, abs=1e-9)
    assert float(fit["loglik"]) == pytest.approx(0.8219581859, abs=1e-9)
    assert float(fit["converged"]) == pytest.approx(1.0)


def test_sv_bootstrap_loglik_matches_the_pre_split_value() -> None:
    returns, _vols = simulate_sv(30, seed=1)
    got = sv_pf_loglik(returns, -4.0, 0.95, 0.15, n_particles=20, seed=2)
    assert got == pytest.approx(19.49741340701, abs=1e-9)


def test_quest_forward_lam_matches_the_pre_split_spectrum() -> None:
    lam = quest_forward(np.array([1.0, 1.0, 2.0, 3.0]), 8).lam
    assert lam.tolist() == pytest.approx(
        [0.3266517136, 0.8593876886, 1.8466522022, 3.9699340656], abs=1e-9
    )


def test_grm_jml_matches_the_pre_split_fit() -> None:
    resp = np.random.default_rng(3).integers(0, 3, size=(8, 4))
    out = grm_jml(resp, n_iter=2)
    assert out["theta"].tolist() == pytest.approx(
        [
            -1.60170781,
            0.50549068,
            -0.76307687,
            -0.44581356,
            0.00322262,
            0.17256449,
            0.03967646,
            2.08964401,
        ],
        abs=1e-8,
    )
    assert float(out["loglik"]) == pytest.approx(-113.05081605, abs=1e-8)


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_synthetic_trajectory_matches_the_pre_split_draw() -> None:
    traj = make_synthetic_trajectory(n_periods=30, n_signals=4, n_persistent_edge=2, seed=7)
    assert traj.signal_matrix[0].tolist() == pytest.approx(
        [0.0102728928, -0.1599907665, 0.0063634206, -0.0593597264], abs=1e-9
    )
    assert traj.true_edge_matrix[0, 0] == pytest.approx(0.00011596, abs=1e-8)
    assert traj.true_edge_matrix[-1, 0] == pytest.approx(0.00986834, abs=1e-8)
    assert np.all(traj.true_edge_matrix[:, 2] == 0.0)
    assert traj.states[0].hist[0].decision == "rejected"
    assert traj.states[-1].hist[-1].rank_ic == pytest.approx(-0.0456024529, abs=1e-9)


def test_app_reuse_swaps_create_app_without_a_suppression() -> None:
    import fx1.serve.api as api_mod
    from fx1.serve.api_audit import _OiAppReuse

    original = api_mod.create_app
    with _OiAppReuse():
        assert api_mod.create_app is not original
        assert callable(api_mod.create_app)
    assert api_mod.create_app is original
