"""Tests for quant_fund.models.deep_bsde — the decoupled deep BSDE solver.

References (verified against the fetched papers): E, Han & Jentzen (2017,
Communications in Mathematics and Statistics 5(4):349–380, arXiv:1706.04702 —
Framework 3.2, terminal-MSE loss eq. (12), Burgers-type Lemma 4.3 with an
explicit solution, Allen-Cahn Subsection 4.2); Han, Jentzen & E (2018, PNAS
115(34):8505–8510, arXiv:1707.02568); Han & Long (2020, Probability,
Uncertainty and Quantitative Risk 5:5, arXiv:1811.01165 — objective (2.4) is
the same terminal MSE with no running term; Theorem 1 makes the minimized
loss an a posteriori error indicator); Pardoux & Peng (1990, Systems &
Control Letters 14(1):55–61); Black & Scholes (1973) / Merton (1973) for the
analytic call-price target.

All data here is SYNTHETIC (seeded Brownian simulations of PDEs with known
solutions) — algorithmic correctness evidence, never market evidence; no
Sharpe/P&L headline, no live-trading claims. Torch tests skip cleanly when
the nn extra is absent; the numpy core (simulation, rollout, loss) is tested
torch-free.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

from quant_fund.models import deep_bsde as db


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="deep BSDE training requires the nn extra (torch)"
)

# Black-Scholes validation setting (normalized spot so the learned theta_0
# starts within one Adam travel budget of the target price).
S0, STRIKE, RATE, SIGMA, MATURITY, BS_STEPS = 1.0, 1.0, 0.05, 0.4, 1.0, 20


def _toy_problem(
    *,
    dim: int = 1,
    n_steps: int = 2,
    horizon: float = 1.0,
    x0: np.ndarray | None = None,
    mu=None,
    sigma=None,
    f=None,
    g=None,
    name: str = "toy",
) -> db.BSDEProblem:
    """Minimal hand-checkable problem: dX = dW, f = 0, g(x) = x_0 coordinate."""
    d = dim
    eye = np.eye(d)
    return db.BSDEProblem(
        name=name,
        dim=d,
        horizon=horizon,
        n_steps=n_steps,
        x0=np.zeros(d) if x0 is None else x0,
        mu=mu if mu is not None else (lambda t, x: 0.0 * x),
        sigma=sigma
        if sigma is not None
        else (lambda t, x: np.broadcast_to(eye, (x.shape[0], d, d))),
        f=f if f is not None else (lambda t, x, y, z: 0.0 * y),
        g=g if g is not None else (lambda x: x[:, 0]),
    )


def _params_closed_form(dim: int, n_steps: int, hidden_width: int) -> int:
    """1 (theta_0) + dim (Z_0) + (N-1) per-layer MLPs of width hidden_width."""
    h = hidden_width
    per_net = dim * h + h + (h * h + h) + (h * dim + dim)
    return 1 + dim + (n_steps - 1) * per_net


# ---------------------------------------------------------------------------
# numpy core (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_brownian_increments_seeded_shape_stats_determinism() -> None:
    dw = db.sample_brownian_increments(20_000, 3, 5, horizon=1.0, seed=7)
    assert dw.shape == (20_000, 5, 3)
    # increments are N(0, dt) with dt = T/N = 0.2
    assert abs(float(dw.std()) - math.sqrt(0.2)) < 0.01
    assert abs(float(dw.mean())) < 0.01
    a = db.sample_brownian_increments(4, 2, 3, horizon=0.6, seed=1)
    b = db.sample_brownian_increments(4, 2, 3, horizon=0.6, seed=1)
    c = db.sample_brownian_increments(4, 2, 3, horizon=0.6, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_brownian_increments_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_paths"):
        db.sample_brownian_increments(0, 1, 2, horizon=1.0)
    with pytest.raises(ValueError, match="dim"):
        db.sample_brownian_increments(2, 0, 2, horizon=1.0)
    with pytest.raises(ValueError, match="n_steps"):
        db.sample_brownian_increments(2, 1, 0, horizon=1.0)
    with pytest.raises(ValueError, match="horizon"):
        db.sample_brownian_increments(2, 1, 2, horizon=0.0)
    with pytest.raises(ValueError, match="horizon"):
        db.sample_brownian_increments(2, 1, 2, horizon=float("nan"))


def test_euler_sde_paths_unit_diffusion_is_exact_cumsum() -> None:
    # mu = 0, sigma = I: Euler-Maruyama is exact, X_{n+1} - X_n == dW_n.
    prob = _toy_problem(dim=2, n_steps=4, horizon=1.0, x0=np.array([1.0, -2.0]))
    dw = db.sample_brownian_increments(6, 2, 4, horizon=1.0, seed=3)
    x = db.euler_sde_paths(prob, dw)
    assert x.shape == (6, 5, 2)
    assert np.all(x[:, 0, :] == prob.x0)
    assert np.allclose(x[:, 1:, :] - x[:, :-1, :], dw)
    assert np.all(np.isfinite(x))


def test_euler_sde_paths_affine_drift_is_exact() -> None:
    # sigma = 0, mu = c: X_T == x0 + c*T exactly.
    c = np.array([0.5, -1.5])
    prob = _toy_problem(
        dim=2,
        n_steps=5,
        horizon=2.0,
        mu=lambda t, x: np.broadcast_to(c, x.shape),
        sigma=lambda t, x: np.zeros((x.shape[0], 2, 2)),
    )
    dw = db.sample_brownian_increments(3, 2, 5, horizon=2.0, seed=4)
    x = db.euler_sde_paths(prob, dw)
    assert np.allclose(x[:, -1, :], prob.x0 + c * 2.0)


def test_euler_sde_paths_gbm_first_moment_and_positivity() -> None:
    prob = db.black_scholes_call_problem(
        s0=100.0, strike=100.0, r=RATE, sigma=SIGMA, horizon=MATURITY, n_steps=BS_STEPS
    )
    dw = db.sample_brownian_increments(50_000, 1, BS_STEPS, horizon=MATURITY, seed=5)
    x = db.euler_sde_paths(prob, dw)
    terminal = x[:, -1, 0]
    assert np.all(terminal > 0.0)  # Euler GBM cannot cross zero at this dt/sigma
    # E[X_T] = s0 e^{rT} up to O(dt) Euler weak bias + MC error (~0.2 at 50k paths)
    assert abs(float(terminal.mean()) - 100.0 * math.exp(RATE * MATURITY)) < 1.0


def test_euler_sde_paths_fail_closed() -> None:
    prob = _toy_problem(dim=2, n_steps=3)
    good = db.sample_brownian_increments(4, 2, 3, horizon=1.0, seed=0)
    with pytest.raises(ValueError, match=r"dw must have shape"):
        db.euler_sde_paths(prob, good[:, :2, :])
    with pytest.raises(ValueError, match=r"dw must have shape"):
        db.euler_sde_paths(prob, np.zeros((4, 3, 1)))
    with pytest.raises(ValueError, match="at least one path"):
        db.euler_sde_paths(prob, np.zeros((0, 3, 2)))
    with pytest.raises(ValueError, match="finite"):
        db.euler_sde_paths(prob, np.full((4, 3, 2), np.nan))
    with pytest.raises(TypeError, match="BSDEProblem"):
        db.euler_sde_paths("not-a-problem", good)  # type: ignore[arg-type]
    # non-broadcastable mu output fails closed with a step-specific message
    bad = _toy_problem(dim=2, n_steps=3, mu=lambda t, x: np.zeros((x.shape[0], 5)))
    with pytest.raises(ValueError, match="broadcast"):
        db.euler_sde_paths(bad, good)


def test_bsde_rollout_hand_computed_and_terminal_loss() -> None:
    """f = 0, constant Z = 0.5, dW = (1, 2), y0 = 1: Y = (1, 1.5, 2.5).

    X = (0, 1, 3) for unit diffusion from x0 = 0; g(x) = x_0 gives the
    terminal target 3, so loss = (2.5 - 3)^2 = 0.25 (hand-computed).
    """
    prob = _toy_problem(dim=1, n_steps=2, horizon=1.0)
    dw = np.array([[[1.0], [2.0]]])
    x = db.euler_sde_paths(prob, dw)
    z = np.full((1, 2, 1), 0.5)
    y = db.bsde_rollout(prob, x, dw, 1.0, z)
    assert np.allclose(y, [[1.0, 1.5, 2.5]])
    assert db.bsde_terminal_loss(prob, x, dw, 1.0, z) == pytest.approx(0.25)
    # y-dependent driver: f = -y gives Y_{n+1} = Y_n (1 + dt), dt = 0.5.
    prob_r = _toy_problem(dim=1, n_steps=2, horizon=1.0, f=lambda t, x, y, z: -1.0 * y)
    y_r = db.bsde_rollout(prob_r, x, np.zeros_like(dw), 2.0, z)
    assert np.allclose(y_r, [[2.0, 3.0, 4.5]])


def test_bsde_rollout_fail_closed() -> None:
    prob = _toy_problem(dim=1, n_steps=2)
    dw = db.sample_brownian_increments(2, 1, 2, horizon=1.0, seed=0)
    x = db.euler_sde_paths(prob, dw)
    z = np.zeros((2, 2, 1))
    with pytest.raises(ValueError, match="x_paths must have shape"):
        db.bsde_rollout(prob, x[:, :-1, :], dw, 0.0, z)
    with pytest.raises(ValueError, match="dw must have shape"):
        db.bsde_rollout(prob, x, dw[:, :1, :], 0.0, z)
    with pytest.raises(ValueError, match="z_values must have shape"):
        db.bsde_rollout(prob, x, dw, 0.0, np.zeros((2, 1, 1)))
    with pytest.raises(ValueError, match="y0 must be a scalar"):
        db.bsde_rollout(prob, x, dw, np.array([1.0, 2.0, 3.0]), z)
    with pytest.raises(ValueError, match="finite"):
        db.bsde_rollout(prob, x, dw, float("nan"), z)
    # f returning the wrong number of values fails closed
    bad_f = _toy_problem(dim=1, n_steps=2, f=lambda t, x, y, z: np.ones(3))
    with pytest.raises(ValueError, match="must return n_paths"):
        db.bsde_rollout(bad_f, x, dw, 0.0, z)


def test_problem_construction_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        _toy_problem(name="")
    with pytest.raises(ValueError, match="dim"):
        db.BSDEProblem(
            name="x",
            dim=0,
            horizon=1.0,
            n_steps=2,
            x0=np.zeros(0),
            mu=lambda t, x: x,
            sigma=lambda t, x: x,
            f=lambda t, x, y, z: y,
            g=lambda x: x,
        )
    with pytest.raises(ValueError, match="horizon"):
        _toy_problem(horizon=-1.0)
    with pytest.raises(ValueError, match="n_steps"):
        _toy_problem(n_steps=0)
    with pytest.raises(ValueError, match="x0 must have length"):
        _toy_problem(dim=2, x0=np.zeros(3))
    with pytest.raises(ValueError, match="x0 entries must be finite"):
        _toy_problem(x0=np.array([np.nan]))
    with pytest.raises(ValueError, match="f must be callable"):
        db.BSDEProblem(
            name="x",
            dim=1,
            horizon=1.0,
            n_steps=2,
            x0=np.zeros(1),
            mu=lambda t, x: x,
            sigma=lambda t, x: x,
            f=0.5,
            g=lambda x: x,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exact_u must be callable"):
        db.BSDEProblem(
            name="x",
            dim=1,
            horizon=1.0,
            n_steps=2,
            x0=np.zeros(1),
            mu=lambda t, x: x,
            sigma=lambda t, x: x,
            f=lambda t, x, y, z: y,
            g=lambda x: x,
            exact_u=1.0,  # type: ignore[arg-type]
        )


def test_problem_grid_and_step_size() -> None:
    prob = _toy_problem(dim=2, n_steps=4, horizon=2.0)
    assert prob.step_size() == pytest.approx(0.5)
    assert np.allclose(prob.time_grid(), [0.0, 0.5, 1.0, 1.5, 2.0])
    assert isinstance(prob.x0, np.ndarray) and prob.x0.dtype == np.float64


def test_burgers_hopf_driver_matches_paper_subsection_4_5_form() -> None:
    """Defaults alpha = d^2, kappa = 1/d reduce the driver to the paper's (57).

    E-Han-Jentzen (2017) Subsection 4.5: f(t,x,y,z) = (y - (2+d)/(2d)) sum_i z_i.
    """
    d = 3
    prob = db.burgers_hopf_problem(dim=d, horizon=0.3, n_steps=6)
    y = np.array([0.3, 0.7])
    z = np.array([[1.0, -2.0, 0.5], [0.1, 0.2, 0.3]])
    expected = (y - (2.0 + d) / (2.0 * d)) * z.sum(axis=-1)
    got = np.asarray(prob.f(0.0, np.zeros((2, d)), y, z), dtype=float)
    assert np.allclose(got, expected)
    assert prob.exact_u is not None


def test_burgers_hopf_exact_solution_satisfies_pde_finite_differences() -> None:
    """The claimed closed form really solves the PDE (torch-free FD check).

    u(t,x) = exp(t + kappa sum x_i)/(1 + exp(...)) must satisfy
    u_t + (alpha/2) Laplacian(u) + f(t, x, u, sqrt(alpha) grad u) = 0
    (E-Han-Jentzen 2017, Lemma 4.3).
    """
    alpha, kappa, d = 4.0, 0.5, 2
    prob = db.burgers_hopf_problem(dim=d, horizon=0.3, n_steps=6, alpha=alpha, kappa=kappa)
    u = prob.exact_u
    assert u is not None
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(12, d))
    ts = rng.uniform(0.05, 0.25, size=12)
    ht, hx = 1e-4, 1e-3
    eye = np.eye(d)
    worst = 0.0
    for t, x in zip(ts, pts, strict=True):
        x1 = x[None, :]
        u_t = (u(t + ht, x1)[0] - u(t - ht, x1)[0]) / (2 * ht)
        u_c = float(u(t, x1)[0])
        grad = np.array(
            [
                (u(t, (x + hx * e)[None, :])[0] - u(t, (x - hx * e)[None, :])[0]) / (2 * hx)
                for e in eye
            ]
        )
        lap = sum(
            (u(t, (x + hx * e)[None, :])[0] - 2 * u_c + u(t, (x - hx * e)[None, :])[0]) / hx**2
            for e in eye
        )
        z = (math.sqrt(alpha) * grad)[None, :]
        f_val = float(np.asarray(prob.f(float(t), x1, np.array([u_c]), z), dtype=float)[0])
        worst = max(worst, abs(u_t + 0.5 * alpha * lap + f_val))
    assert worst < 1e-5
    # u(0, 0) = 1/2 exactly in every dimension, and exact_initial_value sees it
    assert db.exact_initial_value(prob) == pytest.approx(0.5)
    for dim in (1, 7, 10):
        p = db.burgers_hopf_problem(dim=dim, horizon=0.3, n_steps=4)
        assert db.exact_initial_value(p) == pytest.approx(0.5)


def test_black_scholes_analytic_price_satisfies_pde_finite_differences() -> None:
    """The analytic call price solves u_t + r S u_S + (1/2) sig^2 S^2 u_SS - r u = 0."""
    prob = db.black_scholes_call_problem(
        s0=S0, strike=STRIKE, r=RATE, sigma=SIGMA, horizon=MATURITY, n_steps=BS_STEPS
    )
    u = prob.exact_u
    assert u is not None
    ht, hs = 1e-4, 1e-3
    worst = 0.0
    for t in (0.2, 0.5, 0.8):
        for s in (0.8, 1.0, 1.3):
            x = np.array([[s]])
            u_t = (u(t + ht, x)[0] - u(t - ht, x)[0]) / (2 * ht)
            u_c = float(u(t, x)[0])
            u_s = (u(t, np.array([[s + hs]]))[0] - u(t, np.array([[s - hs]]))[0]) / (2 * hs)
            u_ss = (u(t, np.array([[s + hs]]))[0] - 2 * u_c + u(t, np.array([[s - hs]]))[0]) / hs**2
            resid = u_t + RATE * s * u_s + 0.5 * SIGMA**2 * s**2 * u_ss - RATE * u_c
            worst = max(worst, abs(resid))
    assert worst < 1e-4


def test_black_scholes_price_matches_feynman_kac_monte_carlo() -> None:
    """Analytic price == e^{-rT} E[(S_T - K)+] under exact log-normal sampling."""
    analytic = db.black_scholes_call_price(
        s0=S0, strike=STRIKE, r=RATE, sigma=SIGMA, horizon=MATURITY
    )
    rng = np.random.default_rng(11)
    n = 400_000
    z = rng.standard_normal(n)
    log_st = math.log(S0) + (RATE - 0.5 * SIGMA**2) * MATURITY + SIGMA * math.sqrt(MATURITY) * z
    payoff = np.maximum(np.exp(log_st) - STRIKE, 0.0)
    disc = math.exp(-RATE * MATURITY)
    mc = disc * float(payoff.mean())
    mc_se = disc * float(payoff.std()) / math.sqrt(n)
    assert abs(mc - analytic) < 6.0 * mc_se + 1e-4
    # intrinsic-value lower bound and fail-closed edges
    assert analytic > max(S0 - STRIKE * math.exp(-RATE * MATURITY), 0.0)
    with pytest.raises(ValueError, match="s0"):
        db.black_scholes_call_price(s0=0.0, strike=STRIKE, r=RATE, sigma=SIGMA, horizon=MATURITY)
    with pytest.raises(ValueError, match="strike"):
        db.black_scholes_call_price(s0=S0, strike=-1.0, r=RATE, sigma=SIGMA, horizon=MATURITY)
    with pytest.raises(ValueError, match="sigma"):
        db.black_scholes_call_price(s0=S0, strike=STRIKE, r=RATE, sigma=0.0, horizon=MATURITY)
    with pytest.raises(ValueError, match="horizon"):
        db.black_scholes_call_price(s0=S0, strike=STRIKE, r=RATE, sigma=SIGMA, horizon=0.0)
    with pytest.raises(ValueError, match="r must be finite"):
        db.black_scholes_call_price(
            s0=S0, strike=STRIKE, r=float("nan"), sigma=SIGMA, horizon=MATURITY
        )


def test_allen_cahn_problem_is_the_paper_benchmark_without_exact_solution() -> None:
    """E-Han-Jentzen Subsection 4.2: f = y - y^3, g = 1/(2 + 0.4 ||x||^2).

    No closed form exists (the paper's d=100 reference 0.052802 comes from
    branching diffusion); this only pins the benchmark's shape and driver.
    """
    prob = db.allen_cahn_problem(dim=3, horizon=0.3, n_steps=5)
    assert prob.exact_u is None
    with pytest.raises(ValueError, match="exact_u"):
        db.exact_initial_value(prob)
    y = np.array([0.4, -0.2])
    z = np.zeros((2, 3))
    got = np.asarray(prob.f(0.0, np.zeros((2, 3)), y, z), dtype=float)
    assert np.allclose(got, y - y**3)
    x = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.allclose(np.asarray(prob.g(x), dtype=float), 1.0 / (2.0 + 0.4 * (x**2).sum(-1)))
    dw = db.sample_brownian_increments(8, 3, 5, horizon=0.3, seed=2)
    xp = db.euler_sde_paths(prob, dw)
    loss = db.bsde_terminal_loss(prob, xp, dw, 0.5, np.zeros((8, 5, 3)))
    assert np.isfinite(loss) and loss >= 0.0


def test_module_imports_without_torch_and_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No top-level torch import; training entry points fail closed with guidance."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_db_no_torch", db.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_db_no_torch", probe)  # dataclasses resolves via sys.modules
    spec.loader.exec_module(probe)  # must import cleanly without torch
    # the torch-free core stays fully usable
    prob = probe.burgers_hopf_problem(dim=2, horizon=0.3, n_steps=6)
    dw = probe.sample_brownian_increments(8, 2, 6, horizon=0.3, seed=0)
    x = probe.euler_sde_paths(prob, dw)
    loss = probe.bsde_terminal_loss(prob, x, dw, 0.0, np.zeros((8, 6, 2)))
    assert np.isfinite(loss)
    assert probe.exact_initial_value(prob) == pytest.approx(0.5)
    assert (
        probe.black_scholes_call_price(s0=S0, strike=STRIKE, r=RATE, sigma=SIGMA, horizon=MATURITY)
        > 0.0
    )
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.deep_bsde_solve(prob, epochs=2)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.validate_black_scholes_call(epochs=2)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.validate_burgers_hopf(dim=1, epochs=2)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.validate_dimension_scaling(dims=(1, 2), epochs=2)


# ---------------------------------------------------------------------------
# torch lane (skipped when the nn extra is absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bs_call_metrics() -> dict[str, float]:
    """One seeded Black-Scholes validation run shared by the metric tests."""
    return db.validate_black_scholes_call(
        s0=S0,
        strike=STRIKE,
        r=RATE,
        sigma=SIGMA,
        horizon=MATURITY,
        n_steps=BS_STEPS,
        epochs=500,
        lr=1e-2,
        n_paths=512,
        eval_paths=1024,
        seed=0,
    )


@pytest.fixture(scope="module")
def burgers_d1() -> tuple[db.BSDEProblem, db.DeepBSDEResult]:
    """One seeded Burgers-Hopf d=1 run (known solution u(0,0) = 1/2)."""
    prob = db.burgers_hopf_problem(dim=1, horizon=0.3, n_steps=12)
    res = db.deep_bsde_solve(prob, epochs=450, lr=1e-2, n_paths=512, eval_paths=1024, seed=0)
    return prob, res


@requires_torch
def test_black_scholes_call_y0_matches_analytic_price(bs_call_metrics: dict[str, float]) -> None:
    """Linear BSDE recovers the analytic Black-Scholes price within tolerance.

    The absolute tolerance covers Euler time-discretization bias (O(T/N)),
    network-approximation error, and finite-training error; the observed
    error is reported in the metrics dict, never hidden. SYNTHETIC
    correctness check — not market evidence, not a trading claim.
    """
    m = bs_call_metrics
    assert all(np.isfinite(v) for v in m.values())
    assert m["bsde_bs_call_analytic"] > 0.0
    assert m["bsde_bs_call_abs_err"] < 0.01  # observed ~1.2e-3 at 500 epochs
    assert m["bsde_bs_call_rel_err"] < 0.05
    assert m["bsde_bs_call_terminal_loss"] < m["bsde_bs_call_train_loss_start"]
    assert m["bsde_bs_call_n_params"] == _params_closed_form(1, BS_STEPS, 11)


@requires_torch
def test_burgers_hopf_y0_matches_known_solution(
    burgers_d1: tuple[db.BSDEProblem, db.DeepBSDEResult],
) -> None:
    prob, res = burgers_d1
    assert res.y0_exact == pytest.approx(0.5)
    assert res.y0_abs_err is not None and res.y0_abs_err < 0.02  # observed ~2.5e-4
    assert np.all(np.isfinite(res.train_loss_curve))
    assert res.train_loss_curve[-1] < res.train_loss_curve[0]  # training reduced the loss
    assert res.y0_curve.shape == (res.epochs,)
    assert res.y0_curve[-1] == pytest.approx(res.y0)
    assert res.z0.shape == (prob.dim,)
    assert res.n_parameters() == _params_closed_form(1, 12, 11)
    assert np.isfinite(res.terminal_loss) and res.terminal_loss >= 0.0


@requires_torch
def test_numpy_terminal_loss_matches_training_evaluation(
    burgers_d1: tuple[db.BSDEProblem, db.DeepBSDEResult],
) -> None:
    """The torch-free loss evaluation agrees with the trained solver's number."""
    prob, res = burgers_d1
    z = res.z_values(res.eval_x)
    assert z.shape == (res.eval_x.shape[0], prob.n_steps, prob.dim)
    loss = db.bsde_terminal_loss(prob, res.eval_x, res.eval_dw, res.y0, z)
    assert loss == pytest.approx(res.terminal_loss, abs=1e-9)
    # rollout starts from the learned theta_0 and stays finite
    y = db.bsde_rollout(prob, res.eval_x, res.eval_dw, res.y0, z)
    assert y.shape == (res.eval_x.shape[0], prob.n_steps + 1)
    assert np.all(y[:, 0] == res.y0) and np.all(np.isfinite(y))
    with pytest.raises(ValueError, match="x_paths must have shape"):
        res.z_values(res.eval_x[:, :-1, :])


@requires_torch
def test_dimension_scaling_same_architecture_recovers_known_solution() -> None:
    """Headline claim: one architecture rule trains at d=1 and d=10 alike.

    Same problem family (Burgers-Hopf, u(0,0) = 1/2 in every dimension), same
    per-layer MLP width rule (dim + 10), same optimizer settings; only the
    parameter count grows — polynomially, per the closed form. SYNTHETIC
    correctness evidence, never market evidence.
    """
    m = db.validate_dimension_scaling(
        dims=(1, 10), horizon=0.3, n_steps=12, epochs=450, lr=1e-2, n_paths=512, seed=0
    )
    assert all(np.isfinite(v) for v in m.values())
    assert m["bsde_burgers_d1_abs_err"] < 0.02  # observed ~2.5e-4
    assert m["bsde_burgers_d10_abs_err"] < 0.05  # observed ~2.8e-3
    assert m["bsde_dim_scaling_max_abs_err"] == max(
        m["bsde_burgers_d1_abs_err"], m["bsde_burgers_d10_abs_err"]
    )
    p1 = _params_closed_form(1, 12, 11)
    p10 = _params_closed_form(10, 12, 20)
    assert m["bsde_burgers_d1_n_params"] == p1
    assert m["bsde_burgers_d10_n_params"] == p10
    assert m["bsde_dim_scaling_param_ratio"] == pytest.approx(p10 / p1)
    # polynomial, not exponential, growth: ratio ~5 for a 10x dimension jump
    assert m["bsde_dim_scaling_param_ratio"] < 20.0
    with pytest.raises(ValueError, match="dims"):
        db.validate_dimension_scaling(dims=(3,), epochs=1)


@requires_torch
def test_deep_bsde_solve_deterministic_given_seed() -> None:
    prob = db.burgers_hopf_problem(dim=1, horizon=0.3, n_steps=8)
    kwargs = dict(epochs=40, lr=1e-2, n_paths=256, eval_paths=512)
    a = db.deep_bsde_solve(prob, seed=5, **kwargs)
    b = db.deep_bsde_solve(prob, seed=5, **kwargs)
    c = db.deep_bsde_solve(prob, seed=6, **kwargs)
    assert a.y0 == b.y0
    assert np.array_equal(a.train_loss_curve, b.train_loss_curve)
    assert np.array_equal(a.y0_curve, b.y0_curve)
    assert a.terminal_loss == b.terminal_loss
    assert not np.array_equal(a.train_loss_curve, c.train_loss_curve)


@requires_torch
def test_allen_cahn_smoke_trains_without_known_solution() -> None:
    """The paper's cubic benchmark trains (smoke only — no closed form exists).

    Only mechanics are asserted: finite decreasing loss and theta_0 inside the
    comparison-principle range implied by 0 < g <= 1/2. No value is claimed.
    """
    prob = db.allen_cahn_problem(dim=2, horizon=0.3, n_steps=10)
    res = db.deep_bsde_solve(prob, epochs=80, lr=1e-2, n_paths=256, eval_paths=512, seed=1)
    assert res.y0_exact is None and res.y0_abs_err is None
    assert np.all(np.isfinite(res.train_loss_curve))
    assert res.train_loss_curve[-1] < res.train_loss_curve[0]
    assert 0.0 < res.y0 < 1.0
    assert np.isfinite(res.terminal_loss)


@requires_torch
def test_deep_bsde_solve_fail_closed() -> None:
    prob = db.burgers_hopf_problem(dim=1, horizon=0.3, n_steps=4)
    with pytest.raises(ValueError, match="epochs"):
        db.deep_bsde_solve(prob, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        db.deep_bsde_solve(prob, lr=0.0)
    with pytest.raises(ValueError, match="hidden"):
        db.deep_bsde_solve(prob, hidden=())
    with pytest.raises(ValueError, match="n_paths"):
        db.deep_bsde_solve(prob, n_paths=0)
    with pytest.raises(ValueError, match="eval_paths"):
        db.deep_bsde_solve(prob, eval_paths=0)
    with pytest.raises(ValueError, match="y0_init"):
        db.deep_bsde_solve(prob, y0_init=float("nan"))
    with pytest.raises(TypeError, match="BSDEProblem"):
        db.deep_bsde_solve(object())  # type: ignore[arg-type]
