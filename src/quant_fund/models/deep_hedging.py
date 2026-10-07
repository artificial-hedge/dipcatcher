"""Deep hedging: learn a derivative hedge under transaction costs.

Research-grade minimal implementation of the deep-hedging algorithm:

- Buehler, H., Gonon, L., Teichmann, J. & Wood, B. (2019), "Deep Hedging",
  Quantitative Finance 19(10), 1271–1291,
  https://doi.org/10.1080/14697688.2019.1571683, arXiv:1802.03042
  (a neural network maps adapted market features at each rebalancing time to
  a target position in the underlying; the strategy minimizes a monetary risk
  measure of the hedged P&L over simulated paths, under general market
  frictions — here proportional transaction costs).

Risk measures supported (all applied to the hedged *loss* of a book that is
short one derivative and trades the underlying):

- Entropic risk rho_gamma(L) = (1/gamma) log E[exp(gamma L)] — Föllmer, H. &
  Schied, A. (2016), *Stochastic Finance: An Introduction in Discrete Time*,
  4th ed., de Gruyter, §4.9 (a convex monetary risk measure; the certainty
  equivalent of exponential utility).
- Expected shortfall / CVaR ES_alpha(L) = E[L | L >= VaR_alpha(L)] —
  Rockafellar, R.T. & Uryasev, S. (2000), "Optimization of conditional
  value-at-risk", Journal of Risk 2(3), 21–41, and (2002), "Conditional
  value-at-risk for general loss distributions", Journal of Banking & Finance
  26(7), 1443–1471. Implemented as the empirical estimator: the mean of the
  worst ceil((1-alpha) n) of n simulated losses, which is differentiable
  through the sort/top-k (the training gradient flows to tail paths only).
- Variance Var(L) — Markowitz, H. (1952), "Portfolio selection", Journal of
  Finance 7(1), 77–91.

Baselines: the analytic frictionless Black–Scholes replicating delta
Phi(d1) — Black, F. & Scholes, M. (1973), "The pricing of options and
corporate liabilities", Journal of Political Economy 81(3), 637–654; Merton,
R.C. (1973), "Theory of rational option pricing", Bell Journal of Economics
4(1), 141–183 — evaluated with and without friction, a static buy-and-hold
one-unit hedge, and the unhedged short payoff. Proportional transaction
costs follow the classical friction model of Davis, M.H.A. & Norman, A.R.
(1990), "Portfolio selection with transaction costs", Mathematics of
Operations Research 15(4), 676–713 (cost proportional to the notional
traded); paths are GBM (exact log-discretization) or a two-state
Markov-regime-switch GBM in the sense of Hamilton, J.D. (1989), "A new
approach to the economic analysis of nonstationary time series and the
business cycle", Econometrica 57(2), 357–384.

Honesty: everything here runs on SYNTHETIC simulated paths. Outputs are risk
measures of simulated hedged P&L — algorithmic correctness evidence, never
market evidence. No Sharpe/Sortino/P&L headline; no live-trading claims
(AGENTS.md honesty contract). Comparisons on the training batch are
in-sample; pass ``eval_paths`` to :func:`compare_hedges` for a clean
train/eval split.

Conventions: torch is the optional ``nn`` extra and is imported lazily via
:func:`_torch` (mirrors ``diffbacktest/jax_core.py``), so this module imports
cleanly without torch and every torch entry point raises ``ImportError`` with
install guidance. Numpy core for simulation, baselines, P&L accounting, and
risk-measure evaluation. Fail-closed edges: ``ValueError`` on non-positive or
non-finite prices, zero paths, fewer than two time points, negative cost
rates, unknown risk kinds, missing/invalid gamma or alpha, and shape
mismatches. Training is CPU single-thread full-batch Adam and deterministic
given ``seed`` (GPU determinism is not claimed).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import norm

Array = NDArray[np.float64]

__all__ = [
    "DeepHedgeResult",
    "HedgeComparison",
    "black_scholes_delta",
    "buy_and_hold_positions",
    "compare_hedges",
    "deep_hedge",
    "european_payoff",
    "hedged_loss",
    "hedged_pnl_components",
    "risk_measure",
    "simulate_gbm_paths",
    "simulate_regime_switch_paths",
]

_RISK_KINDS = ("entropic", "expected_shortfall", "variance")


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "deep hedging needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


def _check_count(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be an int >= 1; got {value!r}")
    return int(value)


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite; got {value!r}")
    return v


def _check_finite(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return v


def _as_paths(paths: Array | Sequence[Sequence[float]], *, name: str = "paths") -> Array:
    """Validate and normalize simulated price paths to float64 (n_paths, n_steps+1)."""
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array of shape (n_paths, n_steps+1); got ndim={arr.ndim}"
        )
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one path; got n_paths={arr.shape[0]}")
    if arr.shape[1] < 2:
        raise ValueError(
            f"{name} must have at least two time points (n_steps >= 1); got {arr.shape[1]}"
        )
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} entries must be finite (NaN/inf rejected)")
    if not bool(np.all(arr > 0.0)):
        raise ValueError(f"{name} must be strictly positive (price paths)")
    return arr


def _check_risk(
    kind: str, *, gamma: float | None, alpha: float | None, n: int
) -> tuple[str, float, float]:
    """Validate a risk-measure spec; returns (kind, gamma, alpha) normalized."""
    if kind not in _RISK_KINDS:
        raise ValueError(f"unknown risk kind {kind!r}; expected one of {_RISK_KINDS}")
    g = 0.0
    a = 0.0
    if kind == "entropic":
        if gamma is None:
            raise ValueError("entropic risk requires gamma > 0")
        g = _check_positive(gamma, "gamma")
    elif kind == "expected_shortfall":
        if alpha is None or not math.isfinite(float(alpha)) or not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"expected_shortfall requires alpha in (0, 1); got {alpha!r}")
        a = float(alpha)
    elif n < 2:
        raise ValueError(f"variance risk needs at least two losses; got {n}")
    return kind, g, a


def simulate_gbm_paths(
    n_paths: int,
    n_steps: int,
    *,
    s0: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.2,
    dt: float = 1.0 / 252.0,
    seed: int = 0,
) -> Array:
    """Geometric-Brownian-motion paths, exact log-discretization.

    ``S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)`` with i.i.d.
    standard normals ``Z`` from ``numpy.random.default_rng(seed)``. Returns a
    ``(n_paths, n_steps + 1)`` array whose first column equals ``s0``.
    SYNTHETIC data: correctness material, never market evidence.
    """
    n = _check_count(n_paths, "n_paths")
    m = _check_count(n_steps, "n_steps")
    s = _check_positive(s0, "s0")
    sig = _check_positive(sigma, "sigma")
    step = _check_positive(dt, "dt")
    drift = _check_finite(mu, "mu")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, m))
    log_increments = (drift - 0.5 * sig * sig) * step + sig * math.sqrt(step) * z
    out = np.empty((n, m + 1), dtype=float)
    out[:, 0] = s
    out[:, 1:] = s * np.exp(np.cumsum(log_increments, axis=1))
    return out


def simulate_regime_switch_paths(
    n_paths: int,
    n_steps: int,
    *,
    s0: float = 100.0,
    mu: float = 0.0,
    sigma_low: float = 0.1,
    sigma_high: float = 0.4,
    p_stay_low: float = 0.98,
    p_stay_high: float = 0.95,
    dt: float = 1.0 / 252.0,
    initial_regime: int = 0,
    seed: int = 0,
) -> Array:
    """Two-state Markov-regime-switch GBM paths (Hamilton 1989).

    The volatility state flips each step with transition matrix
    ``[[p_stay_low, 1 - p_stay_low], [1 - p_stay_high, p_stay_high]]``; within
    each step the log-price moves exactly as in :func:`simulate_gbm_paths`
    with the state's sigma. Returns ``(n_paths, n_steps + 1)``. SYNTHETIC
    data: correctness material, never market evidence.
    """
    n = _check_count(n_paths, "n_paths")
    m = _check_count(n_steps, "n_steps")
    s = _check_positive(s0, "s0")
    lo = _check_positive(sigma_low, "sigma_low")
    hi = _check_positive(sigma_high, "sigma_high")
    step = _check_positive(dt, "dt")
    drift = _check_finite(mu, "mu")
    for p, nm in ((p_stay_low, "p_stay_low"), (p_stay_high, "p_stay_high")):
        if not math.isfinite(float(p)) or not (0.0 <= float(p) <= 1.0):
            raise ValueError(f"{nm} must be a probability in [0, 1]; got {p!r}")
    if initial_regime not in (0, 1):
        raise ValueError(
            f"initial_regime must be 0 (low vol) or 1 (high vol); got {initial_regime!r}"
        )
    rng = np.random.default_rng(seed)
    u = rng.random((n, m))
    z = rng.standard_normal((n, m))
    regime = np.zeros((n, m), dtype=np.int8)
    regime[:, 0] = int(initial_regime)
    for t in range(1, m):
        stay = np.where(regime[:, t - 1] == 0, float(p_stay_low), float(p_stay_high))
        regime[:, t] = np.where(u[:, t] < stay, regime[:, t - 1], 1 - regime[:, t - 1])
    sig_t = np.where(regime == 0, lo, hi)
    log_increments = (drift - 0.5 * sig_t**2) * step + sig_t * math.sqrt(step) * z
    out = np.empty((n, m + 1), dtype=float)
    out[:, 0] = s
    out[:, 1:] = s * np.exp(np.cumsum(log_increments, axis=1))
    return out


def european_payoff(paths: Array, strike: float, *, option: str = "call") -> Array:
    """Terminal payoff of a European option on the path's last price.

    Returns a length-``n_paths`` vector: ``max(S_T - K, 0)`` for a call,
    ``max(K - S_T, 0)`` for a put.
    """
    arr = _as_paths(paths)
    k = _check_positive(strike, "strike")
    terminal = arr[:, -1]
    if option == "call":
        return np.maximum(terminal - k, 0.0)
    if option == "put":
        return np.maximum(k - terminal, 0.0)
    raise ValueError(f"option must be 'call' or 'put'; got {option!r}")


def black_scholes_delta(
    paths: Array,
    *,
    strike: float,
    maturity: float,
    sigma: float,
    r: float = 0.0,
    call: bool = True,
) -> Array:
    """Analytic Black–Scholes delta evaluated along simulated paths.

    Returns the frictionless replicating position ``(n_paths, n_steps)`` held
    over each interval ``[t_i, t_{i+1}]``: delta at grid time ``t_i = i *
    maturity / n_steps`` with remaining time to maturity ``maturity - t_i``
    (Black & Scholes 1973; Merton 1973). For a *short* call the hedge is long
    ``Phi(d1)`` units; for a short put, long ``Phi(d1) - 1`` units.
    """
    arr = _as_paths(paths)
    k = _check_positive(strike, "strike")
    t_mat = _check_positive(maturity, "maturity")
    sig = _check_positive(sigma, "sigma")
    rate = _check_finite(r, "r")
    steps = arr.shape[1] - 1
    ts = np.arange(steps, dtype=float) * (t_mat / steps)
    tau = t_mat - ts  # strictly positive at every rebalancing time
    sq = sig * np.sqrt(tau)
    d1 = (np.log(arr[:, :-1] / k) + (rate + 0.5 * sig * sig) * tau) / sq
    delta = np.asarray(norm.cdf(d1), dtype=float)
    if not call:
        delta = delta - 1.0
    return delta


def buy_and_hold_positions(paths: Array) -> Array:
    """Static one-unit hedge: buy at t_0, hold to maturity ``(n_paths, n_steps)``."""
    arr = _as_paths(paths)
    return np.ones((arr.shape[0], arr.shape[1] - 1), dtype=float)


def _check_positions(positions: Array, n_paths: int, n_steps: int) -> Array:
    pos = np.asarray(positions, dtype=float)
    if pos.shape != (n_paths, n_steps):
        raise ValueError(
            f"positions must have shape ({n_paths}, {n_steps}) matching "
            f"(n_paths, n_steps); got {pos.shape}"
        )
    if not bool(np.all(np.isfinite(pos))):
        raise ValueError("positions must be finite (NaN/inf rejected)")
    return pos


def _check_cost_rate(cost_rate: float) -> float:
    c = _check_finite(cost_rate, "cost_rate")
    if c < 0.0:
        raise ValueError(f"cost_rate must be non-negative (proportional cost); got {cost_rate!r}")
    return c


def hedged_pnl_components(
    paths: Array, positions: Array, payoff: Array, *, cost_rate: float
) -> dict[str, Array]:
    """Per-path P&L decomposition of a hedged short-derivative book.

    The book is short one derivative (owes ``payoff`` at maturity) and holds
    ``positions[:, i]`` units of the underlying over ``[t_i, t_{i+1}]``,
    starting flat. Proportional transaction costs (Davis & Norman 1990) of
    ``cost_rate * |delta position| * S`` are charged at each rebalancing time
    — including the initial entry — plus a final liquidation cost
    ``cost_rate * |position_T-1| * S_T`` to return to flat. The entry premium
    is a constant across strategies and is omitted; every risk measure used
    here is translation-invariant (Föllmer & Schied 2016, §4.1), so this
    does not affect comparisons.

    Returns per-path arrays: ``pnl`` (= -payoff + trading P&L - costs),
    ``trading_pnl``, ``costs``, and ``loss`` (= -pnl, the quantity risk
    measures are applied to).
    """
    arr = _as_paths(paths)
    n_paths, n_steps = arr.shape[0], arr.shape[1] - 1
    pos = _check_positions(positions, n_paths, n_steps)
    pay = np.asarray(payoff, dtype=float).reshape(-1)
    if pay.shape[0] != n_paths or not bool(np.isfinite(pay).all()):
        raise ValueError(f"payoff must be a finite vector of length n_paths={n_paths}")
    c = _check_cost_rate(cost_rate)
    ds = arr[:, 1:] - arr[:, :-1]
    trades = np.empty_like(pos)
    trades[:, 0] = pos[:, 0]
    trades[:, 1:] = pos[:, 1:] - pos[:, :-1]
    trading_pnl = np.sum(pos * ds, axis=1)
    costs = c * np.sum(np.abs(trades) * arr[:, :-1], axis=1) + c * np.abs(pos[:, -1]) * arr[:, -1]
    pnl = -pay + trading_pnl - costs
    return {"pnl": pnl, "trading_pnl": trading_pnl, "costs": costs, "loss": -pnl}


def hedged_loss(paths: Array, positions: Array, payoff: Array, *, cost_rate: float) -> Array:
    """Per-path hedged loss of the short-derivative book (see :func:`hedged_pnl_components`)."""
    return hedged_pnl_components(paths, positions, payoff, cost_rate=cost_rate)["loss"]


def risk_measure(
    losses: Array, kind: str, *, gamma: float | None = None, alpha: float | None = None
) -> float:
    """Monetary risk measure of a simulated loss sample (numpy evaluation).

    ``kind='entropic'``: rho_gamma(L) = (1/gamma) log E[exp(gamma L)]
    (Föllmer & Schied 2016, §4.9), computed with a stabilized log-sum-exp.
    ``kind='expected_shortfall'``: empirical ES_alpha — the mean of the worst
    ceil((1-alpha) n) losses (Rockafellar & Uryasev 2000, 2002).
    ``kind='variance'``: population variance (Markowitz 1952); needs n >= 2.
    """
    x = np.asarray(losses, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("losses must be non-empty")
    kind, g, a = _check_risk(kind, gamma=gamma, alpha=alpha, n=x.size)
    if not bool(np.isfinite(x).all()):
        raise ValueError("losses must be finite (NaN/inf rejected)")
    if kind == "entropic":
        return float(logsumexp(g * x) - math.log(x.size)) / g
    if kind == "expected_shortfall":
        k = max(1, int(math.ceil((1.0 - a) * x.size)))
        return float(np.sort(x)[-k:].mean())
    return float(x.var())


def _features(torch: Any, paths_t: Any) -> Any:
    """Adapted feature process ``(n_paths, n_steps, 3)`` for the hedging net.

    Features at rebalancing time ``t_i``: elapsed-time fraction ``i / T``,
    log-price relative to the path start ``log(S_i / S_0)``, and the last
    one-step simple return (0 at ``i = 0``). Buehler et al. (2019) admit an
    arbitrary adapted feature process; this is a minimal Markovian choice.
    """
    n_paths, n_cols = paths_t.shape
    steps = n_cols - 1
    spot = paths_t[:, :-1]
    t_frac = torch.arange(steps, dtype=paths_t.dtype).div(float(steps)).expand(n_paths, steps)
    log_rel = torch.log(spot / paths_t[:, :1])
    last_ret = torch.zeros_like(spot)
    last_ret[:, 1:] = spot[:, 1:] / spot[:, :-1] - 1.0
    return torch.stack([t_frac, log_rel, last_ret], dim=-1)


def _build_mlp(torch: Any, n_features: int, hidden: Sequence[int]) -> Any:
    """Shared-across-time MLP strategy network with a bounded tanh output."""
    layers: list[Any] = []
    d = int(n_features)
    for h in hidden:
        layers += [torch.nn.Linear(d, int(h)), torch.nn.Tanh()]
        d = int(h)
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers)


def _torch_hedged_loss(
    torch: Any, paths_t: Any, positions: Any, payoff_t: Any, cost_rate: float
) -> Any:
    """Torch mirror of :func:`hedged_pnl_components` returning the loss tensor."""
    ds = paths_t[:, 1:] - paths_t[:, :-1]
    trades = torch.cat([positions[:, :1], positions[:, 1:] - positions[:, :-1]], dim=1)
    costs = cost_rate * (trades.abs() * paths_t[:, :-1]).sum(dim=1)
    costs = costs + cost_rate * positions[:, -1].abs() * paths_t[:, -1]
    pnl = -payoff_t + (positions * ds).sum(dim=1) - costs
    return -pnl


def _torch_risk(torch: Any, loss_t: Any, kind: str, gamma: float, alpha: float) -> Any:
    """Differentiable risk measure of the per-path loss tensor (torch mirror)."""
    n = int(loss_t.shape[0])
    if kind == "entropic":
        return (torch.logsumexp(gamma * loss_t, dim=0) - math.log(n)) / gamma
    if kind == "expected_shortfall":
        k = max(1, int(math.ceil((1.0 - alpha) * n)))
        worst, _ = torch.topk(loss_t, k)
        return worst.mean()
    centered = loss_t - loss_t.mean()
    return (centered * centered).mean()


@dataclass(frozen=True)
class DeepHedgeResult:
    """Learned strategy and training diagnostics from :func:`deep_hedge`."""

    positions: Array  # (n_paths, n_steps) learned positions on the training paths
    loss: Array  # (n_paths,) hedged loss on the training paths
    risk: float  # final training risk under ``risk_kind``
    risk_kind: str
    loss_curve: Array  # per-epoch training risk (value before each update)
    seed: int
    epochs: int
    n_steps: int
    net: Any = field(repr=False, compare=False)  # torch MLP; use strategy_positions()

    def strategy_positions(self, paths: Array) -> Array:
        """Learned positions on a fresh batch of paths (eval mode, no grad).

        Enables an honest train/eval split: train on one simulated batch, then
        evaluate the frozen strategy on independently seeded paths.
        """
        arr = _as_paths(paths)
        if arr.shape[1] - 1 != self.n_steps:
            raise ValueError(
                f"paths must have n_steps+1={self.n_steps + 1} columns to match the "
                f"trained strategy; got {arr.shape[1]}"
            )
        torch = _torch()
        torch.set_num_threads(1)
        self.net.eval()
        with torch.no_grad():
            feats = _features(torch, torch.as_tensor(arr, dtype=torch.float32))
            pos = torch.tanh(self.net(feats)).squeeze(-1)
        return np.asarray(pos.numpy(), dtype=float)


def deep_hedge(
    paths: Array,
    payoff: Array,
    *,
    cost_rate: float,
    risk: str = "expected_shortfall",
    gamma: float | None = None,
    alpha: float | None = 0.9,
    hidden: Sequence[int] = (16, 16),
    epochs: int = 100,
    lr: float = 8e-3,
    seed: int = 0,
) -> DeepHedgeResult:
    """Learn a hedging strategy minimizing a risk measure under frictions.

    Deep hedging (Buehler, Gonon, Teichmann & Wood 2019, arXiv:1802.03042):
    a shared MLP maps the adapted features at each rebalancing time to a
    target position in the underlying; full-batch Adam minimizes the chosen
    risk measure (:func:`risk_measure`, default empirical ES at ``alpha``) of
    the per-path hedged loss (:func:`hedged_pnl_components`), which charges
    proportional transaction costs ``cost_rate`` on every position change and
    on the final liquidation. Positions are bounded to ``[-1, 1]`` units by a
    tanh output — an admissible-strategy constraint that rules out doubling
    strategies (cf. Buehler et al. 2019, §3). Deterministic given ``seed`` on
    CPU with a single thread; GPU determinism is not claimed.
    """
    torch = _torch()
    arr = _as_paths(paths)
    n_paths = arr.shape[0]
    n_steps = arr.shape[1] - 1
    pay = np.asarray(payoff, dtype=float).reshape(-1)
    if pay.shape[0] != n_paths or not bool(np.isfinite(pay).all()):
        raise ValueError(f"payoff must be a finite vector of length n_paths={n_paths}")
    c = _check_cost_rate(cost_rate)
    kind, g, a = _check_risk(risk, gamma=gamma, alpha=alpha, n=n_paths)
    hidden_widths = tuple(int(h) for h in hidden)
    if not hidden_widths or any(h < 1 for h in hidden_widths):
        raise ValueError(f"hidden must be a non-empty sequence of positive widths; got {hidden!r}")
    if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) < 1:
        raise ValueError(f"epochs must be an int >= 1; got {epochs!r}")
    n_epochs = int(epochs)
    rate = _check_positive(lr, "lr")

    torch.set_num_threads(1)
    paths_t = torch.as_tensor(arr, dtype=torch.float32)
    payoff_t = torch.as_tensor(pay, dtype=torch.float32)
    feats = _features(torch, paths_t)
    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        net = _build_mlp(torch, int(feats.shape[-1]), hidden_widths)
    opt = torch.optim.Adam(net.parameters(), lr=rate)
    curve: list[float] = []
    net.train()
    for _ in range(n_epochs):
        opt.zero_grad(set_to_none=True)
        positions_t = torch.tanh(net(feats)).squeeze(-1)
        loss_t = _torch_hedged_loss(torch, paths_t, positions_t, payoff_t, c)
        risk_t = _torch_risk(torch, loss_t, kind, g, a)
        curve.append(float(risk_t.detach().numpy()))
        risk_t.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        positions_t = torch.tanh(net(feats)).squeeze(-1)
        loss_t = _torch_hedged_loss(torch, paths_t, positions_t, payoff_t, c)
        final_risk = float(_torch_risk(torch, loss_t, kind, g, a).numpy())
        positions = np.asarray(positions_t.numpy(), dtype=float)
        loss = np.asarray(loss_t.numpy(), dtype=float)
    return DeepHedgeResult(
        positions=positions,
        loss=loss,
        risk=final_risk,
        risk_kind=kind,
        loss_curve=np.asarray(curve, dtype=float),
        seed=int(seed),
        epochs=n_epochs,
        n_steps=n_steps,
        net=net,
    )


@dataclass(frozen=True)
class HedgeComparison:
    """Learned vs baseline hedges of a short European option on the same paths.

    ``risk`` maps strategy name -> risk of its hedged loss on the evaluation
    paths: ``deep`` (learned, with friction), ``bs_delta`` (analytic delta,
    with friction), ``bs_delta_frictionless``, ``buy_and_hold`` (static
    one-unit hedge, with friction), and ``unhedged`` (short payoff, no
    trades). ``metrics`` is a flat float dict for scorecard wiring.
    """

    risk_kind: str
    risk: dict[str, float]
    mean_cost: dict[str, float]
    metrics: dict[str, float]
    deep: DeepHedgeResult


def compare_hedges(
    paths: Array,
    *,
    strike: float,
    maturity: float,
    sigma: float,
    r: float = 0.0,
    cost_rate: float,
    risk: str = "expected_shortfall",
    gamma: float | None = None,
    alpha: float | None = 0.9,
    hidden: Sequence[int] = (16, 16),
    epochs: int = 100,
    lr: float = 8e-3,
    seed: int = 0,
    eval_paths: Array | None = None,
) -> HedgeComparison:
    """Train a deep hedge and compare it against baselines on shared paths.

    The learned strategy is trained on ``paths`` (short one European call of
    ``strike``/``maturity``) and, together with the baselines, evaluated on
    ``eval_paths`` when given — otherwise on the training batch, which is
    in-sample and must be labeled as such. Metric keys are scorecard-ready:
    with ``risk='entropic'``, ``dh_hedged_risk`` is the hedged entropic risk
    (a bench would surface it as ``dh_hedged_entropic_risk``) and
    ``dh_friction_gap_vs_bs_delta`` is learned-minus-analytic risk under the
    same friction. SYNTHETIC correctness comparison, never market evidence.
    """
    train = _as_paths(paths, name="paths")
    k = _check_positive(strike, "strike")
    train_payoff = european_payoff(train, k)
    deep = deep_hedge(
        train,
        train_payoff,
        cost_rate=cost_rate,
        risk=risk,
        gamma=gamma,
        alpha=alpha,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )
    if eval_paths is None:
        ev = train
    else:
        ev = _as_paths(eval_paths, name="eval_paths")
        if ev.shape[1] != train.shape[1]:
            raise ValueError(
                f"eval_paths must have the same number of time points as paths "
                f"({train.shape[1]}); got {ev.shape[1]}"
            )
    payoff_ev = european_payoff(ev, k)
    zeros = np.zeros((ev.shape[0], ev.shape[1] - 1), dtype=float)
    bs_pos = black_scholes_delta(ev, strike=k, maturity=maturity, sigma=sigma, r=r, call=True)
    books = {
        "deep": deep.strategy_positions(ev),
        "bs_delta": bs_pos,
        "buy_and_hold": buy_and_hold_positions(ev),
        "unhedged": zeros,
    }
    risk_by: dict[str, float] = {}
    cost_by: dict[str, float] = {}
    for name, pos in books.items():
        comp = hedged_pnl_components(ev, pos, payoff_ev, cost_rate=cost_rate)
        risk_by[name] = risk_measure(comp["loss"], risk, gamma=gamma, alpha=alpha)
        cost_by[name] = float(comp["costs"].mean())
    comp_frictionless = hedged_pnl_components(ev, bs_pos, payoff_ev, cost_rate=0.0)
    risk_frictionless = risk_measure(comp_frictionless["loss"], risk, gamma=gamma, alpha=alpha)
    metrics = {
        "dh_hedged_risk": risk_by["deep"],
        "dh_unhedged_risk": risk_by["unhedged"],
        "dh_buy_and_hold_risk": risk_by["buy_and_hold"],
        "dh_bs_delta_friction_risk": risk_by["bs_delta"],
        "dh_bs_delta_frictionless_risk": risk_frictionless,
        "dh_friction_gap_vs_bs_delta": risk_by["deep"] - risk_by["bs_delta"],
        "dh_bs_friction_penalty": risk_by["bs_delta"] - risk_frictionless,
        "dh_deep_mean_cost": cost_by["deep"],
        "dh_bs_delta_mean_cost": cost_by["bs_delta"],
        "dh_buy_and_hold_mean_cost": cost_by["buy_and_hold"],
        "dh_deep_train_risk": deep.risk,
        "cost_rate": _check_cost_rate(cost_rate),
        "n_paths_eval": float(ev.shape[0]),
        "n_steps": float(ev.shape[1] - 1),
    }
    return HedgeComparison(
        risk_kind=deep.risk_kind,
        risk=risk_by,
        mean_cost=cost_by,
        metrics=metrics,
        deep=deep,
    )
