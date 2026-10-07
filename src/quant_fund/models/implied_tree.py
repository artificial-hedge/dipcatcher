"""Implied binomial tree smile calibration (Derman–Kani / Rubinstein) (SYNTHETIC).

Builds a recombining binomial lattice whose Arrow–Debreu prices reproduce
an observed expiry smile: at each level the next node prices are solved so
that calls (upper half) and puts (lower half) struck at the level's forward
prices match smile-implied prices, using the alternating-center-node
recursion of Derman & Kani (1994). Transition probabilities are read off
the martingale condition ``p = (F_i − S_down)/(S_up − S_down)``; violations
(``p ∉ [0,1]``) are arbitrage flags — the constructor fails closed unless
``repair=True``, which clamps offending nodes a tick outside the forward
(documented: a local-vol clipping repair, not exact calibration).

Intermediate-maturity prices needed by the recursion are produced from a
flat term structure: the expiry smile's implied-vol-vs-strike curve is
reused at every level (documented approximation — true DK interpolates the
full surface in K and t). The result is a *nonparametric lattice*
complement to parametric smile fits that prices path-dependent claims by
the absorbing-node knockout approximation.

Honesty
-------
All bench outputs are ``synthetic_*`` correctness diagnostics on a seeded
synthetic smile — smile fit error, Arrow–Debreu mass, martingale error,
arbitrage counts, knockout bounds. They verify lattice mechanics, never
market value. No Sharpe/Sortino/Calmar/P&L/NAV ever.

References
----------
- Derman, E. & Kani, I. (1994). Riding on a smile. *Risk* 7(2):32–39.
- Derman, E., Kani, I. & Chriss, N. (1996). Implied trinomial trees of the
  volatility smile. *Journal of Derivatives* 3(4):7–22.
- Rubinstein, M. (1994). Implied binomial trees. *Journal of Finance*
  49(3):771–818.
- Breeden, D.T. & Litzenberger, R.H. (1978). Prices of state-contingent
  claims implicit in option prices. *Journal of Business* 51(4):621–651.
  (Second-difference density used in the bench comparator.)

Composition notes
-----------------
- ``models.svi_surface`` (wave 23): parametric smile calibration — this is
  the nonparametric lattice complement that prices path-dependent payoffs.
- ``models.breeden_litzenberger`` (wave 24): static density reader — the
  tree is the dynamic version with transition structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]
FORBIDDEN_KEYS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


@dataclass(frozen=True)
class ImpliedTree:
    """Recombining binomial lattice calibrated to one expiry smile.

    ``nodes[n]`` are the level-n spot prices (ascending); ``probs[n]`` the
    up-transition probabilities from level n; ``ad[n]`` the Arrow–Debreu
    prices (discounted risk-neutral mass). ``violations`` counts nodes
    whose raw transition probability fell outside [0, 1] before repair.
    """

    times: FloatArray
    nodes: tuple[FloatArray, ...]
    probs: tuple[FloatArray, ...]
    ad: tuple[FloatArray, ...]
    forwards: tuple[FloatArray, ...]
    spot: float
    r: float
    q: float
    tau: float
    violations: int
    repaired: bool


def _bs_call(fwd: float, k: float, t: float, vol: float, df: float) -> float:
    if t <= 0 or vol <= 0:
        return df * max(fwd - k, 0.0)
    sqt = vol * np.sqrt(t)
    d1 = (np.log(fwd / k) + 0.5 * vol * vol * t) / sqt
    d2 = d1 - sqt
    return float(df * (fwd * norm.cdf(d1) - k * norm.cdf(d2)))


def _bs_put(fwd: float, k: float, t: float, vol: float, df: float) -> float:
    return _bs_call(fwd, k, t, vol, df) - df * (fwd - k)


def _implied_vol_call(call_price: float, fwd: float, k: float, t: float, df: float) -> float:
    """Bisection inversion of Black-76 call → implied vol."""
    intrinsic = df * max(fwd - k, 0.0)
    if call_price <= intrinsic + 1e-12:
        return 1e-4
    lo, hi = 1e-4, 5.0
    if _bs_call(fwd, k, t, hi, df) < call_price:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _bs_call(fwd, k, t, mid, df) < call_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _smile_iv_interp(strikes: FloatArray, ivs: FloatArray, k: float) -> float:
    """Flat-extrapolated linear smile interpolation in strike."""
    return float(np.interp(k, strikes, ivs))


def implied_binomial_tree(
    spot: float,
    strikes: object,
    call_prices: object,
    r: float,
    q_carry: float,
    n_steps: int,
    tau: float,
    repair: bool = False,
) -> ImpliedTree:
    """Derman–Kani implied binomial tree calibrated to an expiry smile.

    ``strikes``/``call_prices`` describe the expiry-``tau`` smile (calls,
    ascending strikes). Level ``n+1`` node prices are solved bottom-up:

    - center node pinned at the forward ``F = spot·e^{(r−q)t}``;
    - upper nodes solved from smile-implied calls at ``F_i``;
    - lower nodes solved from smile-implied puts at ``F_i``.

    Fails closed on arbitrage (node ordering or ``p ∉ [0,1]`` violations)
    unless ``repair=True``, which nudges offending nodes just outside the
    forward — a local-vol clip, documented, counted in ``violations``.
    """
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be a positive finite float")
    if not np.isfinite(r) or not np.isfinite(q_carry):
        raise ValueError("r and q_carry must be finite")
    if n_steps < 2 or n_steps > 60:
        raise ValueError("n_steps must be in [2, 60]")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive")

    k_arr = np.asarray(strikes, dtype=np.float64)
    c_arr = np.asarray(call_prices, dtype=np.float64)
    if k_arr.ndim != 1 or c_arr.shape != k_arr.shape or k_arr.size < 3:
        raise ValueError("strikes/call_prices must be matching vectors with >= 3 points")
    if not np.isfinite(k_arr).all() or not np.isfinite(c_arr).all() or (k_arr <= 0).any():
        raise ValueError("strikes/call_prices must be finite, strikes positive")
    if np.any(np.diff(k_arr) <= 0):
        raise ValueError("strikes must be strictly ascending")
    if (c_arr < 0).any():
        raise ValueError("call_prices must be nonnegative")

    df_t = np.exp(-r * tau)
    fwd_t = spot * np.exp((r - q_carry) * tau)
    ivs = np.array(
        [_implied_vol_call(c_arr[i], fwd_t, k_arr[i], tau, df_t) for i in range(k_arr.size)]
    )

    dt = tau / n_steps
    times = np.linspace(0.0, tau, n_steps + 1)
    nodes: list[FloatArray] = [np.array([spot])]
    probs: list[FloatArray] = []
    ads: list[FloatArray] = [np.array([1.0])]
    forwards: list[FloatArray] = [np.array([spot * np.exp((r - q_carry) * dt)])]
    violations = 0

    def fwd_at(t: float) -> float:
        return float(spot * np.exp((r - q_carry) * t))

    for n in range(n_steps):
        s_n = nodes[n]
        lam = ads[n]
        t_next = times[n + 1]
        df_next = np.exp(-r * t_next)
        f_i = s_n * np.exp((r - q_carry) * dt)  # forwards of level-n nodes
        n_next = n + 2
        s_next = np.empty(n_next)

        # fully-ITM collapse: a node whose children both finish above/below
        # strike K contributes exactly lam_j * (F_j - K) to the call/put,
        # so only the straddling node needs its unknown child solved.
        def call_residual(
            i: int,
            f_i: FloatArray = f_i,
            lam: FloatArray = lam,
            t_next: float = t_next,
            df_next: float = df_next,
        ) -> float:
            vol = _smile_iv_interp(k_arr, ivs, float(f_i[i]))
            c_mkt = _bs_call(fwd_at(t_next), float(f_i[i]), t_next, vol, df_next)
            upper = float(np.sum(lam[i + 1 :] * (f_i[i + 1 :] - f_i[i])))
            return float(np.exp(r * dt) * c_mkt - upper)

        def put_residual(
            i: int,
            f_i: FloatArray = f_i,
            lam: FloatArray = lam,
            t_next: float = t_next,
            df_next: float = df_next,
        ) -> float:
            vol = _smile_iv_interp(k_arr, ivs, float(f_i[i]))
            p_mkt = _bs_put(fwd_at(t_next), float(f_i[i]), t_next, vol, df_next)
            lower = float(np.sum(lam[:i] * (f_i[i] - f_i[:i])))
            return float(np.exp(r * dt) * p_mkt - lower)

        if n_next % 2 == 1:
            center = n_next // 2
            s_next[center] = fwd_at(t_next)
            i_start, i_center_up = center, center
        else:
            c = n_next // 2 - 1  # lower of the center pair
            f_c = float(f_i[c])
            sum_c = call_residual(c)
            # S'_c = F_c²/S'_{c+1} centering gives closed form:
            # p_c = F_c/(S'_{c+1} + F_c)  ⇒  S'_{c+1} = F_c(λF+Σ)/(λF−Σ)
            denom_c = lam[c] * f_c - sum_c
            if denom_c <= 0 or sum_c <= 0:
                violations += 1
                if not repair:
                    raise ValueError(f"degenerate center pair at level {n}; repair=True to clamp")
                s_next[c + 1] = f_c * 1.001
            else:
                s_next[c + 1] = f_c * (lam[c] * f_c + sum_c) / denom_c
            s_next[c] = f_c * f_c / s_next[c + 1]
            i_start = c - 1  # first lower node to solve
            i_center_up = c + 1  # first upper node to solve

        # upper nodes: call at strike F_i pins the up-child S'_{i+1}
        for i in range(i_center_up, n + 1):
            f_k = float(f_i[i])
            sum_c = call_residual(i)
            s_i = s_next[i]  # known down-child of node i
            # Σ = λ(F−S'_i)(S'_{i+1}−F)/(S'_{i+1}−S'_i) solved for S'_{i+1}
            num = lam[i] * f_k * (f_k - s_i) - sum_c * s_i
            den = lam[i] * (f_k - s_i) - sum_c
            s_up = num / den if abs(den) > 1e-14 else np.nan
            if not np.isfinite(s_up) or s_up <= f_k or sum_c <= 0:
                violations += 1
                if not repair:
                    raise ValueError(
                        f"arbitrage violation solving node {i} at level {n}: "
                        f"S_up={s_up:.6g}, F={f_k:.6g}; pass repair=True to clamp"
                    )
                s_up = f_k * (1.0 + 1e-6)
            s_next[i + 1] = s_up

        # lower nodes: put at strike F_i pins the down-child S'_i
        for i in range(i_start, -1, -1):
            f_k = float(f_i[i])
            sum_p = put_residual(i)
            s_up1 = s_next[i + 1]  # known up-child of node i
            num = lam[i] * f_k * (s_up1 - f_k) - sum_p * s_up1
            den = lam[i] * (s_up1 - f_k) - sum_p
            s_dn = num / den if abs(den) > 1e-14 else np.nan
            if not np.isfinite(s_dn) or s_dn >= f_k or sum_p <= 0:
                violations += 1
                if not repair:
                    raise ValueError(
                        f"arbitrage violation solving node {i} at level {n}: "
                        f"S_down={s_dn:.6g}, F={f_k:.6g}; pass repair=True to clamp"
                    )
                s_dn = f_k * (1.0 - 1e-6)
            s_next[i] = s_dn

        if np.any(np.diff(s_next) <= 0):
            if not repair:
                raise ValueError(f"non-monotone node prices at level {n + 1}")
            s_next = np.maximum.accumulate(s_next * (1.0 + 1e-9 * np.arange(n_next)))
            violations += 1

        p_n = (f_i - s_next[:-1]) / (s_next[1:] - s_next[:-1])
        bad = (~np.isfinite(p_n)) | (p_n < -1e-12) | (p_n > 1 + 1e-12)
        if bad.any():
            violations += int(bad.sum())
            if not repair:
                raise ValueError(
                    f"transition probability outside [0,1] at level {n}; pass repair=True to clamp"
                )
            p_n = np.clip(p_n, 0.0, 1.0)

        lam_next = np.zeros(n_next)
        lam_next[0] = lam[0] * (1.0 - p_n[0])
        lam_next[-1] = lam[-1] * p_n[-1]
        for j in range(1, n_next - 1):
            lam_next[j] = np.exp(-r * dt) * (lam[j - 1] * p_n[j - 1] + lam[j] * (1.0 - p_n[j]))
        lam_next[[0, -1]] *= np.exp(-r * dt)

        nodes.append(s_next)
        probs.append(p_n)
        ads.append(lam_next)
        forwards.append(s_next * np.exp((r - q_carry) * dt))

    return ImpliedTree(
        times=times,
        nodes=tuple(nodes),
        probs=tuple(probs),
        ad=tuple(ads),
        forwards=tuple(forwards),
        spot=spot,
        r=r,
        q=q_carry,
        tau=tau,
        violations=violations,
        repaired=repair,
    )


def arrow_debreu_prices(tree: ImpliedTree) -> FloatArray:
    """Terminal Arrow–Debreu prices (discounted RN state prices)."""
    return np.asarray(tree.ad[-1], dtype=np.float64)


def tree_price(tree: ImpliedTree, payoff: FloatArray) -> float:
    """Price a terminal payoff vector on the calibrated tree leaves."""
    pay = np.asarray(payoff, dtype=np.float64)
    leaf = tree.nodes[-1]
    if pay.shape != leaf.shape or not np.isfinite(pay).all():
        raise ValueError("payoff must be a finite array matching terminal nodes")
    return float(arrow_debreu_prices(tree) @ pay)


def tree_price_path(
    tree: ImpliedTree,
    barrier: float,
    payoff: FloatArray,
) -> float:
    """Down-and-out price via the absorbing-node knockout approximation.

    Arrow–Debreu mass passing through any node at or below ``barrier`` is
    knocked out before reaching the leaves — a standard absorbing-lattice
    approximation (honest: ignores within-step barrier crossings).
    """
    if not np.isfinite(barrier) or barrier <= 0:
        raise ValueError("barrier must be positive")
    pay = np.asarray(payoff, dtype=np.float64)
    leaf = tree.nodes[-1]
    if pay.shape != leaf.shape or not np.isfinite(pay).all():
        raise ValueError("payoff must match terminal nodes")

    dt = tree.tau / (len(tree.times) - 1)
    surv = [np.array([0.0 if tree.nodes[0][0] <= barrier else 1.0])]
    for n in range(len(tree.probs)):
        s_next = tree.nodes[n + 1]
        p_n = tree.probs[n]
        lam = tree.ad[n]
        # live AD mass into next-level nodes: up-moves from j-1, down from j
        up_from = np.zeros_like(s_next)
        dn_from = np.zeros_like(s_next)
        up_from[1:] = lam * surv[n] * p_n
        dn_from[:-1] = lam * surv[n] * (1.0 - p_n)
        # live AD inflow keeps the per-step discount; surv is the alive
        # fraction of each node's AD mass
        live_next = np.where(s_next > barrier, np.exp(-tree.r * dt) * (up_from + dn_from), 0.0)
        ad_next = tree.ad[n + 1]
        frac = np.divide(
            live_next,
            np.maximum(ad_next, 1e-300),
            out=np.zeros_like(ad_next),
            where=ad_next > 0,
        )
        surv.append(frac)
    return float(tree.ad[-1] @ (surv[-1] * pay))


def synth_smile(
    spot: float = 100.0,
    r: float = 0.02,
    q: float = 0.0,
    tau: float = 0.5,
    n_strikes: int = 9,
    atm_vol: float = 0.20,
    skew: float = -0.10,
    kurt: float = 0.15,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Synthetic smile → BS call prices for the bench.

    Implied vol ``σ(K) = atm + skew·m + kurt·m²`` with ``m = log(K/F)``;
    a tiny seeded perturbation keeps the input realistic. Returns strikes,
    call prices, and the iv curve.
    """
    if spot <= 0 or tau <= 0 or n_strikes < 5:
        raise ValueError("spot, tau positive; n_strikes >= 5")
    rng = np.random.default_rng(seed)
    fwd = spot * np.exp((r - q) * tau)
    ks = np.linspace(0.7, 1.35, n_strikes) * fwd
    m = np.log(ks / fwd)
    ivs = atm_vol + skew * m + kurt * m * m + rng.normal(0.0, 0.002, n_strikes)
    ivs = np.clip(ivs, 0.02, 2.0)
    df = np.exp(-r * tau)
    calls = np.array([_bs_call(fwd, ks[i], tau, ivs[i], df) for i in range(n_strikes)])
    return {"strikes": ks, "calls": calls, "ivs": ivs, "fwd": np.array(fwd)}


def bench_implied_tree(seed: int = 0) -> dict[str, float]:
    """SYNTHETIC bench: smile fit, AD mass, martingale, arb, KO bounds."""
    sm = synth_smile(seed=seed)
    ks, calls = sm["strikes"], sm["calls"]
    # 12 steps: zero arbitrage violations on this smile; leaf-staircase
    # repricing uses a 1%-of-spot absolute floor for far-OTM calls
    tree = implied_binomial_tree(
        100.0, ks, calls, r=0.02, q_carry=0.0, n_steps=12, tau=0.5, repair=True
    )
    leaf = tree.nodes[-1]
    ad = arrow_debreu_prices(tree)

    # reprice input calls on the tree
    tree_calls = np.array([tree_price(tree, np.maximum(leaf - k, 0.0)) for k in ks])
    fit_err = float(np.max(np.abs(tree_calls - calls) / np.maximum(calls, 0.01 * 100.0)))

    mass = float(ad.sum())
    # martingale: E[S_tau] under AD / df should equal the forward
    df_t = np.exp(-tree.r * tree.tau)
    e_s = float(ad @ leaf) / df_t
    fwd = 100.0 * np.exp((tree.r - tree.q) * tree.tau)
    mart_err = float(abs(e_s - fwd) / fwd)

    # path claim bounds: KO call ≤ vanilla call, ≥ 0
    strike = float(ks[len(ks) // 2])
    vanilla = tree_price(tree, np.maximum(leaf - strike, 0.0))
    ko = tree_price_path(tree, 0.85 * 100.0, np.maximum(leaf - strike, 0.0))
    bounds_ok = float(0.0 <= ko <= vanilla + 1e-10)
    deep_ko = tree_price_path(tree, 1e-6, np.maximum(leaf - strike, 0.0))
    ko_consistency = float(abs(deep_ko - vanilla) / max(vanilla, 1e-8) < 0.05)

    t2 = implied_binomial_tree(
        100.0, ks, calls, r=0.02, q_carry=0.0, n_steps=12, tau=0.5, repair=True
    )
    determinism = float(np.array_equal(t2.nodes[-1], leaf))

    blob: dict[str, float] = {
        "synthetic_smile_fit_err": fit_err,
        "synthetic_density_mass": mass,
        "synthetic_density_mass_err": abs(mass - df_t) / df_t,
        "synthetic_martingale_err": mart_err,
        "synthetic_arb_violations": float(tree.violations),
        "synthetic_path_claim_bounds": bounds_ok,
        "synthetic_ko_consistency": ko_consistency,
        "synthetic_n_leaf_nodes": float(leaf.size),
        "synthetic_terminal_range_ratio": float(leaf[-1] / leaf[0]),
        "synthetic_determinism": determinism,
    }
    for k in blob:
        if FORBIDDEN_KEYS.intersection(k.split("_")):
            raise ValueError(f"forbidden bench key {k!r}")
    return blob
