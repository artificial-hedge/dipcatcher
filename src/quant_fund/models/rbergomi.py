"""Rough Bergomi model — hybrid-scheme simulation + implied-vol surface.

Bayer, Friz & Gatheral (2016), "Pricing under rough volatility":

    d log S_t = -1/2 V_t dt + sqrt(V_t) dW_t,   W = rho W1 + sqrt(1-rho^2) W2
    V_t = xi0(t) exp( eta * sqrt(2H) * W^H_t - 0.5 eta^2 t^{2H} )
    W^H_t = ∫_0^t (t-s)^{H-1/2} dW1_s        (Volterra process)

The Volterra driver is simulated with the Bennedsen–Lunde–Pakkanen hybrid
scheme of order kappa: cells within kappa of the evaluation time use the
optimal power-kernel evaluation point b*_k, older cells are frozen into a
per-cell Gaussian functional Z_j (kernel anchored kappa cells ahead) whose
exact variance/covariance with the cell's Brownian increment is used to
draw the pair (dW1_j, Z_j) jointly by Cholesky.

Anchors pinned by the receipt:
  - Var(W^H_t) * (2H) / t^{2H} ~ 1 across the grid (the scheme's own
    self-consistency check against Volterra theory).
  - H -> 0.5 recovers a flat smile (|ATM skew| small).
  - ATM skew blows up ~ T^{H-1/2} (Fukasawa 2011 short-maturity law) —
    measured exponent reported against theory.
  - E[S_T] = S0 (martingale, no drift).
  - rho < 0 produces negative skew (leverage asymmetry).

References:
- Bayer, Friz, Gatheral (2016), arXiv:1410.6512.
- Bennedsen, Lunde, Pakkanen (2017), "Hybrid scheme for Brownian
  semistationary processes", Finance & Stochastics.
- Fukasawa (2011), "Asymptotic expansion for rough volatility".
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

RBERGOMI_SCHEMA = "rough_vol.v1"


class RBergomiParams(NamedTuple):
    """Rough Bergomi parameters (flat forward variance curve)."""

    s0: float
    xi0: float
    h: float
    eta: float
    rho: float
    t: float


def _bstar(k: int, alpha: float, dt: float) -> float:
    """Optimal power-kernel evaluation point in cell k (BLP Thm. 4.4)."""
    return float(
        dt * ((k ** (alpha + 1.0) - (k - 1.0) ** (alpha + 1.0)) / (alpha + 1.0)) ** (1.0 / alpha)
    )


def _cell_params(kappa: int, alpha: float, dt: float) -> tuple[float, float]:
    """Variance of the frozen cell field Z_j and its covariance with the
    cell increment dW1_j.

    Z_j = ∫_{cell j} K(t_{j+kappa+1} - s) dW_s / sqrt(dt),  K(r) = r^alpha:
      Var(Z_j)  = dt^{2a} * ((kappa+1)^{2a+1} - kappa^{2a+1}) / (2a+1)
      Cov(dW1_j/√dt, Z_j) = dt^{a} * ((kappa+1)^{a+1} - kappa^{a+1}) / (a+1)
    """
    var_z = (
        dt ** (2.0 * alpha)
        * ((kappa + 1.0) ** (2.0 * alpha + 1.0) - kappa ** (2.0 * alpha + 1.0))
        / (2.0 * alpha + 1.0)
    )
    cov_ez = dt**alpha * ((kappa + 1.0) ** (alpha + 1.0) - kappa ** (alpha + 1.0)) / (alpha + 1.0)
    return var_z, cov_ez


def hybrid_volterra(
    n_steps: int,
    h: float,
    dt: float,
    n_paths: int,
    seed: int,
    kappa: int = 1,
) -> tuple[Array, Array]:
    """Simulate the Volterra W^H on the grid plus its driving increments.

    Returns ``(wH, dW1)``: ``wH[i]`` is W^H at t_i (shape
    (n_steps, n_paths)) — built on cells 0..i-1 so the Euler vol at step
    i stays causal — and ``dW1`` the unit-variance-scaled Brownian
    increments the field was built on (for price correlation).
    """
    alpha = h - 0.5
    if not (-0.5 < alpha < 0.0):
        raise ValueError("rough regime requires H in (0, 0.5)")
    if kappa < 1 or kappa >= n_steps:
        raise ValueError("kappa must satisfy 1 <= kappa < n_steps")
    var_z, cov_ez = _cell_params(kappa, alpha, dt)
    rho_ez = cov_ez / math.sqrt(var_z)  # corr(dW1/sqrt(dt), Z)
    if not 0.0 <= rho_ez < 1.0:
        raise ValueError("cell correlation out of range")
    rng = np.random.default_rng(seed)
    # per-cell joint draw: (eps_j ~ dW1/sqrt(dt), zeta_j) bivariate normal
    eps = rng.standard_normal((n_steps, n_paths))
    eta_ = rng.standard_normal((n_steps, n_paths))
    zeta = rho_ez * eps + math.sqrt(1.0 - rho_ez**2) * eta_
    z_cell = math.sqrt(var_z) * zeta  # Z_j with exact variance
    bstar = np.array([_bstar(k, alpha, dt) for k in range(1, kappa + 1)])
    # weights: frozen cells carry their exact per-time L2-norm rescale;
    # the most recent κ cells use K(b*_k) point masses.
    wh = np.empty((n_steps, n_paths))
    k_pow = np.power(bstar, alpha)  # K(b*_k)
    for i in range(n_steps):
        # wh[i] is W^H(t_i): integrates cells 0..i-1 only — cell i's own
        # increment must NOT feed the vol at t_i (it belongs to V_{i+1});
        # otherwise the Euler step loses martingaleity.
        j_far = i - kappa  # frozen cells c = 0..i-κ-1
        acc = np.zeros(n_paths)
        if j_far > 0:
            # norm-exact rescale: Var of cell c's contribution at time i is
            # its exact L2 norm ∫_h^{h+1} v^{2a} dv with h = i-c-1; anchored
            # at h=κ the weight is 1 by construction.
            h0 = i - j_far  # closest frozen cell's distance (= κ)
            hh = np.arange(i - 1, h0 - 1, -1, dtype=float)
            num = (hh + 1.0) ** (2.0 * alpha + 1.0) - hh ** (2.0 * alpha + 1.0)
            den = (kappa + 1.0) ** (2.0 * alpha + 1.0) - float(kappa) ** (2.0 * alpha + 1.0)
            w = np.sqrt(num / den)
            acc = w @ z_cell[:j_far]
        for k in range(1, min(kappa, i) + 1):
            acc = acc + k_pow[k - 1] * eps[i - k]
        wh[i] = acc * math.sqrt(dt)
    return wh, eps * math.sqrt(dt)


def rbergomi_paths(
    params: RBergomiParams,
    n_steps: int,
    n_paths: int,
    seed: int,
    kappa: int = 1,
) -> Array:
    """Terminal prices S_T under the rough Bergomi model (flat xi0)."""
    s0, xi0, h, eta, rho, t = params
    if s0 <= 0 or xi0 <= 0 or eta <= 0 or not (-1.0 < rho < 1.0):
        raise ValueError("bad rBergomi params")
    dt = t / n_steps
    wh, dW1 = hybrid_volterra(n_steps, h, dt, n_paths, seed, kappa)
    rng = np.random.default_rng(seed + 1)
    dW2 = rng.standard_normal((n_steps, n_paths)) * math.sqrt(dt)
    dW = rho * dW1 + math.sqrt(1.0 - rho**2) * dW2
    tt = np.arange(0, n_steps)[:, None] * dt
    v = xi0 * np.exp(eta * math.sqrt(2.0 * h) * wh - 0.5 * eta**2 * tt ** (2.0 * h))
    log_s = math.log(s0) + np.cumsum(-0.5 * v * dt + np.sqrt(v) * dW, axis=0)
    out: Array = np.exp(log_s[-1])
    return out


def _implied_vol(price: float, s0: float, k: float, t: float) -> float:
    """Black-Scholes implied vol by bisection (r=0)."""
    from scipy import stats as sstats

    def call(iv: float) -> float:
        if iv <= 0.0:
            return max(s0 - k, 0.0)
        sq = iv * math.sqrt(t)
        d1 = math.log(s0 / k) / sq + 0.5 * sq
        d2 = d1 - sq
        return float(s0 * float(sstats.norm.cdf(d1)) - k * float(sstats.norm.cdf(d2)))

    lo, hi = 1e-6, 5.0
    if abs(price - call(lo)) < 1e-10:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if call(mid) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rbergomi_smile(
    params: RBergomiParams,
    strikes: Array,
    n_steps: int,
    n_paths: int,
    seed: int,
    kappa: int = 1,
) -> Array:
    """Implied-vol smile: per-strike Black implied vol of MC call prices."""
    s_t = rbergomi_paths(params, n_steps, n_paths, seed, kappa)
    out = np.empty(strikes.size)
    for i, k in enumerate(strikes):
        px = float(np.mean(np.maximum(s_t - k, 0.0)))
        out[i] = _implied_vol(px, params.s0, float(k), params.t)
    return out


def rough_vol_bench(
    n_paths: int = 12000,
    n_steps: int = 100,
    seed: int = 91,
) -> dict[str, Any]:
    """Measure the rough-vol machinery's defining signatures.

    Claims (all measured, never asserted):
      * volterra_variance: Var(W^H_t)·2H/t^{2H} inside (0.6, 1.5) mid-grid
      * gbm_limit_flat: |ATM skew| at H=0.45 well below the H=0.10 level
      * skew_power_law: fitted exponent of |skew(T)| in (-0.55, -0.30) for
        H=0.10 (theory: H-1/2 = -0.4)
      * martingale: E[S_T] within 1% of S0
      * smile_smirking: negative-ρ smile is negatively sloped
    """
    s0, xi0, eta, rho = 1.0, 0.04, 1.9, -0.7

    # -- volterra self-consistency
    h_check = 0.10
    wh, _ = hybrid_volterra(200, h_check, 1.0 / 200, 4000, seed)
    tt = np.arange(0, 200) / 200.0
    var_w = wh.var(axis=1)
    ratio = var_w * (2.0 * h_check) / np.maximum(tt, 1e-12) ** (2.0 * h_check)
    mid = ratio[len(ratio) // 4 : -1]
    volterra_ok = bool(bool(np.all((mid > 0.6) & (mid < 1.5))))

    # -- skew vs maturity (strikes in ATM-sigma units so the slope estimator
    # stays inside the stochastically-relevant moneyness band at tiny T)
    mats = np.array([1 / 52.0, 1 / 24.0, 1 / 12.0, 0.25])
    ksig = np.linspace(-1.5, 1.5, 9)
    skew_h10: list[float] = []
    skew_h45: list[float] = []
    for j, t in enumerate(mats):
        logks = ksig * math.sqrt(xi0 * float(t))
        for h_val, sink in ((0.10, skew_h10), (0.45, skew_h45)):
            iv = rbergomi_smile(
                RBergomiParams(s0, xi0, h_val, eta, rho, float(t)),
                s0 * np.exp(logks),
                n_steps,
                n_paths,
                seed + 1000 * j + (0 if h_val < 0.2 else 500),
            )
            slope, _ = np.polyfit(logks, iv, 1)
            sink.append(float(-slope / math.sqrt(xi0)))  # normalized ATM skew
    skew_arr = np.array(skew_h10)
    skew45 = np.array(skew_h45)
    expo, _ = np.polyfit(np.log(mats), np.log(np.abs(skew_arr)), 1)
    expo45, _ = np.polyfit(np.log(mats), np.log(np.abs(skew45)), 1)

    # -- martingale
    s_t = rbergomi_paths(
        RBergomiParams(s0, xi0, 0.10, eta, rho, 0.25), n_steps, n_paths, seed + 7777
    )
    mean_st = float(s_t.mean())

    results = {
        # the kappa=1 hybrid understates Var(W^H) ~15% at H=0.10 — the
        # divergent cell-0 kernel integral defeats a single point mass; the
        # measured ratio is pinned, not hidden
        "volterra_variance_shape": volterra_ok,
        # at H->0.5 the short-maturity smile flattens (GBM limit): skew at
        # T=1/52 sits materially below the rough-H level and decays slower
        "gbm_limit_flat": bool(abs(skew45[0]) < 0.75 * abs(skew_arr[0])),
        "skew_power_law": bool(-0.55 < expo < -0.30),
        "flat_h_power_law": bool(expo45 > expo - 0.05),
        "martingale": abs(mean_st / s0 - 1.0) < 0.01,
        "smile_smirking": bool(skew_arr[0] > 0 and skew_arr[-1] > 0),
        "all_fits_finite": bool(np.isfinite(skew_arr).all() and np.isfinite(expo)),
    }
    n_passed = sum(1 for v in results.values() if v)
    claim = {
        "ok": n_passed == len(results),
        "n_probes": len(results),
        "n_passed": n_passed,
        "results": results,
        "skew_exponent": {"measured": float(expo), "theoretical": 0.10 - 0.5},
        "skew_exponent_h45": {"measured": float(expo45), "theoretical": 0.45 - 0.5},
        "atm_skew_h10": skew_h10,
        "atm_skew_h45": skew_h45,
        "maturities": mats.tolist(),
        "mean_terminal": mean_st,
        "volterra_variance_ratio": {"mid_min": float(mid.min()), "mid_max": float(mid.max())},
    }
    payload: dict[str, Any] = {
        "kind": "rough_vol",
        "schema": RBERGOMI_SCHEMA,
        "git_revision": _git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "params": {"n_paths": n_paths, "n_steps": n_steps, "seed": seed},
        "claim": claim,
        "interpretation": (
            "Rough Bergomi under the BLP hybrid scheme (kappa=1): the Volterra "
            "driver's variance shape is self-checked against Volterra theory, "
            "the H->0.5 limit flattens the smile, and the short-maturity ATM "
            "skew exponent is measured against the Fukasawa T^(H-1/2) law. "
            "SYNTHETIC model-correctness evidence; no market claims."
        ),
    }
    return payload


def rough_vol_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Deep-verify a ``rough_vol.v1`` receipt's internal coherence."""
    errors: list[str] = []
    claim = payload.get("claim")
    if not isinstance(claim, dict):
        return ["missing_claim"]
    results = claim.get("results")
    if not isinstance(results, dict) or not all(isinstance(v, bool) for v in results.values()):
        errors.append("results_not_bool_map")
        results = {}
    if claim.get("n_probes") != len(results):
        errors.append("n_probes_mismatch")
    if claim.get("n_passed") != sum(1 for v in results.values() if v):
        errors.append("n_passed_mismatch")
    if claim.get("ok") != all(results.values()):
        errors.append("ok_mismatch")
    for key in ("skew_exponent", "skew_exponent_h45"):
        expo = claim.get(key)
        if (
            not isinstance(expo, dict)
            or not isinstance(expo.get("measured"), (int, float))
            or not isinstance(expo.get("theoretical"), (int, float))
        ):
            errors.append(f"{key}_shape")
    vr = claim.get("volterra_variance_ratio")
    if (
        not isinstance(vr, dict)
        or not isinstance(vr.get("mid_min"), (int, float))
        or not isinstance(vr.get("mid_max"), (int, float))
        or not vr.get("mid_min", 0) <= vr.get("mid_max", -1)
    ):
        errors.append("volterra_variance_ratio_shape")
    for key in ("atm_skew_h10", "atm_skew_h45", "maturities"):
        v = claim.get(key)
        if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
            errors.append(f"{key}_shape")
    if payload.get("research_only") is not True:
        errors.append("research_only")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if payload.get("data_label") != "SYNTHETIC":
        errors.append("data_label")
    return errors


def _git_revision() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def write_rough_vol_receipt(
    receipt: dict[str, Any],
    receipts_dir: Any = "receipts",
) -> Any:
    """Seal (receipt_sha256) and atomically write ``rough_vol.json``."""
    import json
    from pathlib import Path

    from quant_fund.research.receipt_v2 import verify_receipt_payload
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    if receipt.get("kind") != "rough_vol" or receipt.get("schema") != RBERGOMI_SCHEMA:
        raise ValueError("receipt identity mismatch")
    canonical = json.loads(canonical_json_bytes(receipt))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    v = verify_receipt_payload(payload)
    if not v["valid"]:
        raise ValueError(f"sealed receipt fails verification: {v['errors']}")
    path = Path(receipts_dir) / "rough_vol.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path
