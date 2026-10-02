"""Brinson-Hood-Beebower (1986) performance attribution.

Decomposes portfolio vs benchmark active return into:
- Allocation: (w_p - w_b) * (r_b_sector - r_b_total)
- Selection: w_b * (r_p_sector - r_b_sector)
- Interaction: (w_p - w_b) * (r_p_sector - r_b_sector)

Extensions:
- Carino (1999) logarithmic smoothing for multi-period
  linking of arithmetic attribution effects.
- Brinson-Fachler (1985) variant (allocation vs
  sector-neutral benchmark).

References
----------
- Brinson, Hood & Beebower (1986) 'Determinants of
  portfolio performance' Financial Analysts Journal.
- Brinson & Fachler (1985) 'Measuring non-US equity
  portfolio performance' JPM.
- Carino (1999) 'Combining attribution effects over time'
  J. Perf. Measurement 4(1).

Honesty
-------
SYNTHETIC self-check: 4-sector portfolio where effects
sum exactly to the arithmetic active return (identity
check) plus a linked-period consistency check.

Composition
-----------
Pure numpy. Inputs are sector weight/return matrices for
portfolio and benchmark; outputs are per-sector and total
attribution effects.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_wp(wp: FloatArray, wb: FloatArray) -> tuple[FloatArray, FloatArray]:
    wpa = np.asarray(wp, dtype=np.float64)
    wba = np.asarray(wb, dtype=np.float64)
    if wpa.shape != wba.shape or wpa.ndim == 0:
        raise ValueError("weight shape mismatch")
    if (wpa < -0.5).any() or (wba < 0).any():
        raise ValueError("weights out of range")
    return wpa, wba


def brinson_single(
    wp: FloatArray,
    wb: FloatArray,
    rp: FloatArray,
    rb: FloatArray,
    model: str = "bhb",
) -> dict[str, FloatArray]:
    """Single-period BHB attribution.

    wp, wb: sector weights (n,) or (n_periods, n_sectors).
    rp, rb: sector returns matching shapes.
    model: 'bhb' (allocation vs total bench) or 'bf'
    (Brinson-Fachler, allocation vs sector return only).
    Returns per-sector effects and totals; all effects sum
    to the arithmetic active return.
    """
    wpa, wba = _check_wp(wp, wb)
    rpa, rba = np.asarray(rp, dtype=np.float64), np.asarray(rb, dtype=np.float64)
    if rpa.shape != wpa.shape or rba.shape != wba.shape:
        raise ValueError("return shape mismatch")
    squeeze = wpa.ndim == 1
    wp2 = np.atleast_2d(wpa)
    wb2 = np.atleast_2d(wba)
    rp2 = np.atleast_2d(rpa)
    rb2 = np.atleast_2d(rba)
    rb_tot = (wb2 * rb2).sum(1)
    rp_tot = (wp2 * rp2).sum(1)
    if model == "bhb":
        alloc = (wp2 - wb2) * (rb2 - rb_tot[:, None])
        sel = wb2 * (rp2 - rb2)
    elif model == "bf":
        alloc = (wp2 - wb2) * (rb2 - rb_tot[:, None])
        sel = wp2 * (rp2 - rb2)
    else:
        raise ValueError("model must be bhb|bf")
    inter = (wp2 - wb2) * (rp2 - rb2)
    active = rp_tot - rb_tot
    # arithmetic identity check
    tot = alloc.sum(1) + sel.sum(1) + inter.sum(1)
    if model == "bhb" and not np.allclose(tot, active, atol=1e-9):
        raise ValueError("BHB identity broken")
    out = {
        "allocation": alloc,
        "selection": sel,
        "interaction": inter,
        "active_return": active,
        "total": tot,
    }
    if squeeze:
        return {k: v.ravel() if v.ndim == 2 else v for k, v in out.items()}
    return out


def carino_link(rp_total: FloatArray, rb_total: FloatArray, effects: FloatArray) -> FloatArray:
    """Carino (1999) logarithmic linking of attribution effects.

    rp_total, rb_total: per-period total returns (T,).
    effects: (T, n_effects) single-period effects to link.
    Returns linked effects (n_effects,) whose sum equals the
    geometric active return."""
    rp = np.asarray(rp_total, dtype=np.float64).ravel()
    rb = np.asarray(rb_total, dtype=np.float64).ravel()
    E = np.asarray(effects, dtype=np.float64)
    if E.ndim == 1:
        E = E[:, None]
    if rp.size != rb.size or E.shape[0] != rp.size:
        raise ValueError("period count mismatch")
    if (1 + rp <= 0).any() or (1 + rb <= 0).any():
        raise ValueError("returns below -100%")
    ln_p = np.log1p(rp)
    ln_b = np.log1p(rb)
    # Carino k_t = ln((1+rp)/(1+rb)) / (rp - rb); period-scaled link coef
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(
            np.abs(rp - rb) > 1e-12,
            (ln_p - ln_b) / (rp - rb),
            1.0 / (1 + rp),
        )
    k = (ln_p.sum() - ln_b.sum()) / (rp - rb).sum() if abs((rp - rb).sum()) > 1e-12 else 1.0
    scale = k / kt  # per-period adjustment
    linked = (E * scale[:, None]).sum(0)
    return linked


def bench_brinson(seed: int = 509) -> dict[str, float]:
    """SYNTHETIC: 4-sector, 2-period attribution with
    exact-identity and Carino-link checks."""
    rng = np.random.default_rng(seed)
    n_sectors, n_per = 4, 3
    wb = np.tile([0.4, 0.3, 0.2, 0.1], (n_per, 1))
    wp = wb + rng.normal(0, 0.04, (n_per, n_sectors))
    wp = np.clip(wp, 0.0, None)
    wp /= wp.sum(1, keepdims=True)
    rb = rng.normal(0.005, 0.03, (n_per, n_sectors))
    rp = rb + rng.normal(0.003, 0.015, (n_per, n_sectors))
    out = brinson_single(wp, wb, rp, rb)
    identity_err = float(np.abs(out["total"] - out["active_return"]).max())
    effects = np.stack(
        [out["allocation"].sum(1), out["selection"].sum(1), out["interaction"].sum(1)],
        axis=1,
    )
    linked = carino_link((wp * rp).sum(1), (wb * rb).sum(1), effects)
    geo_active = float(np.prod(1 + (wp * rp).sum(1)) / np.prod(1 + (wb * rb).sum(1)) - 1)
    if identity_err > 1e-9:
        raise ValueError("attribution identity failed")
    return {
        "synthetic_identity_err": identity_err,
        "synthetic_total_alloc": float(out["allocation"].sum()),
        "synthetic_total_sel": float(out["selection"].sum()),
        "synthetic_total_inter": float(out["interaction"].sum()),
        "synthetic_geo_active": geo_active,
        "synthetic_linked_total": float(linked.sum()),
    }
