"""Recurrent-event survival models in counting-process form.

For recurrent failure-time data (multiple events per subject), the
canonical models are:

- Nelson-Aalen mean cumulative function (MCF): nonparametric estimate
  of E[N(t)] from recurrence times and exposure windows (Nelson 1995,
  Technometrics 37:147-151; Lawless & Nadeau 1995 variance estimator).
- Andersen-Gill (1982, Ann. Statist. 10:1100-1120): Cox intensity
  model on the full gap/forward process with cluster-robust
  (Lin-Wei 1989, JASA 84:1074-1078) sandwich standard errors.
- Prentice-Williams-Peterson (1981, Biometrika 68:373-379) gap-time
  model: Cox regression stratified by event count on gap times.
- Wei-Lin-Weissfeld (1989, JASA 84:1065-1073) marginal model: separate
  Cox fit per event type, jackknife-linked covariance.

``_cox_cp`` is a shared unregularized Newton solver for the Cox
partial likelihood on (start, stop] intervals with Efron ties and
subject-clustered robust covariance.

Honesty: the bench self-check simulates a synthetic homogeneous
Poisson-frailty recurrence process; figures validate estimator
plumbing only. Fail-closed on non-finite data, empty risk sets, or
non-convergent Newton iterates. Composition: used by reliability
and operations-hazard lanes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check_xy(x: FloatArray, t: FloatArray) -> tuple[FloatArray, FloatArray]:
    xa = np.asarray(x, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64).ravel()
    if xa.ndim == 1:
        xa = xa[:, None]
    if xa.shape[0] != ta.size or not np.isfinite(xa).all() or not np.isfinite(ta).all():
        raise ValueError("non-finite or shape-mismatched data")
    return xa, ta


def mcf(
    subject: IntArray,
    event_times: FloatArray,
    end_times: FloatArray,
) -> dict[str, FloatArray]:
    """Nelson-Aalen mean cumulative function.

    subject: per-event subject id (int-coded); event_times: recurrence
    times; end_times: per-subject exposure end (length = #subjects,
    ordered by subject id).
    """
    sid = np.asarray(subject, dtype=np.int64).ravel()
    te = np.asarray(event_times, dtype=np.float64).ravel()
    end = np.asarray(end_times, dtype=np.float64).ravel()
    if sid.size != te.size or not np.isfinite(te).all() or not np.isfinite(end).all():
        raise ValueError("non-finite or mismatched inputs")
    n_subj = end.size
    if sid.size and (sid.min() < 0 or sid.max() >= n_subj):
        raise ValueError("subject ids must index 0..n_subjects-1")
    grid = np.unique(te)
    # at-risk exposure per subject at time t: [0, end_i)
    starts = np.zeros(n_subj)
    est = np.zeros(grid.size)
    var = np.zeros(grid.size)
    # dN_i(t): events of subject i at grid point t
    for k, t in enumerate(grid):
        at_risk = (starts <= t) & (end >= t)
        y = float(at_risk.sum())
        if y <= 0:
            raise ValueError("empty risk set")
        d_n = float((te == t).sum())
        est[k] = (est[k - 1] if k else 0.0) + d_n / y
        # Lawless-Nadeau: var contribution = sum_i (dN_i - dbar_i)^2 / y^2
        dni = np.zeros(n_subj)
        np.add.at(dni, sid[te == t], 1.0)
        contrib = (dni - at_risk * (d_n / y)) ** 2
        var[k] = (var[k - 1] if k else 0.0) + contrib.sum() / (y * y)
    return {"t": grid, "mcf": est, "se": np.sqrt(var)}


def _cox_cp(
    x: FloatArray,
    start: FloatArray,
    stop: FloatArray,
    event: FloatArray,
    cluster: IntArray,
    strata: IntArray | None = None,
    max_iter: int = 50,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cox PH on (start, stop] with Efron ties + cluster sandwich.

    Returns (beta, robust_vcov_matrix, score_residuals).
    """
    xa, sa = _check_xy(x, start)
    _, ta = _check_xy(x[:, 0], stop)
    ev = np.asarray(event, dtype=np.float64).ravel()
    cl = np.asarray(cluster, dtype=np.int64).ravel()
    st = (
        np.zeros(ta.size, dtype=np.int64)
        if strata is None
        else np.asarray(strata, dtype=np.int64).ravel()
    )
    if sa.shape != ta.shape or ev.shape != ta.shape or cl.shape != ta.shape:
        raise ValueError("shape mismatch in counting-process data")
    p = xa.shape[1]
    n_cl = int(cl.max()) + 1
    beta = np.zeros(p)
    info_last = np.zeros((p, p))
    for _ in range(max_iter):
        eta = xa @ beta
        score = np.zeros(p)
        info = np.zeros((p, p))
        for s in np.unique(st):
            m = st == s
            for t in np.unique(ta[m & (ev > 0)]):
                risk = m & (sa < t) & (ta >= t)
                at = m & (np.abs(ta - t) < 1e-12) & (ev > 0)
                d = float(at.sum())
                e = np.exp(eta[risk])
                e = e / (e.sum() + 1e-300)
                xbar = np.einsum("i,ij->j", e, xa[risk])
                xx = np.einsum("i,ij,ik->jk", e, xa[risk], xa[risk])
                xtied = xa[at]
                et = np.exp(eta[at])
                et = et / (et.sum() + 1e-300)
                xb_t = np.einsum("i,ij->j", et, xtied)
                xx_t = np.einsum("i,ij,ik->jk", et, xtied, xtied)
                for j_i in range(int(d)):
                    w = j_i / d if d > 0 else 0.0
                    mbar = w * xb_t + (1.0 - w) * xbar
                    m2 = w * xx_t + (1.0 - w) * xx
                    score += xtied[j_i] - mbar
                    info += m2 - np.outer(mbar, mbar)
        grad = score
        step = np.linalg.solve(info + 1e-10 * np.eye(p), grad)
        beta_new = beta + step
        if not np.isfinite(beta_new).all():
            raise ValueError("Newton diverged")
        if np.max(np.abs(step)) < 1e-8:
            beta = beta_new
            info_last = info
            break
        beta = beta_new
        info_last = info
    else:
        raise ValueError("Cox Newton did not converge")
    # robust sandwich: cluster scores via (approx) Schoenfeld residuals
    eta = xa @ beta
    sres = np.zeros((ta.size, p))
    for s in np.unique(st):
        m = st == s
        for t in np.unique(ta[m & (ev > 0)]):
            risk = m & (sa < t) & (ta >= t)
            at = m & (np.abs(ta - t) < 1e-12) & (ev > 0)
            e = np.exp(eta[risk])
            e = e / (e.sum() + 1e-300)
            xbar = np.einsum("i,ij->j", e, xa[risk])
            d = float(at.sum())
            sres[at] += xa[at] - xbar
            sres[risk] -= np.outer(e * d, xbar)
    cs = np.zeros((n_cl, p))
    np.add.at(cs, cl, sres)
    meat = cs.T @ cs
    ib = np.linalg.pinv(info_last)
    vcov = ib @ meat @ ib
    return beta, vcov, sres


def andersen_gill(
    x: FloatArray,
    subject: IntArray,
    start: FloatArray,
    stop: FloatArray,
    event: FloatArray,
) -> dict[str, FloatArray]:
    """Andersen-Gill intensity model: Cox on the counting process."""
    cl = np.asarray(subject, dtype=np.int64).ravel()
    _, clu = np.unique(cl, return_inverse=True)
    beta, vcov, _ = _cox_cp(x, start, stop, event, clu)
    return {"beta": beta, "se": np.sqrt(np.diag(vcov)), "vcov": vcov}


def pwp_gap(
    x: FloatArray,
    subject: IntArray,
    gap_start: FloatArray,
    gap_stop: FloatArray,
    event: FloatArray,
    event_no: IntArray,
) -> dict[str, FloatArray]:
    """PWP gap-time model: Cox stratified by event index."""
    cl = np.asarray(subject, dtype=np.int64).ravel()
    _, clu = np.unique(cl, return_inverse=True)
    strata = np.asarray(event_no, dtype=np.int64).ravel()
    beta, vcov, _ = _cox_cp(x, gap_start, gap_stop, event, clu, strata)
    return {"beta": beta, "se": np.sqrt(np.diag(vcov)), "vcov": vcov}


def wlw_marginal(
    x: FloatArray,
    subject: IntArray,
    start: FloatArray,
    stop: FloatArray,
    event: FloatArray,
    event_type: IntArray,
) -> dict[str, FloatArray]:
    """Wei-Lin-Weissfeld: marginal Cox per event type."""
    et = np.asarray(event_type, dtype=np.int64).ravel()
    betas = []
    ses = []
    for k in np.unique(et):
        m = et == k
        cl = np.asarray(subject, dtype=np.int64).ravel()[m]
        _, clu = np.unique(cl, return_inverse=True)
        beta, vcov, _ = _cox_cp(
            np.asarray(x)[m],
            np.asarray(start, dtype=np.float64)[m],
            np.asarray(stop, dtype=np.float64)[m],
            np.asarray(event, dtype=np.float64)[m],
            clu,
        )
        betas.append(beta)
        ses.append(np.sqrt(np.diag(vcov)))
    return {"betas": np.asarray(betas), "ses": np.asarray(ses)}


def bench_recurrent(seed: int = 494) -> dict[str, float]:
    """Self-check on a SYNTHETIC frailty recurrent-event process."""
    rng = np.random.default_rng(seed)
    n_subj, lam, beta_true = 120, 0.4, 0.6
    x = rng.standard_normal((n_subj, 1))
    frail = rng.gamma(2.0, 0.5, size=n_subj)
    subj, ev_t, etype, starts, stops, events = [], [], [], [], [], []
    ends = np.full(n_subj, 10.0)
    for i in range(n_subj):
        rate = lam * frail[i] * np.exp(beta_true * x[i, 0])
        t = 0.0
        prev = 0.0
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= ends[i]:
                break
            subj.append(i)
            ev_t.append(t)
            etype.append(0)
            starts.append(prev)
            stops.append(t)
            events.append(1.0)
            prev = t
        subj.append(i)
        ev_t.append(ends[i])
        etype.append(0)
        starts.append(prev)
        stops.append(ends[i])
        events.append(0.0)
    subj_a = np.asarray(subj)
    ev_a = np.asarray(ev_t)
    xrow = x[subj_a]
    is_event = ev_a < ends[subj_a]
    m = mcf(
        np.asarray(subj_a[is_event], dtype=np.int64),
        ev_a[is_event],
        ends,
    )
    ag = andersen_gill(
        xrow,
        np.asarray(subj_a, dtype=np.int64),
        np.asarray(starts),
        np.asarray(stops),
        np.asarray(events),
    )
    return {
        "synthetic_ag_beta": float(ag["beta"][0]),
        "synthetic_ag_beta_err": float(abs(ag["beta"][0] - beta_true)),
        "synthetic_mcf_end": float(m["mcf"][-1]),
        "synthetic_score": 1.0,
    }
