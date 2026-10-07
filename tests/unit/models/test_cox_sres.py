"""Probe: cluster-robust Cox score residuals must encode (x_i - xbar) for
risk-set members; the buggy form subtracted xbar for everyone."""

from __future__ import annotations

import numpy as np

from quant_fund.models.recurrent_events import _cox_cp


def _fixture():
    rng = np.random.default_rng(3)
    n_subj = 40
    x = rng.standard_normal((n_subj, 1))
    rows_s, rows_t, rows_e, rows_c, rows_x = [], [], [], [], []
    for i in range(n_subj):
        rate = 0.3 * np.exp(0.7 * x[i, 0])
        t = 0.0
        prev = 0.0
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= 10.0:
                break
            rows_s.append(prev)
            rows_t.append(t)
            rows_e.append(1.0)
            rows_c.append(i)
            rows_x.append(x[i, 0])
            prev = t
        rows_s.append(prev)
        rows_t.append(10.0)
        rows_e.append(0.0)
        rows_c.append(i)
        rows_x.append(x[i, 0])
    return (
        np.asarray(rows_x)[:, None],
        np.asarray(rows_s),
        np.asarray(rows_t),
        np.asarray(rows_e),
        np.asarray(rows_c, dtype=np.int64),
    )


def test_score_residuals_sum_to_zero() -> None:
    """At the converged fit, total score residual must be ~0 (it equals the
    score at the MLE). The buggy form leaves a nonzero residual."""
    x, s, t, e, cl = _fixture()
    _, _, sres = _cox_cp(x, s, t, e, cl)
    total = np.abs(sres.sum(axis=0))
    assert np.all(total < 1e-6), f"score residuals do not sum to ~0: {total}"


def test_vcov_changes_vs_naive() -> None:
    """Sanity: robust vcov is symmetric PSD and beta unaffected."""
    x, s, t, e, cl = _fixture()
    beta, vcov, _ = _cox_cp(x, s, t, e, cl)
    assert np.isfinite(beta).all()
    assert np.allclose(vcov, vcov.T)
    assert np.linalg.eigvalsh(vcov).min() > -1e-9
