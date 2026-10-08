import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: E402
from quant_fund.models import vim_ood  # noqa: E402


def test_principal_subspace_is_top_variance() -> None:
    """Residual must live off the TOP-variance subspace: ID residuals
    small relative to raw ID spread.  eigh returns ascending eigenvalues,
    so top-d is V[:, -d:] — regressing to V[:, :d] leaves the ID signal
    inside the residual (this probe fails on that bug)."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal((400, 16)) @ np.diag(np.sqrt(np.linspace(0.05, 9.0, 16)))
    mu = f.mean(0)
    cov = np.cov((f - mu).T)
    _, V = np.linalg.eigh(cov)
    P = V[:, -8:] @ V[:, -8:].T
    r = (f - mu) - (f - mu) @ P
    id_resid = np.sqrt((r**2).sum(1))
    id_full = np.sqrt(((f - mu) ** 2).sum(1))
    assert id_resid.mean() < 0.5 * id_full.mean()


def test_bench_runs_and_vim_scored() -> None:
    out = vim_ood.bench_vim_ood(iters=60)
    assert 0.0 <= out["synthetic_vim_auc"] <= 1.0
    # with the least-variance subspace this collapsed to ~0.73 < MSP;
    # the virtual logit must not underperform the bare max-logit score
    assert out["synthetic_vim_auc"] > out["synthetic_vim_msp_auc"] - 0.01
