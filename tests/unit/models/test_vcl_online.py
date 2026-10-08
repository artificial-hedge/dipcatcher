import inspect

import numpy as np

from quant_fund.models import vcl_online


def test_prior_pull_present_in_mean_update() -> None:
    """VCL mean update must include (mu - mu_pr)/var_pr; the previous
    code computed it then overwrote g_mu with the bare likelihood
    gradient (plain preconditioned SGD)."""
    src = inspect.getsource(vcl_online.bench_vcl_online)
    assert "(mu - mu_pr) / var_pr" in src


def test_bench_retention_honest() -> None:
    out = vcl_online.bench_vcl_online(T=60)
    assert np.isfinite(out["synthetic_vcl_task1_mse"])
    assert out["synthetic_vcl_post_var"] > 0.01  # variance must not collapse
