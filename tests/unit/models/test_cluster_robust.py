"""Adversarial probes for cluster_robust."""

import numpy as np
import pytest

from quant_fund.models import cluster_robust as cr


def _panel(n_clusters: int = 12, n_per: int = 8, seed: int = 0):
    d = cr.synth_cluster(n_clusters=n_clusters, n_per=n_per, seed=seed)
    return d["y"], d["x"], d["cluster"]


def test_crve_rejects_unknown_corr():
    y, x, cl = _panel()
    with pytest.raises(ValueError, match="corr"):
        cr.crve(y, x, cl, corr="bogus")


def test_crve_fails_closed_on_zero_se():
    """Perfect fit -> zero meat -> must raise, not return t=inf."""
    y, x, cl = _panel()
    n = y.size
    a = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    y_exact = a @ beta  # residual-free y -> every cluster score is 0
    with pytest.raises(ValueError, match="degenerate"):
        cr.crve(y_exact, x, cl)


def test_wild_rejects_unknown_weight():
    y, x, cl = _panel()
    with pytest.raises(ValueError, match="weight"):
        cr.wild_cluster_bootstrap(y, x, cl, n_boot=9, weight="bogus")


def test_wild_rejects_zero_boot():
    y, x, cl = _panel()
    with pytest.raises(ValueError, match="n_boot"):
        cr.wild_cluster_bootstrap(y, x, cl, n_boot=0)
