import numpy as np
import pytest

from quant_fund.metrics.two_way_cluster import (
    bench_two_way_cluster,
    synth_twoway,
    two_way_cluster,
)


def test_twc_beta_recovered() -> None:
    d = synth_twoway(beta=0.5, seed=2)
    out = two_way_cluster(d["y"], d["x"], d["g1"], d["g2"])
    assert abs(out["beta"] - 0.5) < 0.2


def test_twc_inflates_se_under_correlation() -> None:
    d = synth_twoway(rho1=0.5, rho2=0.5, seed=3)
    out = two_way_cluster(d["y"], d["x"], d["g1"], d["g2"])
    assert out["se_two_way"] > out["se_iid"]
    assert out["inflation_vs_iid"] > 1.2


def test_twc_iid_no_inflation() -> None:
    d = synth_twoway(rho1=0.0, rho2=0.0, seed=4)
    out = two_way_cluster(d["y"], d["x"], d["g1"], d["g2"])
    assert abs(out["inflation_vs_iid"] - 1.0) < 0.3


def test_twc_validation() -> None:
    d = synth_twoway(g1=5, g2=40, seed=5)
    with pytest.raises(ValueError):
        two_way_cluster(d["y"], d["x"], d["g1"], d["g2"])
    d2 = synth_twoway(seed=6)
    with pytest.raises(ValueError):
        two_way_cluster(d2["y"][:100], d2["x"][:100], d2["g1"][:100], d2["g2"][:100])
    with pytest.raises(ValueError):
        two_way_cluster(d2["y"], d2["x"], d2["g1"][:-1], d2["g2"])


def test_twc_deterministic() -> None:
    d = synth_twoway(seed=7)
    a = two_way_cluster(d["y"], d["x"], d["g1"], d["g2"])
    b = two_way_cluster(d["y"], d["x"], d["g1"], d["g2"])
    assert a["se_two_way"] == b["se_two_way"]


def test_bench_two_way_cluster() -> None:
    out = bench_two_way_cluster()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
