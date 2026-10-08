import numpy as np
import pytest

from quant_fund.models.c51_dqn import _ATOMS, bench_c51_dqn


def test_project_delta_rows_normalized() -> None:
    # every projected target row must carry unit mass — a degenerate row
    # would silently drop the sample from the cross-entropy target.
    torch = pytest.importorskip("torch")
    from quant_fund.models.c51_dqn import _project_delta

    support = np.linspace(-4.0, 2.0, _ATOMS)
    m = _project_delta(torch, np.array([-4.0, -1.0, 0.5, 2.0]), support)
    assert np.allclose(m.numpy().sum(1), 1.0)


def test_bench_tail_gauss_is_left_tail() -> None:
    # tail_pred[a] = P(Z < -1) under the learned distribution; the gaussian
    # baseline must be the SAME tail: P(X <= -1), which is tiny under a
    # right-of-zero return law. The old code emitted 1 - CDF(-1) (~0.86),
    # i.e. the right tail — a sign-inverted comparator.
    out = bench_c51_dqn(iters=50, mc_eval=500)
    assert 0.0 <= out["synthetic_c51_tail_gauss"] < 0.5
    assert out["synthetic_c51_tail_gauss"] < out["synthetic_c51_tail_pred"] + 0.4


def test_bench_deterministic() -> None:
    a = bench_c51_dqn(iters=20, mc_eval=300)
    b = bench_c51_dqn(iters=20, mc_eval=300)
    assert a == b
