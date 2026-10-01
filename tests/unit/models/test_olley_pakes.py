import numpy as np
import pytest

from quant_fund.models.olley_pakes import (
    bench_olley_pakes,
    op_estimate,
    synth_op_panel,
)


def test_op_recovers_beta_k() -> None:
    d = synth_op_panel(seed=0)
    out = op_estimate(
        np.asarray(d["y"]),
        np.asarray(d["lnk"]),
        np.asarray(d["lnl"]),
        np.asarray(d["inv"]),
        np.asarray(d["firm"]),
        np.asarray(d["period"]),
    )
    assert abs(out["beta_k"] - 0.6) < 0.2


def test_op_beats_ols_on_labor() -> None:
    d = synth_op_panel(seed=1)
    out = op_estimate(
        np.asarray(d["y"]),
        np.asarray(d["lnk"]),
        np.asarray(d["lnl"]),
        np.asarray(d["inv"]),
        np.asarray(d["firm"]),
        np.asarray(d["period"]),
    )
    x = np.column_stack([np.ones(d["y"].size), np.asarray(d["lnk"]), np.asarray(d["lnl"])])
    ols_l = float(np.linalg.lstsq(x, np.asarray(d["y"]), rcond=None)[0][2])
    assert abs(ols_l - 0.4) > abs(out["beta_l"] - 0.4)


def test_lagged_pairs_counted() -> None:
    d = synth_op_panel(seed=2)
    out = op_estimate(
        np.asarray(d["y"]),
        np.asarray(d["lnk"]),
        np.asarray(d["lnl"]),
        np.asarray(d["inv"]),
        np.asarray(d["firm"]),
        np.asarray(d["period"]),
    )
    assert out["n_lagged"] > 100


def test_synth_shapes() -> None:
    d = synth_op_panel(n_firms=10, t=4, seed=3)
    n = 10 * 4
    for key in ("y", "lnk", "lnl", "inv", "firm", "period"):
        assert np.asarray(d[key]).shape == (n,)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 120
    good = dict(
        y=rng.normal(0, 1, n),
        lnk=rng.normal(0, 1, n),
        lnl=rng.normal(0, 1, n),
        inv=rng.normal(0, 1, n),
        firm=np.repeat(np.arange(30), 4).astype(float),
        period=np.tile(np.arange(4), 30).astype(float),
    )
    with pytest.raises(ValueError):
        op_estimate(**{**good, "y": rng.normal(0, 1, 30)})
    with pytest.raises(ValueError):
        op_estimate(**{**good, "lnk": rng.normal(0, 1, n) * np.nan})
    with pytest.raises(ValueError):
        op_estimate(**{k: v[:40] for k, v in good.items()})


def test_bench() -> None:
    out = bench_olley_pakes()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
