"""Tests for wave-123 NLP/generative SOTA: say_echo_do,
multimodal_fusion, ts_diffusion, synthetic_gan, econ_calendar,
quantcode_bench."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.econ_calendar import bench_econ_calendar, fit_impact
from quant_fund.models.multimodal_fusion import (
    GatedFusion,
    bench_multimodal_fusion,
    chart_feats,
    price_feats,
    text_feats,
)
from quant_fund.models.quantcode_bench import (
    StrategySpec,
    backtest,
    bench_quantcode_bench,
    compile_spec,
    static_check,
    synth_bars,
)
from quant_fund.models.say_echo_do import (
    SayDoTracker,
    bench_say_echo_do,
    echo_score,
    vectorize,
)
from quant_fund.models.synthetic_gan import Discriminator, Generator, bench_synthetic_gan
from quant_fund.models.ts_diffusion import DiffusionSampler, bench_ts_diffusion


def test_vectorize_norm():
    v = vectorize("bullish surge growth strong")
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_echo_detects_copy():
    stmt = vectorize("bearish risk weak downgrade")
    echo = vectorize("bearish risk weak downgrade reported")
    other = vectorize("bullish surge growth record")
    assert echo_score(echo, np.array([stmt])) > echo_score(other, np.array([stmt]))


def test_saydo_tracker_cov():
    trk = SayDoTracker(window=10)
    for i in range(20):
        trk.update(0, float(np.sin(i)), float(-np.sin(i)))
    assert trk.covariance(0) < -0.5


def test_gated_fusion_learns():
    rng = np.random.default_rng(0)
    y = (rng.random(200) > 0.5).astype(float)
    s = np.column_stack([2 * y - 1 + 0.1 * rng.standard_normal(200), rng.standard_normal(200)])
    fus = GatedFusion()
    fus.fit(s, y)
    p = fus.predict(s)
    assert np.corrcoef(p, y)[0, 1] > 0.5


def test_feats_shapes():
    rng = np.random.default_rng(0)
    px = np.cumprod(1 + 0.01 * rng.standard_normal(100)) * 100
    assert price_feats(px).shape == (4,)
    assert chart_feats(px).shape == (4,)
    assert text_feats("hello world").shape == (16,)


def test_diffusion_sample_shape():
    rng = np.random.default_rng(0)
    wins = rng.standard_normal((50, 8))
    ds = DiffusionSampler(8, T=10)
    ds.train(wins, 1, rng)
    assert ds.sample(rng).shape == (8,)


def test_gan_shapes():
    g = Generator(4, 8)
    d = Discriminator(8)
    rng = np.random.default_rng(0)
    f = g.forward(rng.standard_normal(4))
    assert f.shape == (8,)
    logit, _ = d.forward(f)
    assert np.isfinite(logit)


def test_quantcode_compile_backtest():
    rng = np.random.default_rng(0)
    spec = StrategySpec("t", "mom", "momentum", {"window": 5})
    code = compile_spec(spec)
    assert not static_check(code)
    res = backtest(code, synth_bars(200, rng), spec.params)
    assert res["n_trades"] > 0


def test_quantcode_rejects_bad():
    with pytest.raises(ValueError):
        compile_spec(StrategySpec("x", "x", "hack", {}))
    assert static_check("import os\ndef f(): pass")


def test_econ_impact_recovers():
    rng = np.random.default_rng(0)
    names = ["cpi"] * 100
    z = rng.standard_normal(100)
    mv = 15.0 * z + 0.1 * rng.standard_normal(100)
    coefs = fit_impact(names, z, mv)
    assert abs(coefs["cpi"] - 15.0) < 1.0


def test_benches():
    for fn in (
        bench_say_echo_do,
        bench_multimodal_fusion,
        bench_ts_diffusion,
        bench_synthetic_gan,
        bench_econ_calendar,
        bench_quantcode_bench,
    ):
        out = fn()
        assert out and all(k.startswith("synthetic_") for k in out)
