import numpy as np
import pytest

from quant_fund.models.pesaran_cd import bench_pesaran_cd, pesaran_cd, synth_cd_panel


def test_rejects_under_factor() -> None:
    d = synth_cd_panel(lam=0.7, seed=0)
    out = pesaran_cd(np.asarray(d["resid"]))
    assert out["p"] < 0.001
    assert out["cd"] > 10


def test_accepts_independence() -> None:
    d = synth_cd_panel(lam=0.0, seed=1)
    out = pesaran_cd(np.asarray(d["resid"]))
    assert out["p"] > 0.05
    assert abs(out["cd"]) < 2


def test_rho_bar_matches_theory() -> None:
    d = synth_cd_panel(lam=0.7, seed=2)
    out = pesaran_cd(np.asarray(d["resid"]))
    assert abs(out["rho_bar"] - 0.7**2 / (1 + 0.7**2)) < 0.08


def test_weak_factor_middle() -> None:
    d = synth_cd_panel(lam=0.3, seed=3)
    out = pesaran_cd(np.asarray(d["resid"]))
    assert 0 < out["rho_bar"] < 0.2


def test_synth_shapes() -> None:
    d = synth_cd_panel(t=40, n=25, seed=4)
    assert np.asarray(d["resid"]).shape == (40, 25)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pesaran_cd(rng.normal(0, 1, (5, 40)))
    with pytest.raises(ValueError):
        pesaran_cd(rng.normal(0, 1, (40, 3)))
    with pytest.raises(ValueError):
        pesaran_cd(rng.normal(0, 1, (40, 20)) * np.nan)
    with pytest.raises(ValueError):
        bad = rng.normal(0, 1, (40, 20))
        bad[:, 0] = 0.0
        pesaran_cd(bad)


def test_bench() -> None:
    out = bench_pesaran_cd()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
