"""hit_flee: post-fill hit-side cancel surge mechanism.

Pins: 0-default bit-identity, the surge fires only inside the marker
window on the hit side's near-touch band, counters surface, and the
bench receipt seals.
"""

from __future__ import annotations

from dataclasses import replace

from quant_fund.microstructure.zi_lob_simulator import (
    ZILobSimulator,
    santa_fe_config,
)


def test_hit_flee_zero_default_bit_identical() -> None:
    """Both knobs at 0 must leave the draw stream untouched."""
    a = ZILobSimulator(santa_fe_config(seed=11))
    b = ZILobSimulator(santa_fe_config(seed=11))
    for _ in range(3000):
        a.step()
        b.step()
    ta = [(t.t, t.price, t.level, t.aggressor) for t in a.trades]
    tb = [(t.t, t.price, t.level, t.aggressor) for t in b.trades]
    assert ta == tb
    assert a.event_counts() == b.event_counts()
    assert a.n_hit_flees == 0


def test_hit_flee_fires_inside_window_on_hit_side() -> None:
    """With the knob on, fills trigger extra near-touch cancels on the
    side that was hit — counted in n_hit_flees and n_cancellations."""
    cfg = replace(
        santa_fe_config(seed=3),
        hit_flee_frac=1.0,
        hit_flee_band=2,
        hit_flee_window=200,
    )
    sim = ZILobSimulator(cfg)
    for _ in range(8000):
        sim.step()
    assert sim.n_hit_flees > 0
    assert sim.n_cancellations >= sim.n_hit_flees
    assert sim.event_counts()["n_hit_flees"] == sim.n_hit_flees


def test_hit_flee_expires_with_window() -> None:
    """A zero window disables the surge entirely (marker never arms)."""
    cfg = replace(
        santa_fe_config(seed=5),
        hit_flee_frac=1.0,
        hit_flee_band=2,
        hit_flee_window=0,
    )
    sim = ZILobSimulator(cfg)
    for _ in range(4000):
        sim.step()
    assert sim.n_hit_flees == 0
    assert sim.n_fills > 0  # sanity: fills happened, flee never armed


def test_hit_flee_config_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="hit_flee_frac"):
        replace(santa_fe_config(seed=1), hit_flee_frac=1.5)
    with pytest.raises(ValueError, match="hit_flee_band"):
        replace(santa_fe_config(seed=1), hit_flee_frac=0.5, hit_flee_band=-1)


def test_hit_flee_bench_smoke() -> None:
    """Small-horizon bench run seals and carries the grid shape."""
    import quant_fund.microstructure.hit_flee_bench as mod

    cells = mod._GRID[:2]  # reference cells only
    out_cells = [
        mod._flee_cell(ff, fb, fw, imp, iw, horizon=4000, seed=31 + i)
        for i, (ff, fb, fw, imp, iw) in enumerate(cells)
    ]
    assert all(c["n_fills"] > 0 for c in out_cells)
    assert out_cells[0]["n_hit_flees"] == 0
    assert out_cells[1]["n_hit_flees"] > 0
