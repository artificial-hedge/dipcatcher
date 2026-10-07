"""Determinism/control probes for bench_sac_agent (torch RNG + shared init)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="nn extra not installed")

from quant_fund.models.sac_agent import bench_sac_agent  # noqa: E402


def test_bench_sac_agent_deterministic_under_seed() -> None:
    a = bench_sac_agent(seed=20261231, steps=600, batch=64)
    b = bench_sac_agent(seed=20261231, steps=600, batch=64)
    for key, val in a.items():
        assert val == pytest.approx(b[key], rel=0.0, abs=0.0), key


def test_bench_sac_agent_seed_changes_output() -> None:
    a = bench_sac_agent(seed=20261231, steps=600, batch=64)
    b = bench_sac_agent(seed=999, steps=600, batch=64)
    changed = [k for k in a if not np.isclose(a[k], b[k], rtol=1e-9)]
    assert changed, "bench produced identical output under different seeds"


def test_arms_share_initial_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both run() arms must start from identical module inits; seeding torch
    # inside run() before make() is what guarantees it. If the seed moved or
    # was dropped, the two inits diverge and the margin is confounded.
    inits: list[np.ndarray] = []
    orig_sequential = torch.nn.Sequential

    class SpySeq(orig_sequential):
        def __init__(self, *mods):  # type: ignore[no-untyped-def]
            super().__init__(*mods)
            p = next(self.parameters(), None)
            inits.append(p.detach().numpy().copy() if p is not None else np.array([]))

    monkeypatch.setattr(torch.nn, "Sequential", SpySeq)
    bench_sac_agent(seed=20261231, steps=220, batch=32)
    # run(True) and run(False) each build 6 modules in the same order.
    assert len(inits) == 12
    np.testing.assert_allclose(inits[0], inits[6], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(inits[5], inits[11], rtol=0.0, atol=0.0)
