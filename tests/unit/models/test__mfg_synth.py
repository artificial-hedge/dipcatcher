"""Invariant probes for quant_fund.models._mfg_synth (constants module)."""

from __future__ import annotations

import numpy as np

from quant_fund.models import _mfg_synth as m


def test_stochastic_game_transition_rows_are_distributions() -> None:
    for s in range(2):
        p = m.SG_P[s]
        assert p.shape == (2, 2, 2)
        assert np.allclose(p.sum(-1), 1.0)
        assert (p >= 0).all()


def test_stochastic_game_shapes() -> None:
    assert len(m.SG_M) == 2
    assert all(mm.shape == (2, 2) for mm in m.SG_M)
    assert 0 < m.SG_GAMMA < 1


def test_cournot_costs_sorted_planted() -> None:
    assert m.COU_A > 0 and m.COU_B > 0
    assert m.COU_C.shape == (4,)
    assert (m.COU_C > 0).all()
    assert (m.COU_C < m.COU_A).all()  # every firm can price above cost


def test_lq_constants_sane() -> None:
    assert m.LQ_Q > 0 and m.LQ_R > 0 and m.LQ_SIG > 0
    assert m.LQ_H > 0


def test_congestion_params() -> None:
    assert m.POT_A.shape == m.POT_B.shape == (4,)
    assert (m.POT_A > 0).all()  # latency increasing in load
    assert m.POT_N > 0
