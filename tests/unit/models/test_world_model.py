import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quant_fund.models import world_model as wm  # noqa: E402


def test_transition_pairs_never_cross_episode_boundary() -> None:
    """A flat shift of the concatenated rollout buffer pairs each
    episode's terminal state with the next episode's opening state —
    ~10% of 'transitions' would be cross-episode hallucinations."""
    eps = wm.collect(4, np.random.default_rng(0))
    ns, na, nr, targ = wm._transition_pairs(eps, torch)
    assert ns.shape[0] == 4 * 9
    for e in range(4):
        for t in range(9):
            k = e * 9 + t
            assert np.allclose(targ[k].numpy(), eps[e]["s"][t + 1])
            assert np.allclose(ns[k].numpy(), eps[e]["s"][t])
            assert np.isclose(na[k].item(), eps[e]["a"][t])
            assert np.isclose(nr[k].item(), eps[e]["r"][t])


def test_exec_sim_step_respects_remaining() -> None:
    rng = np.random.default_rng(0)
    s = np.array([0.5, 0.0, 0.0])
    s2, _ = wm.exec_sim_step(s, 1.0, rng)
    assert s2[0] == pytest.approx(0.0)
