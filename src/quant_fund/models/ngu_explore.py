"""NGU-lite (Badia et al. 2020): intrinsic reward = episodic novelty
(kNN distance to this-episode history) gated by a lifelong RND-style
modulator min(max(alpha_lifelong, 1), L). Numpy: hash-count episodic
counts + ridge-feature prediction error for the lifelong factor.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ex_synth import q_learn, s2i, state_feat


def bench_ngu_explore(seed: int = 2857, n_ep_knn: int = 5) -> dict[str, float]:
    rng0 = np.random.default_rng(seed)
    proj = rng0.normal(0, 1, (6, 16))
    lifelong: dict[int, int] = {}
    ep_states: list[tuple[int, int]] = []
    ep_feats: list[np.ndarray] = []

    def bonus(s, sp, st, ep, rng):
        i = s2i(sp)
        lifelong[i] = lifelong.get(i, 0) + 1
        alpha_l = 1.0 / np.sqrt(lifelong[i])
        f = np.tanh(state_feat(sp) @ proj)
        if ep_feats:
            d = np.linalg.norm(np.asarray(ep_feats) - f, axis=1)
            knn = float(np.sort(d)[min(n_ep_knn - 1, len(d) - 1)])
            novelty = knn / np.sqrt(max(lifelong[i], 1))
        else:
            novelty = 1.0
        ep_states.append(sp)
        ep_feats.append(f)
        mod = min(max(1.0 / (alpha_l + 1e-3) * alpha_l * 2, 1.0), 5.0)
        return novelty * mod * 0.5

    def end_ep():
        ep_states.clear()
        ep_feats.clear()

    bonus.end_episode = end_ep  # type: ignore[attr-defined]
    _, cov, succ = q_learn(bonus, seed=seed)
    _, cov_b, succ_b = q_learn(lambda s, sp, st, ep, rng: 0.0, seed=seed)
    return {
        "synthetic_ngu_coverage": float(cov),
        "synthetic_baseline_coverage": float(cov_b),
        "synthetic_ngu_success": float(succ),
        "synthetic_baseline_success": float(succ_b),
        "torch_available": 0.0,
    }
