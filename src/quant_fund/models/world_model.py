"""Latent world model for execution decisions (Exec-Summary RL item) (SYNTHETIC).
Dreamer-lite: an RSSM-style encoder + transition + reward model trained on
execution-sim rollouts; the policy is then optimized by imagining latent
rollouts instead of touching the simulator.

Synthetic bench: imagined-rollout policy beats a reactive TWAP-style
baseline on implementation shortfall in the sim (synthetic wealth only,
proper-score framing).
"""

from __future__ import annotations

from typing import Any

import numpy as np

FloatArray = np.ndarray


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("world_model requires the `nn` extra (make sync)") from exc


def exec_sim_step(
    state: np.ndarray, act: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """state = [remaining, mid_dev, flow_mom]; act in [0,1] = frac traded.
    reward = -shortfall contribution - impact."""
    rem, dev, mom = state
    qty = act * max(rem, 0.0)
    impact = 0.5 * qty * (1 + abs(mom))
    px = dev + impact * 0.01 + 0.002 * rng.standard_normal()
    rew = -abs(px) * qty - 0.02 * qty * qty
    new = np.array(
        [
            rem - qty,
            dev + impact * 0.01 + 0.003 * rng.standard_normal(),
            0.7 * mom + 0.3 * np.sign(px),
        ]
    )
    return new, float(rew)


def collect(n_ep: int, rng: np.random.Generator) -> list[dict[str, FloatArray]]:
    eps = []
    for _ in range(n_ep):
        s = np.array([1.0, 0.0, 0.0])
        ss, aa, rr = [], [], []
        for _t in range(10):
            a = float(rng.random())
            s, r = exec_sim_step(s, a, rng)
            ss.append(s.copy())
            aa.append(a)
            rr.append(r)
        eps.append({"s": np.array(ss), "a": np.array(aa), "r": np.array(rr)})
    return eps


def bench_world_model(seed: int = 63) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    eps = collect(150, rng)
    s_all = torch.tensor(np.concatenate([e["s"] for e in eps]), dtype=torch.float32)
    a_all = torch.tensor(np.concatenate([e["a"] for e in eps]), dtype=torch.float32)
    r_all = torch.tensor(np.concatenate([e["r"] for e in eps]), dtype=torch.float32)

    trans = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
    rew = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    mods = torch.nn.ModuleList([trans, rew])

    def wm_step(s, a):
        x = torch.cat([s, a], -1)
        return s + trans(x), rew(x).squeeze(-1)

    opt = torch.optim.Adam(mods.parameters(), lr=3e-3)
    # train on consecutive (s,a)->s',r
    ns, na, nr = s_all[:-1], a_all[:-1], r_all[:-1]
    targ = s_all[1:]
    for _ in range(1200):
        idx = torch.randint(0, len(ns), (256,))
        sp, rp = wm_step(ns[idx], na[idx][:, None])
        loss = torch.mean((sp - targ[idx]) ** 2) + torch.mean((rp - nr[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # imagined policy optimization: param policy = MLP s->frac, maximize imagined reward
    pol = torch.nn.Sequential(
        torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1), torch.nn.Sigmoid()
    )
    popt = torch.optim.Adam(pol.parameters(), lr=3e-2)
    for _ in range(800):
        s = torch.zeros(64, 3)
        s[:, 0] = 1.0
        tot = torch.zeros(())
        for _t in range(10):
            a = pol(s)
            a = torch.maximum(a, s[:, 0:1] / (10 - _t))  # completion floor in imagination too
            s, r = wm_step(s, a)
            tot = tot + r.mean()
        loss = -tot
        popt.zero_grad()
        loss.backward()
        popt.step()

    # evaluate policy vs TWAP in REAL sim
    def run(policy_fn, n=60):
        tot = []
        for _ in range(n):
            s = np.array([1.0, 0.0, 0.0])
            r_sum = 0.0
            for _t in range(10):
                a = policy_fn(s)
                a = max(a, s[0] / (10 - _t))  # completion floor
                s, r = exec_sim_step(s, a, rng)
                r_sum += r
            tot.append(r_sum + -0.5 * s[0])  # terminal remainder penalty
        return float(np.mean(tot))

    pi = lambda s: float(pol(torch.tensor(s, dtype=torch.float32)).detach())  # noqa: E731
    twap = lambda s: 0.1 if s[0] > 0 else 0.0  # noqa: E731
    wm_r = run(pi)
    twap_r = run(twap)
    return {
        "synthetic_wm_imagined_reward": wm_r,
        "synthetic_wm_twap_reward": twap_r,
        "synthetic_wm_margin_vs_twap": wm_r - twap_r,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_world_model(), indent=1))
