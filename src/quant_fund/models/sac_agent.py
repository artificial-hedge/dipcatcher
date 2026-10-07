"""SAC: soft actor-critic with entropy regularization and twin Q.

Haarnoja et al. 2018: maximize reward + entropy; twin critics clipped
by the min reduce overestimation; continuous Gaussian actor with
tanh squashing. Distinct from w117's DDPG/TD3 (deterministic actors).

Bench: continuous-action position MDP (a in [-1,1]); SAC vs a
deterministic actor trained with the same critics (DDPG-lite).
Metric: SYNTHETIC mean reward; entropy diagnostics.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sac_agent requires the `nn` extra (make sync)") from exc


def _env_step(state: FloatArray, a: float, rng: np.random.Generator) -> tuple[FloatArray, float]:
    r_prev, r_prev2, pos, sig = state
    new_pos = float(np.clip(a, -1, 1))
    r_next = 0.6 * np.tanh(2.0 * sig) + 0.3 * rng.standard_normal()
    reward = new_pos * r_next - 0.05 * abs(new_pos - pos)
    new_sig = 0.7 * sig + 0.4 * rng.standard_normal()
    return np.array([r_next, r_prev, new_pos, new_sig]), float(reward)


def bench_sac_agent(
    seed: int = 20261231,
    steps: int = 4000,
    batch: int = 128,
) -> dict[str, float]:
    """SAC online vs deterministic-actor baseline; SYNTHETIC."""
    torch = _torch()
    rng = np.random.default_rng(seed)

    def make() -> tuple[Any, Any, Any]:
        q1 = torch.nn.Sequential(torch.nn.Linear(5, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        q2 = torch.nn.Sequential(torch.nn.Linear(5, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        pi = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
        return q1, q2, pi

    def run(stochastic: bool) -> tuple[float, float]:
        # Seed torch per arm: module inits and policy noise reproduce, and
        # both arms start from the SAME initial weights — otherwise the
        # SAC-vs-deterministic margin is confounded by the init draw.
        torch.manual_seed(seed)
        q1, q2, pi = make()
        tq1, tq2, _ = make()
        for a_, b_ in zip(tq1.parameters(), q1.parameters(), strict=True):
            a_.data.copy_(b_.data)
        for a_, b_ in zip(tq2.parameters(), q2.parameters(), strict=True):
            a_.data.copy_(b_.data)
        opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
        opt_a = torch.optim.Adam(pi.parameters(), lr=1e-3)
        log_alpha = torch.tensor(-1.0, requires_grad=True)
        opt_al = torch.optim.Adam([log_alpha], lr=1e-3)
        buf: list[tuple[Any, float, float, Any]] = []
        s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])

        def act(st: FloatArray) -> tuple[float, float]:
            torch_t = torch.tensor(st, dtype=torch.float32)[None]
            mu, log_sd = pi(torch_t)[0].split(1)
            sd = torch.exp(torch.clamp(log_sd, -2, 1))
            if stochastic:
                eps = torch.randn_like(mu)
                a_raw = mu + sd * eps
                lp = float(
                    torch.sum(
                        -0.5 * (eps**2) - log_sd - torch.log(1 - torch.tanh(a_raw) ** 2 + 1e-6)
                    ).item()
                )
                return float(torch.tanh(a_raw).item()), lp
            return float(torch.tanh(mu).item()), 0.0

        for t in range(steps):
            a, lp = act(s)
            sn, r = _env_step(s, a, rng)
            buf.append((s.copy(), a, r, sn.copy()))
            if len(buf) > 4000:
                buf.pop(0)
            s = sn
            if len(buf) < 200:
                continue
            bi = rng.integers(0, len(buf), batch)
            S = torch.tensor(np.array([buf[i][0] for i in bi]), dtype=torch.float32)
            Ac = torch.tensor(np.array([buf[i][1] for i in bi]), dtype=torch.float32).unsqueeze(1)
            Rr = torch.tensor(np.array([buf[i][2] for i in bi]), dtype=torch.float32)
            Sn = torch.tensor(np.array([buf[i][3] for i in bi]), dtype=torch.float32)
            with torch.no_grad():
                out = pi(Sn)
                mu, log_sd = out[:, 0], out[:, 1]
                sd = torch.exp(torch.clamp(log_sd, -2, 1))
                if stochastic:
                    an = torch.tanh(mu + sd * torch.randn_like(mu))
                    lp_t = torch.sum(-0.5 * ((an - mu) / sd) ** 2 - log_sd, -1)
                    alpha = torch.exp(log_alpha)
                    tgt = Rr + 0.95 * (
                        torch.min(
                            tq1(torch.cat([Sn, an.unsqueeze(1)], 1)).squeeze(1),
                            tq2(torch.cat([Sn, an.unsqueeze(1)], 1)).squeeze(1),
                        )
                        - alpha * lp_t
                    )
                else:
                    an = torch.tanh(mu)
                    tgt = Rr + 0.95 * torch.min(
                        tq1(torch.cat([Sn, an.unsqueeze(1)], 1)).squeeze(1),
                        tq2(torch.cat([Sn, an.unsqueeze(1)], 1)).squeeze(1),
                    )
            lq = torch.mean((q1(torch.cat([S, Ac], 1)).squeeze(1) - tgt) ** 2) + torch.mean(
                (q2(torch.cat([S, Ac], 1)).squeeze(1) - tgt) ** 2
            )
            opt_c.zero_grad()
            lq.backward()
            opt_c.step()
            if t % 2 == 0:
                out = pi(S)
                mu, log_sd = out[:, 0], out[:, 1]
                sd = torch.exp(torch.clamp(log_sd, -2, 1))
                if stochastic:
                    aa = torch.tanh(mu + sd * torch.randn_like(mu))
                    lp_a = torch.sum(-0.5 * ((aa - mu) / sd) ** 2 - log_sd, -1)
                    aloss = torch.mean(
                        torch.exp(log_alpha) * lp_a
                        - q1(torch.cat([S, aa.unsqueeze(1)], 1)).squeeze(1)
                    )
                    aalpha = torch.mean(-torch.exp(log_alpha) * (lp_a.detach() + 1.0))
                    opt_al.zero_grad()
                    aalpha.backward()
                    opt_al.step()
                else:
                    aa = torch.tanh(mu)
                    aloss = -torch.mean(q1(torch.cat([S, aa.unsqueeze(1)], 1)))
                opt_a.zero_grad()
                aloss.backward()
                opt_a.step()
                for a_, b_ in zip(tq1.parameters(), q1.parameters(), strict=True):
                    a_.data.copy_(0.995 * a_.data + 0.005 * b_.data)
                for a_, b_ in zip(tq2.parameters(), q2.parameters(), strict=True):
                    a_.data.copy_(0.995 * a_.data + 0.005 * b_.data)
        tot = 0.0
        ent = float(torch.exp(log_alpha).item())
        # Common-random-numbers eval: both arms score the same seeded
        # episode stream, so the margin reflects policy, not eval draws.
        eval_rng = np.random.default_rng(seed + 0x5AC)
        for _e in range(16):
            s = np.array([0.0, 0.0, 0.0, eval_rng.standard_normal()])
            for _t in range(16):
                a, _ = act(s)
                s, r = _env_step(s, a, eval_rng)
                tot += r
        return tot / 16, ent

    rew_sac, alpha = run(True)
    rew_det, _ = run(False)
    return {
        "synthetic_sac_reward": rew_sac,
        "synthetic_sac_det_reward": rew_det,
        "synthetic_sac_margin_vs_det": rew_sac - rew_det,
        "synthetic_sac_alpha": alpha,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_sac_agent()))
