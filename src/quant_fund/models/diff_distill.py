"""Progressive distillation (Salimans & Ho 2022) — train DDPM teacher, (SYNTHETIC)
then a student matching two teacher steps in one; student MMD at half
the steps vs teacher at full steps.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._dfx_synth import _mmd, gauss_mmd, synth_regime_windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("diff_distill requires torch (pip install -e .[nn])") from exc
    return torch


def bench_diff_distill(seed: int = 1529, iters: int = 900, steps: int = 16) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, _ = synth_regime_windows(400, 16, rng)
    Xtr, Xte = torch.tensor(X[:300]).float(), X[300:].astype(np.float64)
    teacher = torch.nn.Sequential(
        torch.nn.Linear(16 + 1, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16)
    )
    opt = torch.optim.Adam(teacher.parameters(), lr=0.01)
    T = 200
    betas = torch.linspace(1e-4, 0.05, T)
    ab = torch.cumprod(1 - betas, 0)
    for _ in range(iters):
        idx = rng.integers(0, len(Xtr), 64)
        x0 = Xtr[idx]
        t = torch.randint(0, T, (64,))
        tt = (t.float() / T)[:, None]
        eps = torch.randn_like(x0)
        xt = ab[t].sqrt()[:, None] * x0 + (1 - ab[t]).sqrt()[:, None] * eps
        loss = ((teacher(torch.cat([xt, tt], 1)) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def ddim_step(net, xs, t_cur, t_next):
        a_c, a_n = ab[t_cur], ab[t_next]
        tt = torch.full((len(xs), 1), t_cur / T)
        eps = net(torch.cat([xs, tt], 1))
        return a_n.sqrt() * (xs - (1 - a_c).sqrt() * eps) / a_c.sqrt() + (1 - a_n).sqrt() * eps

    # distill: student predicts one step for two teacher steps
    student = torch.nn.Sequential(
        torch.nn.Linear(16 + 1, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16)
    )
    opt2 = torch.optim.Adam(student.parameters(), lr=0.01)
    tgrid = np.linspace(T - 1, 0, steps + 1).astype(int)
    for _ in range(iters):
        i = int(rng.integers(0, steps - 1, 1)[0])
        t1, t3 = int(tgrid[i]), int(tgrid[min(i + 2, steps)])
        t = torch.randint(t3 + 1, t1 + 1, (64,))
        idx = rng.integers(0, len(Xtr), 64)
        x0 = Xtr[idx]
        eps = torch.randn_like(x0)
        xs = ab[t].sqrt()[:, None] * x0 + (1 - ab[t]).sqrt()[:, None] * eps
        with torch.no_grad():
            # two teacher steps to t3 region target: teacher ddim toward t3
            t_mid = (t + t3) // 2
            x_mid = torch.stack(
                [ddim_step(teacher, xs[j : j + 1], int(t[j]), int(t_mid[j])) for j in range(64)]
            ).squeeze(1)
            x_tgt = torch.stack(
                [ddim_step(teacher, x_mid[j : j + 1], int(t_mid[j]), t3) for j in range(64)]
            ).squeeze(1)
        # student one-step target toward t3 directly
        tt = (t.float() / T)[:, None]
        a_c, a_n = ab[t], ab[t3]
        eps_s = student(torch.cat([xs, tt], 1))
        pred = (
            a_n.sqrt() * (xs - (1 - a_c).sqrt()[:, None] * eps_s) / a_c.sqrt()[:, None]
            + (1 - a_n).sqrt() * eps_s
        )
        loss = ((pred - x_tgt) ** 2).mean()
        opt2.zero_grad()
        loss.backward()
        opt2.step()

    # sample: teacher at `steps` steps, student at steps//2
    def sample(net, nsteps, double=False):
        xs = torch.randn(150, 16)
        tg = np.linspace(T - 1, 0, nsteps + 1).astype(int)
        with torch.no_grad():
            for i in range(nsteps):
                xs = ddim_step(net, xs, int(tg[i]), int(tg[i + 1]))
        return xs.numpy().astype(np.float64)

    m_t = _mmd(sample(teacher, steps), Xte, bw=1.0)
    m_s = _mmd(sample(student, max(steps // 2, 2)), Xte, bw=1.0)
    g = gauss_mmd(np.random.default_rng(seed + 1), Xte)
    return {
        "synthetic_dd_teacher_mmd": m_t,
        "synthetic_dd_student_mmd": m_s,
        "synthetic_dd_student_gap": m_s - m_t,
        "synthetic_dd_gauss_mmd": g,
        "synthetic_torch_available": 1.0,
    }
