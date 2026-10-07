"""Task-vector model merging (Ilharco et al. 2023).

τ_i = θ_i − θ_0 is the task vector; merging adds scaled task vectors to
the base: θ_merge = θ_0 + λ Σ τ_i. On two rotated tasks, the merged
model retains both tasks where full fine-tuning on one forgets the
other.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._peft_synth import synth_peft_base, synth_peft_shift

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("task_vector_merge needs the torch `nn` extra") from exc


def _train_on(base, xs, ys, iters: int) -> None:
    torch = _torch()
    opt = torch.optim.Adam(base.parameters(), lr=5e-3)
    xt, yt = torch.tensor(xs).float(), torch.tensor(ys)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(base(xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()


def bench_task_vector_merge(
    seed: int = 149,
    n_train: int = 400,
    n_shift: int = 200,
    iters_base: int = 400,
    iters_adapt: int = 300,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = synth_peft_base(n_train, rng)
    xs1, ys1 = synth_peft_shift(n_shift, np.random.default_rng(seed + 1), rot=0.6)
    xs2, ys2 = synth_peft_shift(n_shift, np.random.default_rng(seed + 2), rot=-0.6)

    def net():
        return torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    base = net()
    _train_on(base, x, y, iters_base)
    import copy

    m1 = copy.deepcopy(base)
    _train_on(m1, xs1, ys1, iters_adapt)
    m2 = copy.deepcopy(base)
    _train_on(m2, xs2, ys2, iters_adapt)
    merged = copy.deepcopy(base)
    lam = 0.7
    with torch.no_grad():
        for pm, p0, p1, p2 in zip(
            merged.parameters(), base.parameters(), m1.parameters(), m2.parameters(), strict=True
        ):
            pm.copy_(p0 + lam * ((p1 - p0) + (p2 - p0)))
    x1_t = torch.tensor(xs1).float()
    x2_t = torch.tensor(xs2).float()
    y1_t = torch.tensor(ys1)
    y2_t = torch.tensor(ys2)
    with torch.no_grad():
        acc_m_t1 = float((merged(x1_t).argmax(-1) == y1_t).float().mean())
        acc_m_t2 = float((merged(x2_t).argmax(-1) == y2_t).float().mean())
        acc_1_t1 = float((m1(x1_t).argmax(-1) == y1_t).float().mean())
        acc_1_t2 = float((m1(x2_t).argmax(-1) == y2_t).float().mean())
        acc_2_t2 = float((m2(x2_t).argmax(-1) == y2_t).float().mean())
        acc_base_t1 = float((base(x1_t).argmax(-1) == y1_t).float().mean())
    return {
        "synthetic_merge_t1_acc": acc_m_t1,
        "synthetic_merge_t2_acc": acc_m_t2,
        "synthetic_merge_mean_acc": (acc_m_t1 + acc_m_t2) / 2,
        "synthetic_merge_ft1_t1": acc_1_t1,
        "synthetic_merge_ft1_t2": acc_1_t2,
        "synthetic_merge_ft2_t2": acc_2_t2,
        "synthetic_merge_base_t1": acc_base_t1,
        "synthetic_torch_available": 1.0,
    }
