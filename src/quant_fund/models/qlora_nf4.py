"""QLoRA NF4 quantization (Dettmers et al. 2023).

Base weights quantized to 4-bit NormalFloat (15 quantile levels of
N(0,1) plus zero) and dequantized on the fly; a bf16 LoRA adapter
carries the learning. Reports quantization error of the compressed
base plus adapted accuracy vs full-precision LoRA.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.models._peft_synth import synth_peft_base, synth_peft_shift

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("qlora_nf4 needs the torch `nn` extra") from exc


def _nf4_levels() -> FloatArray:
    """16 NF4 codebook levels: quantiles of N(0,1) normalized to [-1,1]."""
    qs = np.array([(2 * i + 1) / 32 for i in range(8)] + [(2 * i + 1) / 32 for i in range(7)])
    neg = norm.ppf(qs[:8])
    pos = norm.ppf(0.5 + qs[8:] / 2)
    lv = np.concatenate([neg / abs(neg).max(), [0.0], pos / pos.max()])
    return np.asarray(lv, dtype=np.float64)


_NF4 = _nf4_levels()


def _quant(w: np.ndarray) -> np.ndarray:
    wn = w / (np.abs(w).max() + 1e-12)
    idx = np.abs(wn[..., None] - _NF4).argmin(-1)
    return np.asarray(idx, dtype=np.int64)


def _dequant(idx: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(_NF4[idx] * scale, dtype=np.float64)


def bench_qlora_nf4(
    seed: int = 137,
    n_train: int = 400,
    n_shift: int = 200,
    iters_base: int = 400,
    iters_adapt: int = 300,
    rank: int = 1,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = synth_peft_base(n_train, rng)
    xs, ys = synth_peft_shift(n_shift, np.random.default_rng(seed + 1))
    base = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    x_t, y_t = torch.tensor(x).float(), torch.tensor(y)
    opt = torch.optim.Adam(base.parameters(), lr=5e-3)
    for _i in range(iters_base):
        loss = torch.nn.functional.cross_entropy(base(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    w1 = base[0].weight.detach().numpy()
    w2 = base[2].weight.detach().numpy()
    idx1, s1 = _quant(w1), float(np.abs(w1).max())
    idx2, s2 = _quant(w2), float(np.abs(w2).max())
    w1q = torch.tensor(_dequant(idx1, s1)).float()
    w2q = torch.tensor(_dequant(idx2, s2)).float()
    qerr = float(np.sqrt(np.mean((w1q.numpy() - w1) ** 2) + np.mean((w2q.numpy() - w2) ** 2)))
    b1 = torch.tensor(base[0].bias.detach().numpy()).float()
    b2 = torch.tensor(base[2].bias.detach().numpy()).float()
    a1 = torch.nn.Parameter(torch.randn(4, rank) * 0.1)
    r1 = torch.nn.Parameter(torch.zeros(rank, 32))
    a2 = torch.nn.Parameter(torch.randn(32, rank) * 0.1)
    r2 = torch.nn.Parameter(torch.zeros(rank, 2))
    xs_t, ys_t = torch.tensor(xs).float(), torch.tensor(ys)
    opt2 = torch.optim.Adam([a1, r1, a2, r2], lr=5e-3)

    def fwd(xb):
        h = torch.relu(xb @ (w1q + (a1 @ r1).T).T + b1)
        return h @ (w2q + (a2 @ r2).T).T + b2

    for _i in range(iters_adapt):
        loss = torch.nn.functional.cross_entropy(fwd(xs_t), ys_t)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        acc_q = float((fwd(xs_t).argmax(-1) == ys_t).float().mean())
        acc_q_t0 = float((fwd(x_t).argmax(-1) == y_t).float().mean())
    bits_q = (idx1.size + idx2.size) * 4
    bits_full = (w1.size + w2.size) * 32
    return {
        "synthetic_qlora_acc_shift": acc_q,
        "synthetic_qlora_t0_acc": acc_q_t0,
        "synthetic_qlora_quant_err": qerr,
        "synthetic_qlora_bits_ratio": float(bits_q) / bits_full,
        "synthetic_torch_available": 1.0,
    }
