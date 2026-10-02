"""Surrogate-gradient SNN (Neftci et al. 2019) — 1-hidden-layer SNN
trained via arctan-surrogate spike gradients on Poisson-encoded inputs
vs a same-size ANN; reports acc and mean spike count per sample.
"""

from __future__ import annotations

from quant_fund.models._sk_synth import poisson_encode, sk_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("surrogate_snn requires torch (pip install -e .[nn])") from exc
    return torch


def bench_surrogate_snn(seed: int = 1913, iters: int = 500, T: int = 20) -> dict[str, float]:
    torch = _torch()
    X, y = sk_data(seed)
    sp = poisson_encode(X, seed=seed)[:, :, :T]
    Xt, yt = sk_data(seed + 1, n=200)
    spt = poisson_encode(Xt, seed=seed + 1)[:, :, :T]
    torch.manual_seed(seed)
    H = 16
    W1 = torch.nn.Linear(8, H)
    W2 = torch.nn.Linear(H, 1)
    opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=0.01)
    sp_t = torch.tensor(sp).float()
    yt_ = torch.tensor(y).float()

    from torch.autograd import Function as _Fn

    class Spk(_Fn):
        @staticmethod
        def forward(ctx, v):
            return (v > 0).float()

        @staticmethod
        def backward(ctx, g):
            return g  # straight-through surrogate

    for _ in range(iters):
        v1 = torch.zeros(len(sp), H)
        out_sum = torch.zeros(len(sp))
        for t in range(T):
            v1 = 0.8 * v1 + W1(sp_t[:, :, t])
            s1 = Spk.apply(v1)
            v1 = v1 * (1 - s1)  # reset
            out_sum = out_sum + W2(s1).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out_sum / T, yt_)
        opt.zero_grad()
        loss.backward()
        opt.step()
    spt_ = torch.tensor(spt).float()
    with torch.no_grad():
        v1 = torch.zeros(len(spt), H)
        out_sum = torch.zeros(len(spt))
        spk_cnt = 0.0
        for t in range(T):
            v1 = 0.8 * v1 + W1(spt_[:, :, t])
            s1 = (v1 > 0).float()
            v1 = v1 * (1 - s1)
            spk_cnt += float(s1.mean())
            out_sum += W2(s1).squeeze(-1)
        acc_snn = float(((out_sum / T > 0).float().numpy() == yt).mean())
    # ANN baseline on raw
    Xr, yr = sk_data(seed)
    Xtr, ytr = sk_data(seed + 1, n=200)
    ann = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt2 = torch.optim.Adam(ann.parameters(), lr=0.01)
    Xt_, yt2 = torch.tensor(Xr).float(), torch.tensor(yr).float()
    for _ in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(ann(Xt_).squeeze(-1), yt2)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        acc_ann = float(
            ((ann(torch.tensor(Xtr).float()).squeeze(-1) > 0).float().numpy() == ytr).mean()
        )
    return {
        "synthetic_snn_acc": acc_snn,
        "synthetic_snn_ann_acc": acc_ann,
        "synthetic_snn_gap": acc_snn - acc_ann,
        "synthetic_snn_spike_rate": spk_cnt / T,
        "torch_available": 1.0,
    }
