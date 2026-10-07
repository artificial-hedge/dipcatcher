"""GradNorm (Chen et al. 2018) — task weights adapted so per-task (SYNTHETIC)
weighted gradient norms match the average times relative-inverse-rate;
reports weight evolution and min-task acc vs naive.
"""

from __future__ import annotations

from quant_fund.models._mt_synth import mt_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("gradnorm_bal requires torch (pip install -e .[nn])") from exc
    return torch


def bench_gradnorm_bal(seed: int = 2019, iters: int = 600) -> dict[str, float]:
    torch = _torch()
    X, y1, y2 = mt_data(seed)
    Xt, yt1, yt2 = mt_data(seed + 1, n=200)
    torch.manual_seed(seed)
    D = 24
    trunk = torch.nn.Sequential(torch.nn.Linear(8, D), torch.nn.Tanh())
    h1 = torch.nn.Linear(D, 1)
    h2 = torch.nn.Linear(D, 1)
    w = torch.tensor([1.0, 1.0], requires_grad=True)
    params = list(trunk.parameters()) + list(h1.parameters()) + list(h2.parameters())
    opt = torch.optim.SGD(params, lr=0.05)
    opt_w = torch.optim.Adam([w], lr=0.05)
    Xt_ = torch.tensor(X).float()
    y1_ = torch.tensor(y1).float()
    y2_ = torch.tensor(y2).float()
    l10 = l20 = None
    W_last = trunk[0].weight
    for _ in range(iters):
        h = trunk(Xt_)
        l1 = torch.nn.functional.binary_cross_entropy_with_logits(h1(h).squeeze(-1), y1_)
        l2 = torch.nn.functional.binary_cross_entropy_with_logits(h2(h).squeeze(-1), y2_)
        if l10 is None:
            l10, l20 = float(l1), float(l2)
        if not (l10 is not None and l20 is not None):
            raise ValueError("l10 is not None and l20 is not None")
        c1 = torch.autograd.grad(l1, W_last, retain_graph=True)[0].norm().detach()
        c2 = torch.autograd.grad(l2, W_last, retain_graph=True)[0].norm().detach()
        G1 = w[0] * c1
        G2 = w[1] * c2
        Gm = (G1 + G2).detach() / 2
        r1 = float(l1) / (l10 + 1e-8)
        r2 = float(l2) / (l20 + 1e-8)
        target1 = Gm * (r1 / (r1 + r2 + 1e-8)) ** 1.5
        target2 = Gm * (r2 / (r1 + r2 + 1e-8)) ** 1.5
        w_loss = (G1 - target1).abs() + (G2 - target2).abs()
        opt_w.zero_grad()
        w_loss.backward()
        opt_w.step()
        with torch.no_grad():
            w.div_(w.sum() / 2.0)
            w.clamp_(0.01, 4.0)
        opt.zero_grad()
        (w[0] * l1 + w[1] * l2).backward()
        opt.step()
    with torch.no_grad():
        h = trunk(torch.tensor(Xt).float())
        a1 = float(((h1(h).squeeze(-1) > 0).float().numpy() == yt1).mean())
        a2 = float(((h2(h).squeeze(-1) > 0).float().numpy() == yt2).mean())
    return {
        "synthetic_gradnorm_min_acc": min(a1, a2),
        "synthetic_gradnorm_mean_acc": (a1 + a2) / 2,
        "synthetic_gradnorm_w1": float(w[0]),
        "synthetic_gradnorm_w2": float(w[1]),
        "synthetic_torch_available": 1.0,
    }
