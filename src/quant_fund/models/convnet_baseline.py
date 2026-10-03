"""Small CNN baseline — locality + weight sharing.

3x3 conv → relu → flatten → linear, vs a permutation-invariant MLP
on flattened pixels: the inductive-bias gap on the synth images.
"""

from __future__ import annotations

from quant_fund.models._vision_synth import synth_images


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("convnet_baseline requires torch (pip install -e .[nn])") from exc
    return torch


def bench_convnet_baseline(
    seed: int = 503,
    n: int = 400,
    iters: int = 80,
) -> dict[str, float]:
    torch = _torch()
    x, y = synth_images(seed, n)
    half = min(n // 2, 60)  # data-limited: inductive bias shows here
    x_tr, y_tr = x[:half], y[:half]
    x_te, y_te = x[half:], y[half:]
    x_tr_t = torch.tensor(x_tr).float().unsqueeze(1)
    x_te_t = torch.tensor(x_te).float().unsqueeze(1)
    y_tr_t = torch.tensor(y_tr)
    y_te_t = torch.tensor(y_te)

    def acc(net, xt):
        net.eval()
        with torch.no_grad():
            return float((net(xt).argmax(1) == y_te_t).float().mean())

    torch.manual_seed(seed)
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 36, 2),
    )
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(36, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2)
    )
    for net, xt in [(cnn, x_tr_t), (mlp, x_tr_t.reshape(-1, 1, 6, 6))]:
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        for _i in range(iters):
            loss = torch.nn.functional.cross_entropy(net(xt), y_tr_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
    acc_c = acc(cnn, x_te_t)
    acc_m = acc(mlp, x_te_t.reshape(-1, 1, 6, 6))
    return {
        "synthetic_cnn_acc": acc_c,
        "synthetic_cnn_mlp_acc": acc_m,
        "synthetic_cnn_gain": acc_c - acc_m,
        "torch_available": 1.0,
    }
