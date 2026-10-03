"""SimCLR — augmentation-invariance pretraining (Chen 2020).

Two noised views per image → NT-Xent; linear probe on frozen
representation vs probe on raw pixels — the augmentation gain.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from quant_fund.models._vision_synth import synth_images


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("simclr_views requires torch (pip install -e .[nn])") from exc
    return torch


def bench_simclr_views(
    seed: int = 523,
    n: int = 300,
    iters: int = 100,
    d: int = 16,
) -> dict[str, float]:
    torch = _torch()
    x, y = synth_images(seed, n)
    half = min(n // 2, 40)  # tiny-label regime
    xf = torch.tensor(x.reshape(n, -1)).float()
    torch.manual_seed(seed)
    enc = torch.nn.Sequential(torch.nn.Linear(36, d), torch.nn.ReLU(), torch.nn.Linear(d, d))
    proj = torch.nn.Linear(d, d)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=0.01)
    tau = 0.2
    gen = torch.Generator().manual_seed(seed)
    for _i in range(iters):
        v1 = xf + 0.5 * torch.randn(xf.shape, generator=gen)
        v2 = xf + 0.5 * torch.randn(xf.shape, generator=gen)
        z1 = torch.nn.functional.normalize(proj(enc(v1)), dim=1)
        z2 = torch.nn.functional.normalize(proj(enc(v2)), dim=1)
        logits = z1 @ z2.T / tau
        labels = torch.arange(n)
        loss = 0.5 * (
            torch.nn.functional.cross_entropy(logits, labels)
            + torch.nn.functional.cross_entropy(logits.T, labels)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = enc(xf).numpy()
    probe_ssl = LogisticRegression(max_iter=400).fit(z[:half], y[:half]).score(z[half:], y[half:])
    probe_raw = (
        LogisticRegression(max_iter=400)
        .fit(x[:half].reshape(half, -1), y[:half])
        .score(x[half:].reshape(n - half, -1), y[half:])
    )
    return {
        "synthetic_simclr_probe": probe_ssl,
        "synthetic_simclr_raw_probe": probe_raw,
        "synthetic_simclr_gain": probe_ssl - probe_raw,
        "torch_available": 1.0,
    }
