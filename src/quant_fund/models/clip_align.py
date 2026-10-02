"""CLIP-style contrastive image-text alignment (Radford 2021).

Image encoder (patch features) + text encoder (class-word features)
trained with NT-Xent; retrieval accuracy of image→text vs random.
"""

from __future__ import annotations

from quant_fund.models._vision_synth import patches, synth_images


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("clip_align requires torch (pip install -e .[nn])") from exc
    return torch


def bench_clip_align(
    seed: int = 521,
    n: int = 300,
    iters: int = 150,
    d: int = 16,
) -> dict[str, float]:
    torch = _torch()
    x, y = synth_images(seed, n)
    half = n // 2
    xi = torch.tensor(patches(x[:half]).reshape(half, -1)).float()
    yi = torch.tensor(y[:half])
    xi_te = torch.tensor(patches(x[half:]).reshape(n - half, -1)).float()
    yi_te = torch.tensor(y[half:])
    torch.manual_seed(seed)
    enc_i = torch.nn.Sequential(torch.nn.Linear(36, d), torch.nn.ReLU(), torch.nn.Linear(d, d))
    txt_emb = torch.nn.Parameter(torch.randn(2, d))
    opt = torch.optim.Adam(list(enc_i.parameters()) + [txt_emb], lr=0.01)
    tau = 0.1
    for _i in range(iters):
        zi = torch.nn.functional.normalize(enc_i(xi), dim=1)
        zt = torch.nn.functional.normalize(txt_emb, dim=1)
        logits = zi @ zt.T / tau
        loss = torch.nn.functional.cross_entropy(logits, yi)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        zi = torch.nn.functional.normalize(enc_i(xi_te), dim=1)
        zt = torch.nn.functional.normalize(txt_emb, dim=1)
        acc = float(((zi @ zt.T).argmax(1) == yi_te).float().mean())
        # unaligned baseline: random image encoder sim
        zi_r = torch.nn.functional.normalize(torch.randn_like(zi), dim=1)
        acc_r = float(((zi_r @ zt.T).argmax(1) == yi_te).float().mean())
    return {
        "synthetic_clip_acc": acc,
        "synthetic_clip_unaligned": acc_r,
        "synthetic_clip_gain": acc - acc_r,
        "torch_available": 1.0,
    }
