"""Patch-token classifier (ViT-style, Dosovitskiy 2021).

2x2 patches → linear embed → 1 transformer layer → CLS head.
Data-limited (n=120 train): ViT vs CNN gap — ViT underperforms
without pretraining scale (documented inductive-bias result).
"""

from __future__ import annotations

from quant_fund.models._vision_synth import patches, synth_images


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("vit_classifier requires torch (pip install -e .[nn])") from exc
    return torch


def bench_vit_classifier(
    seed: int = 509,
    n: int = 240,
    iters: int = 100,
    d: int = 24,
) -> dict[str, float]:
    torch = _torch()
    x, y = synth_images(seed, n)
    half = n // 2
    xtr = torch.tensor(patches(x[:half])).float()
    ytr = torch.tensor(y[:half])
    xte = torch.tensor(patches(x[half:])).float()
    yte = torch.tensor(y[half:])
    torch.manual_seed(seed)
    embed = torch.nn.Linear(4, d)
    cls_tok = torch.nn.Parameter(torch.zeros(1, 1, d))
    attn = torch.nn.MultiheadAttention(d, 2, batch_first=True)
    ff = torch.nn.Linear(d, d)
    head = torch.nn.Linear(d, 2)
    params = (
        list(embed.parameters())
        + list(attn.parameters())
        + list(ff.parameters())
        + list(head.parameters())
        + [cls_tok]
    )
    opt = torch.optim.Adam(params, lr=0.01)
    for _i in range(iters):
        tok = embed(xtr)
        tok = torch.cat([cls_tok.expand(len(tok), -1, -1), tok], 1)
        h, _w = attn(tok, tok, tok)
        out = head(torch.relu(ff(h[:, 0])))
        loss = torch.nn.functional.cross_entropy(out, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        tok = embed(xte)
        tok = torch.cat([cls_tok.expand(len(tok), -1, -1), tok], 1)
        h, _w = attn(tok, tok, tok)
        acc_v = float((head(torch.relu(ff(h[:, 0]))).argmax(1) == yte).float().mean())
    # CNN reference on same budget
    xg_tr = torch.tensor(x[:half]).float().unsqueeze(1)
    xg_te = torch.tensor(x[half:]).float().unsqueeze(1)
    torch.manual_seed(seed)
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 36, 2),
    )
    opt = torch.optim.Adam(cnn.parameters(), lr=0.01)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(cnn(xg_tr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_c = float((cnn(xg_te).argmax(1) == yte).float().mean())
    return {
        "synthetic_vit_acc": acc_v,
        "synthetic_vit_cnn_acc": acc_c,
        "synthetic_vit_gap": acc_v - acc_c,
        "synthetic_torch_available": 1.0,
    }
