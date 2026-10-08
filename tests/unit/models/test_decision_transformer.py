"""Unit tests for quant_fund.models.decision_transformer."""

from __future__ import annotations

import numpy as np
import pytest

import quant_fund.models.decision_transformer as dt_mod

torch = pytest.importorskip("torch")


def _modules(d: int = 8, horizon: int = 4):
    torch.manual_seed(0)
    emb_r = torch.nn.Linear(1, d)
    emb_s = torch.nn.Linear(4, d)
    emb_a = torch.nn.Embedding(3, d)
    head = torch.nn.Linear(d, 3)
    pos_emb = torch.nn.Parameter(torch.randn(1, horizon * 3, d) * 0.02)
    mask = torch.triu(torch.ones(horizon * 3, horizon * 3), 1).bool()
    return emb_r, emb_s, emb_a, head, pos_emb, mask


def test_policy_logits_passes_causal_mask() -> None:
    """Inference ran the encoder UNMASKED — the query state token
    attended to its own fabricated future action slot (and every
    earlier token saw future context). The mask must reach trf."""
    emb_r, emb_s, emb_a, head, pos_emb, mask = _modules()
    seen: dict[str, object] = {}

    class Spy(torch.nn.Module):
        def forward(self, toks, mask=None):  # noqa: ANN001
            seen["mask"] = mask
            return toks

    dt_mod._policy_logits(
        emb_r,
        emb_s,
        emb_a,
        Spy(),
        head,
        pos_emb,
        mask,
        [1.0, 0.5],
        [np.zeros(4), np.ones(4)],
        [0],
        8,
        torch,
    )
    m = seen["mask"]
    assert m is not None and m.shape == (6, 6)
    assert bool(m[2, 5])  # token 2 may not see token 5


def test_policy_logits_uses_action_history() -> None:
    """a_pad was always zeros — the actually-chosen actions never
    entered context. Different past actions must move the logits."""
    emb_r, emb_s, emb_a, head, pos_emb, mask = _modules()
    trf = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(8, 2, 16, batch_first=True), 1
    )
    rt_seq = [1.0, 0.5]
    s_seq = [np.zeros(4), np.ones(4)]
    l1 = dt_mod._policy_logits(
        emb_r,
        emb_s,
        emb_a,
        trf,
        head,
        pos_emb,
        mask,
        rt_seq,
        s_seq,
        [0],
        8,
        torch,
    )
    l2 = dt_mod._policy_logits(
        emb_r,
        emb_s,
        emb_a,
        trf,
        head,
        pos_emb,
        mask,
        rt_seq,
        s_seq,
        [2],
        8,
        torch,
    )
    assert not torch.allclose(l1, l2)


def test_bench_decision_transformer_keys() -> None:
    out = dt_mod.bench_decision_transformer(iters=5, n_ep=20, horizon=8)
    for k in (
        "synthetic_dt_reward",
        "synthetic_dt_dataset_reward",
        "synthetic_dt_random_reward",
    ):
        assert np.isfinite(out[k])
