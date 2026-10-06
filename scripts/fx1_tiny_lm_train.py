"""Build the committed tiny fx-1 fixture checkpoint at ``artifacts/fx1_tiny_lm/``.

Trains a one-block byte-level transformer (~31k params, dim 32, 4 heads,
ctx 64) on the message contents of the repo's own ``fx1_seed_corpus.jsonl``
for a few hundred AdamW steps on CPU, then writes a REAL loadable
checkpoint:

* ``weights.safetensors`` — float32 tensors named after the torch
  state_dict (``tok_emb.weight``, ``blocks.0.attn.qkv.weight``, …),
* ``weights.manifest.json`` — sha256 pin of the weight file plus
  architecture / training provenance,
* ``modelcard.json`` — a ship-gate-passing ``fx1.modelcard.ModelCard``.

The serving path (``fx1.serve.local_engine``) reproduces the same forward
math in numpy, so what was trained here is what gets served — real weights
loaded through real deserialization, not a stub. This artifact exists so
the golden path's ``weights_direct_ran`` leg measures a real load; it is a
fixture-scale model, not an fx-1 release candidate (the card says so).

Determinism: fixed seed, fixed corpus file, fixed step schedule. CPU BLAS
reduction order is not byte-stable across machines, so regeneration may
produce a bit-different artifact with equivalent behavior; the committed
file is the pinned one.

Usage::

    uv run --extra nn python scripts/fx1_tiny_lm_train.py [--out-dir artifacts/fx1_tiny_lm]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

VOCAB_SIZE = 259  # 256 byte-value tokens + BOS + EOS + PAD
BOS_ID = 256
EOS_ID = 257

_DIM = 32
_CTX = 64
_N_HEADS = 4
_MLP_RATIO = 4
_STEPS = 3000
_LR = 3e-3
_SEED = 0
_CORPUS_FILE = "fx1_seed_corpus.jsonl"


def _corpus_text(repo_root: Path) -> str:
    """The training corpus: every message string in the committed seed corpus."""
    path = repo_root / _CORPUS_FILE
    parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for m in row.get("messages", []):
            content = m.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
    text = "\n".join(parts)
    if len(text.encode("utf-8")) < 512:
        raise RuntimeError(f"corpus {path} too small to train the fixture ({len(text)} chars)")
    return text


def _build_torch_model(vocab: int, dim: int, ctx: int, n_heads: int) -> torch.nn.Module:
    """The torch twin of ``fx1.serve.local_engine._TinyLM`` — identical math."""
    import torch
    from torch import nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = nn.ModuleDict({"qkv": nn.Linear(dim, 3 * dim), "proj": nn.Linear(dim, dim)})
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = nn.ModuleDict(
                {
                    "fc1": nn.Linear(dim, _MLP_RATIO * dim),
                    "fc2": nn.Linear(_MLP_RATIO * dim, dim),
                }
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            n = x.shape[1]
            dh = dim // n_heads
            a = self.ln1(x)
            qkv = self.attn["qkv"](a)
            q, k, v = qkv.split(dim, dim=-1)
            qh = q.reshape(1, n, n_heads, dh).transpose(1, 2)
            kh = k.reshape(1, n, n_heads, dh).transpose(1, 2)
            vh = v.reshape(1, n, n_heads, dh).transpose(1, 2)
            scores = qh @ kh.transpose(-2, -1) / math.sqrt(dh)
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            attended = (torch.softmax(scores, dim=-1) @ vh).transpose(1, 2).reshape(1, n, dim)
            x = x + self.attn["proj"](attended)
            m = self.ln2(x)
            x = x + self.mlp["fc2"](nn.functional.gelu(self.mlp["fc1"](m), approximate="tanh"))
            return x

    class _TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(vocab, dim)
            self.pos_emb = nn.Embedding(ctx, dim)
            self.blocks = nn.ModuleList([_Block()])
            self.ln_f = nn.LayerNorm(dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            x = self.tok_emb(ids) + self.pos_emb.weight[: ids.shape[1]]
            for blk in self.blocks:
                x = blk(x)
            return self.lm_head(self.ln_f(x))

    return _TinyLM()


def _train(repo_root: Path) -> tuple[torch.nn.Module, float, str]:
    """Seconds-scale CPU training; returns (model, final_loss, corpus_sha256)."""
    import torch
    from torch import nn

    torch.manual_seed(_SEED)
    text = _corpus_text(repo_root)
    corpus_sha = hashlib.sha256(text.encode()).hexdigest()
    ids = torch.tensor([BOS_ID, *text.encode("utf-8")], dtype=torch.long)
    model = _build_torch_model(VOCAB_SIZE, _DIM, _CTX, _N_HEADS)
    opt = torch.optim.AdamW(model.parameters(), lr=_LR)
    gen = torch.Generator().manual_seed(_SEED)
    n_positions = ids.numel() - _CTX - 1
    final_loss = float("nan")
    for _ in range(_STEPS):
        start = int(torch.randint(n_positions, (1,), generator=gen).item())
        window = ids[start : start + _CTX + 1].unsqueeze(0)
        logits = model(window[:, :-1])
        loss = nn.functional.cross_entropy(logits[0], window[0, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
    return model, final_loss, corpus_sha


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_checkpoint(
    out_dir: Path, model: torch.nn.Module, final_loss: float, corpus_sha: str
) -> dict[str, str]:
    """Save the real artifact set; returns the headline facts."""
    import numpy as np
    from safetensors.numpy import save_file

    from fx1.modelcard import EvalDelta, ModelCard

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.safetensors"
    tensors = {
        name: param.detach().cpu().numpy().astype(np.float32)
        for name, param in model.state_dict().items()
    }
    n_params = int(sum(int(np.prod(t.shape)) for t in tensors.values()))
    save_file(tensors, str(weights_path), metadata={"format": "fx1-tiny-byte-transformer"})
    weights_sha = _sha256_file(weights_path)

    manifest = {
        "schema": "fx1-weights-manifest.v1",
        "weights_file": "weights.safetensors",
        "weights_sha256": weights_sha,
        "format": "safetensors/float32",
        "tokenizer": "byte-level-259",
        "arch": {
            "kind": "tiny-byte-transformer",
            "vocab": VOCAB_SIZE,
            "dim": _DIM,
            "ctx": _CTX,
            "n_layers": 1,
            "n_heads": _N_HEADS,
            "mlp_ratio": _MLP_RATIO,
            "n_params": n_params,
        },
        "training": {
            "corpus": _CORPUS_FILE,
            "corpus_sha256": corpus_sha,
            "steps": _STEPS,
            "seq_len": _CTX,
            "optimizer": f"adamw lr={_LR}",
            "seed": _SEED,
            "final_loss": final_loss,
        },
        "generator": "scripts/fx1_tiny_lm_train.py",
        "generator_sha256": _sha256_file(Path(__file__).resolve()),
        "created_unix": int(time.time()),
    }
    manifest_path = out_dir / "weights.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    ModelCard(
        version="fx-1.v0.1",
        corpus_sha256=corpus_sha,
        corpus_receipt_range=_CORPUS_FILE,
        training_manifest_sha256=_sha256_file(manifest_path),
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.51,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
        known_limits=[
            "fixture-scale byte-level LM (~31k params) trained for seconds on CPU over "
            "fx1_seed_corpus.jsonl — exists so the weights-direct path loads a real "
            "artifact; not an fx-1 release candidate",
            "eval_delta values are synthetic ship-gate placeholders, not measured evals",
        ],
    ).save(out_dir / "modelcard.json")
    return {"weights_sha256": weights_sha, "n_params": str(n_params)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "fx1_tiny_lm",
        help="checkpoint dir to write (default artifacts/fx1_tiny_lm)",
    )
    args = parser.parse_args(argv)
    try:
        import torch  # noqa: F401 — presence check for a clear message
    except ImportError:
        print("torch is required: `uv sync --all-extras` (the `nn` extra)", file=sys.stderr)
        return 2
    started = time.monotonic()
    model, final_loss, corpus_sha = _train(_REPO_ROOT)
    facts = _write_checkpoint(args.out_dir, model, final_loss, corpus_sha)
    print(
        f"trained {_STEPS} steps in {time.monotonic() - started:.1f}s; "
        f"final_loss={final_loss:.4f} n_params={facts['n_params']} "
        f"weights_sha256={facts['weights_sha256'][:16]}… -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
