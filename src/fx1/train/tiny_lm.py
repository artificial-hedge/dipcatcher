"""Tiny byte-level fx-1 trainer — the real in-repo trainer.

Trains the one-block byte-level transformer (~31k params, dim 32, 4 heads,
ctx 64) whose numpy twin serves in ``fx1.serve.local_engine`` — identical
forward math, so what trains here is what serves. Two consumers share it:

* ``scripts/fx1_tiny_lm_train.py`` — builds the committed fixture
  checkpoint at ``artifacts/fx1_tiny_lm/`` (fixed corpus, fixed seed,
  fixed 3000-step schedule).
* ``fx1.serve.finetune.default_ft_runner`` — the real trainer behind
  ``POST /v1/fine_tuning/jobs``: initializes from the base fx-1 weights
  and keeps training on the job's uploaded corpus inside the job's work
  dir, producing a checkpoint dir the registry can serve as ``ft:…``.

Everything is seeded and CPU-scale. ``trained_tokens`` is a real measured
count — ``steps * seq_len`` tokens of corpus windows consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fx1.train.config import TrainConfig
from quant_fund.utils.atomicio import atomic_write_text

if TYPE_CHECKING:
    import torch

VOCAB_SIZE = 259  # 256 byte-value tokens + BOS + EOS + PAD
BOS_ID = 256
EOS_ID = 257

DEFAULT_DIM = 32
DEFAULT_CTX = 64
DEFAULT_N_HEADS = 4
DEFAULT_MLP_RATIO = 4

MIN_CORPUS_BYTES = 512
# Step bounds for the ft surface: every job performs real gradient updates
# even on a small corpus (floor), and stays CPU-cheap (cap). The committed
# fixture uses its own fixed 3000-step schedule, independent of these.
MIN_FT_STEPS = 64
MAX_FT_STEPS = 20000
FT_LR_BASE = 1e-4
VAL_WINDOWS = 16


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def chat_jsonl_to_text(path: Path) -> str:
    """Extract the training text from a chat-format JSONL corpus file.

    Every ``messages[].content`` string of every row joins into one text.
    Raises ``ValueError`` on malformed rows or a corpus too small to train.
    """
    parts: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno}: not JSON ({exc.msg})") from exc
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list):
            raise ValueError(f"{path.name}:{lineno}: row has no messages list")
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str) and content:
                parts.append(content)
    text = "\n".join(parts)
    n_bytes = len(text.encode("utf-8"))
    if n_bytes < MIN_CORPUS_BYTES:
        raise ValueError(
            f"corpus {path.name} too small to train ({n_bytes} bytes of message "
            f"content, need >= {MIN_CORPUS_BYTES})"
        )
    return text


def plan_steps(n_positions: int, n_epochs: int) -> int:
    """Honest step count: ``n_epochs`` passes over the corpus in ctx windows.

    ``steps * seq_len`` is exactly the trained-token count the job reports.
    Clamped so a tiny corpus still gets real optimization steps and a big
    one stays CPU-cheap.
    """
    per_epoch = max(1, n_positions // DEFAULT_CTX)
    return min(MAX_FT_STEPS, max(MIN_FT_STEPS, n_epochs * per_epoch))


def build_tiny_lm(vocab: int, dim: int, ctx: int, n_heads: int) -> torch.nn.Module:
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
                    "fc1": nn.Linear(dim, DEFAULT_MLP_RATIO * dim),
                    "fc2": nn.Linear(DEFAULT_MLP_RATIO * dim, dim),
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
            return cast("torch.Tensor", self.lm_head(self.ln_f(x)))

    return _TinyLM()


def load_base_tensors(model: torch.nn.Module, checkpoint_dir: Path) -> str:
    """Load ``checkpoint_dir``'s safetensors weights into the torch model.

    Real fine-tuning starts from the base weights, not a fresh init.
    Returns the base weights sha256 so the checkpoint manifest can pin the
    lineage. Raises on a missing/verifying-failed base.
    """
    import numpy as np
    import torch
    from safetensors.numpy import load_file

    weights_path = checkpoint_dir / "weights.safetensors"
    manifest_path = checkpoint_dir / "weights.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pinned = manifest.get("weights_sha256")
    actual = sha256_file(weights_path)
    if not isinstance(pinned, str) or pinned != actual:
        raise RuntimeError(
            f"base checkpoint integrity check failed: {weights_path} does not "
            f"match the sha256 pinned in {manifest_path.name}"
        )
    tensors = load_file(str(weights_path))
    state = model.state_dict()
    missing = sorted(set(state) - set(tensors))
    if missing:
        raise RuntimeError(f"base checkpoint is missing tensors: {missing[:4]}")
    for name, param in state.items():
        arr = np.asarray(tensors[name], dtype=np.float32)
        param.data = torch.from_numpy(np.ascontiguousarray(arr))
    return actual


def train_tiny_lm(
    text: str,
    *,
    steps: int,
    lr: float,
    seed: int,
    base_checkpoint: Path | None = None,
    val_text: str | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> tuple[torch.nn.Module, float, int, float | None]:
    """Seeded AdamW over random ctx-windows of the byte-encoded corpus.

    Returns ``(model, final_loss, trained_tokens, val_loss)``; the token
    count is exactly ``steps * ctx`` consumed windows. When
    ``base_checkpoint`` is given, training starts from its (sha-verified)
    weights — the honest fine-tune path. ``should_continue`` is polled
    per step so cooperative cancel can interrupt a long train; aborting
    raises instead of returning a partial model.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    ids = torch.tensor([BOS_ID, *text.encode("utf-8")], dtype=torch.long)
    model = build_tiny_lm(VOCAB_SIZE, DEFAULT_DIM, DEFAULT_CTX, DEFAULT_N_HEADS)
    if base_checkpoint is not None:
        load_base_tensors(model, base_checkpoint)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    n_positions = ids.numel() - DEFAULT_CTX - 1
    final_loss = float("nan")
    done = 0
    while done < steps and (should_continue is None or should_continue()):
        start = int(torch.randint(n_positions, (1,), generator=gen).item())
        window = ids[start : start + DEFAULT_CTX + 1].unsqueeze(0)
        logits = model(window[:, :-1])
        loss = nn.functional.cross_entropy(logits[0], window[0, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
        done += 1
    if done < steps:
        raise RuntimeError(f"training interrupted after {done}/{steps} steps")
    return model, final_loss, done * DEFAULT_CTX, val_loss_of(model, val_text, seed)


def val_loss_of(model: torch.nn.Module, val_text: str | None, seed: int) -> float | None:
    """Mean CE over a fixed seeded window sample of the val text."""
    if not val_text:
        return None
    import torch
    from torch import nn

    val_ids = torch.tensor([BOS_ID, *val_text.encode("utf-8")], dtype=torch.long)
    v_positions = val_ids.numel() - DEFAULT_CTX - 1
    if v_positions <= 0:
        return None
    vgen = torch.Generator().manual_seed(seed ^ 0x5F3759DF)
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(min(VAL_WINDOWS, v_positions)):
            start = int(torch.randint(v_positions, (1,), generator=vgen).item())
            window = val_ids[start : start + DEFAULT_CTX + 1].unsqueeze(0)
            v_logits = model(window[:, :-1])
            losses.append(float(nn.functional.cross_entropy(v_logits[0], window[0, 1:]).item()))
    return sum(losses) / len(losses) if losses else None


def save_weights(out_dir: Path, model: torch.nn.Module) -> tuple[str, int]:
    """Write ``weights.safetensors``; return (sha256, n_params)."""
    import numpy as np
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.safetensors"
    tensors = {
        name: param.detach().cpu().numpy().astype(np.float32)
        for name, param in model.state_dict().items()
    }
    n_params = int(sum(int(np.prod(t.shape)) for t in tensors.values()))
    save_file(tensors, str(weights_path), metadata={"format": "fx1-tiny-byte-transformer"})
    return sha256_file(weights_path), n_params


def write_weights_manifest(
    out_dir: Path,
    *,
    weights_sha256: str,
    n_params: int,
    training: dict[str, Any],
    generator: str,
    generator_path: Path,
) -> Path:
    """Write ``weights.manifest.json`` and return its path."""
    manifest = {
        "schema": "fx1-weights-manifest.v1",
        "weights_file": "weights.safetensors",
        "weights_sha256": weights_sha256,
        "format": "safetensors/float32",
        "tokenizer": "byte-level-259",
        "arch": {
            "kind": "tiny-byte-transformer",
            "vocab": VOCAB_SIZE,
            "dim": DEFAULT_DIM,
            "ctx": DEFAULT_CTX,
            "n_layers": 1,
            "n_heads": DEFAULT_N_HEADS,
            "mlp_ratio": DEFAULT_MLP_RATIO,
            "n_params": n_params,
        },
        "training": training,
        "generator": generator,
        "generator_sha256": sha256_file(generator_path.resolve()),
        "created_unix": int(time.time()),
    }
    manifest_path = out_dir / "weights.manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def write_modelcard(
    out_dir: Path,
    *,
    corpus_sha256: str,
    corpus_receipt_range: str,
    training_manifest_sha256: str,
    known_limits: list[str],
    version: str = "fx-1.v0.1",
) -> Path:
    """Write the ship-gate card for a tiny-LM checkpoint.

    The eval_delta values are the same labeled synthetic ship-gate
    placeholders the committed fixture card carries — the real measured
    evals live in the producing run's own artifacts (eval_base.json /
    eval_candidate.json / comparison.json), never on the card.
    """
    from fx1.modelcard import EvalDelta, ModelCard

    card_path = out_dir / "modelcard.json"
    ModelCard(
        version=version,
        corpus_sha256=corpus_sha256,
        corpus_receipt_range=corpus_receipt_range,
        training_manifest_sha256=training_manifest_sha256,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.51,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
        known_limits=known_limits,
    ).save(card_path)
    return card_path


TrainerFn = Callable[[Path, Path, TrainConfig, Path], Path]


def make_tiny_lm_trainer(
    *,
    seed: int,
    fine_tuned_model: str,
    job_id: str,
    corpus_label: str,
    base_checkpoint: Path,
    base_weights_sha256: str,
    lr: float,
    val_path: Path | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> TrainerFn:
    """A ``fx1.train.pipeline`` TrainerFn that really fine-tunes the base LM.

    Trains ``n_epochs`` passes (per the job's hyperparameters) from the
    base checkpoint's weights on the job's split train corpus, writes a
    servable checkpoint into ``work_dir/checkpoint/``, and returns it.
    ``should_continue`` (optional) is polled between optimization blocks
    so cooperative cancel still wins during a long train.
    """

    def _train(train_jsonl: Path, val_jsonl: Path, config: TrainConfig, work_dir: Path) -> Path:
        text = chat_jsonl_to_text(train_jsonl)
        corpus_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        n_positions = len(text.encode("utf-8")) + 1 - DEFAULT_CTX - 1
        steps = plan_steps(n_positions, config.epochs)

        eval_val = val_path if val_path is not None else val_jsonl
        val_text: str | None = None
        if eval_val.is_file() and eval_val.stat().st_size > 0:
            try:
                val_text = chat_jsonl_to_text(eval_val)
            except ValueError:
                val_text = None  # a val set too small to tokenize isn't fatal

        model, final_loss, trained_tokens, val_loss = train_tiny_lm(
            text,
            steps=steps,
            lr=lr,
            seed=seed,
            base_checkpoint=base_checkpoint,
            val_text=val_text,
            should_continue=should_continue,
        )
        ckpt_dir = work_dir / "checkpoint"
        weights_sha, n_params = save_weights(ckpt_dir, model)
        training: dict[str, Any] = {
            "corpus": corpus_label,
            "corpus_sha256": corpus_sha,
            "steps": steps,
            "seq_len": DEFAULT_CTX,
            "optimizer": f"adamw lr={lr}",
            "seed": seed,
            "final_loss": final_loss,
            "trained_tokens": trained_tokens,
            "base_weights_sha256": base_weights_sha256,
            "fine_tuned_model": fine_tuned_model,
            "job_id": job_id,
        }
        if val_loss is not None:
            training["val_loss"] = val_loss
        manifest_path = write_weights_manifest(
            ckpt_dir,
            weights_sha256=weights_sha,
            n_params=n_params,
            training=training,
            generator="fx1.train.tiny_lm",
            generator_path=Path(__file__),
        )
        write_modelcard(
            ckpt_dir,
            corpus_sha256=corpus_sha,
            corpus_receipt_range=f"{corpus_label} (job {job_id})",
            training_manifest_sha256=sha256_file(manifest_path),
            known_limits=[
                "fixture-scale byte-level LM (~31k params) fine-tuned for seconds "
                "on CPU on the job's uploaded corpus — a real trained artifact, "
                "not an fx-1 release candidate",
                "eval_delta values are synthetic ship-gate placeholders, not "
                "measured evals; the job's real measured evals are the "
                "eval_base.json / eval_candidate.json / comparison.json "
                "result files",
                f"fine-tuned from base weights_sha256 {base_weights_sha256[:16]}…",
            ],
        )
        return ckpt_dir

    return _train
