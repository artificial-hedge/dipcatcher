"""Train repository bytes into a resumable language-model checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from quant_fund.repo_llm.config import RepoLMConfig
from quant_fund.repo_llm.corpus import (
    CORPUS_FILENAME,
    MANIFEST_FILENAME,
    RepoLLMError,
    ensure_corpus,
    open_corpus,
)
from quant_fund.repo_llm.model import RepoLM, fold_corpus_into_model, unique_parameters

CHECKPOINT_FILENAME = "checkpoint.pt"
SUMMARY_FILENAME = "summary.json"
INIT_NAME = "corpus_fold_v1"


def convert_repository(
    repo_root: Path,
    out_dir: Path,
    *,
    steps: int,
    seed: int = 42,
    batch_size: int = 8,
    lr: float = 3e-4,
    resume: bool = False,
    probe_batches: int = 32,
    log_every: int = 25,
    config: RepoLMConfig | None = None,
    files: list[str] | None = None,
) -> dict[str, object]:
    """Build the corpus, fold it into a transformer, and train next-byte loss.

    ``steps`` is the number of new optimizer updates. ``--resume`` continues
    from ``checkpoint.pt`` when the corpus hash still matches.
    """
    if steps < 1:
        raise RepoLLMError("steps must be positive")
    if batch_size < 1 or probe_batches < 1:
        raise RepoLLMError("batch sizes must be positive")
    if not np.isfinite(lr) or lr <= 0:
        raise RepoLLMError("lr must be finite and positive")

    repo_root = repo_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_corpus(repo_root, out_dir, files)
    tokens = open_corpus(out_dir / CORPUS_FILENAME, manifest.token_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    checkpoint_path = out_dir / CHECKPOINT_FILENAME
    if resume:
        if not checkpoint_path.is_file():
            raise RepoLLMError(f"no checkpoint to resume at {checkpoint_path}")
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if blob.get("corpus_sha256") != manifest.corpus_sha256:
            raise RepoLLMError(
                "corpus changed since the checkpoint was written; "
                "start a new conversion instead of resuming"
            )
        cfg = RepoLMConfig.from_dict(blob["config"])
        model = RepoLM(cfg).to(device)
        model.load_state_dict(blob["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
        optimizer.load_state_dict(blob["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        step0 = int(blob["step"])
        tokens_seen = int(blob["tokens_seen"])
        probe_starts = np.asarray(blob["probe_starts"], dtype=np.int64)
        probe_initial = float(blob["probe_loss_initial"])
        train_loss_first = float(blob["train_loss_first"])
        history = [float(x) for x in blob["loss_history"]]
        rng = np.random.default_rng()
        rng.bit_generator.state = blob["rng_state"]
        torch.set_rng_state(blob["torch_rng_state"])
    else:
        cfg = config or RepoLMConfig()
        if manifest.token_count < cfg.seq_len + 1:
            raise RepoLLMError(
                f"corpus has {manifest.token_count} bytes; need at least {cfg.seq_len + 1}"
            )
        model = RepoLM(cfg).to(device)
        fold_corpus_into_model(model, tokens, seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
        step0 = 0
        tokens_seen = 0
        probe_starts = _sample_starts(
            manifest.token_count, cfg.seq_len, probe_batches, np.random.default_rng(seed + 17)
        )
        probe_initial = _probe_loss(model, tokens, probe_starts, cfg.seq_len, batch_size, device)
        train_loss_first = probe_initial
        history = []
        rng = np.random.default_rng(seed)

    if manifest.token_count < cfg.seq_len + 1:
        raise RepoLLMError("corpus is shorter than one training window")

    model.train()
    last_loss = train_loss_first
    for update in range(1, steps + 1):
        starts = _sample_starts(manifest.token_count, cfg.seq_len, batch_size, rng)
        inputs, targets = _windows(tokens, starts, cfg.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RepoLLMError("training loss is not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        history.append(last_loss)
        tokens_seen += int(batch_size * cfg.seq_len)
        if log_every and (update == 1 or update == steps or update % log_every == 0):
            print(
                f"step {step0 + update} loss {last_loss:.4f} tokens_seen {tokens_seen}",
                flush=True,
            )

    probe_final = _probe_loss(model, tokens, probe_starts, cfg.seq_len, batch_size, device)
    step = step0 + steps
    if step0 == 0:
        train_loss_first = history[0]
    payload = {
        "version": 1,
        "init": INIT_NAME,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "seed": seed,
        "corpus_sha256": manifest.corpus_sha256,
        "token_count": manifest.token_count,
        "file_count": len(manifest.files),
        "rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "probe_starts": probe_starts.astype(np.int64).tolist(),
        "probe_loss_initial": probe_initial,
        "probe_loss_final": probe_final,
        "train_loss_first": train_loss_first,
        "train_loss_last": last_loss,
        "loss_history": history,
    }
    tmp = checkpoint_path.with_suffix(".pt.partial")
    torch.save(payload, tmp)
    os.replace(tmp, checkpoint_path)
    summary = _summary(manifest, cfg, payload, out_dir, device)
    _write_summary(out_dir / SUMMARY_FILENAME, summary)
    _write_readme(out_dir / "README.txt", summary)
    return summary


def _summary(
    manifest,
    cfg: RepoLMConfig,
    payload: dict[str, object],
    out_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    model = RepoLM(cfg)
    model.load_state_dict(payload["model"])  # type: ignore[arg-type]
    n_params = sum(param.numel() for _name, param in unique_parameters(model))
    return {
        "init": INIT_NAME,
        "checkpoint": str(out_dir / CHECKPOINT_FILENAME),
        "corpus": str(out_dir / CORPUS_FILENAME),
        "manifest": str(out_dir / MANIFEST_FILENAME),
        "file_count": len(manifest.files),
        "token_count": manifest.token_count,
        "corpus_sha256": manifest.corpus_sha256,
        "excluded_prefixes": list(manifest.excluded_prefixes),
        "git_head": manifest.git_head,
        "parameter_count": n_params,
        "step": payload["step"],
        "tokens_seen": payload["tokens_seen"],
        "probe_loss_initial": payload["probe_loss_initial"],
        "probe_loss_final": payload["probe_loss_final"],
        "train_loss_first": payload["train_loss_first"],
        "train_loss_last": payload["train_loss_last"],
        "config": cfg.to_dict(),
        "device": str(device),
        "execution_claim": "research_only",
        "sizes_the_book": False,
        "live_pnl_claim": False,
    }


def _sample_starts(token_count: int, seq_len: int, count: int, rng: np.random.Generator) -> np.ndarray:
    max_start = token_count - seq_len - 1
    if max_start < 0:
        raise RepoLLMError("corpus is shorter than one training window")
    return rng.integers(0, max_start + 1, size=count, dtype=np.int64)


def _windows(
    tokens,
    starts: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(starts.shape[0])
    inputs = np.empty((batch, seq_len), dtype=np.int64)
    targets = np.empty((batch, seq_len), dtype=np.int64)
    for row, start in enumerate(starts.tolist()):
        window = np.asarray(tokens[start : start + seq_len + 1], dtype=np.int64)
        inputs[row] = window[:-1]
        targets[row] = window[1:]
    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device),
    )


def _probe_loss(model, tokens, starts: np.ndarray, seq_len: int, batch_size: int, device) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for cursor in range(0, int(starts.shape[0]), batch_size):
            batch_starts = starts[cursor : cursor + batch_size]
            inputs, targets = _windows(tokens, batch_starts, seq_len, device)
            _logits, loss = model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise RepoLLMError("probe loss is not finite")
            n = int(batch_starts.shape[0])
            total += float(loss) * n
            seen += n
    if was_training:
        model.train()
    return total / seen


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_readme(path: Path, summary: dict[str, object]) -> None:
    text = "\n".join(
        [
            "Dipcatcher repository byte language model",
            "",
            "This checkpoint was initialized by folding every byte of the repository",
            "corpus into the transformer weights, then trained with next-byte",
            "cross-entropy. It is not a forecast and it does not size the book.",
            "",
            f"files: {summary['file_count']}",
            f"corpus_bytes: {summary['token_count']}",
            f"corpus_sha256: {summary['corpus_sha256']}",
            f"parameters: {summary['parameter_count']}",
            f"optimizer_steps: {summary['step']}",
            f"probe_loss_initial: {summary['probe_loss_initial']}",
            f"probe_loss_final: {summary['probe_loss_final']}",
            "",
            "Continue training from this directory:",
            "  python -m quant_fund.repo_llm --resume --steps 500",
            "  dipcatcher repo-llm --resume --steps 500",
            "",
            "Requires the optional torch extra: pip install 'dipcatcher[nn]'.",
            "Excluded from the corpus: artifacts/repo_llm/ (this output).",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a git repository into trainable byte-level LM weights."
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("artifacts/repo_llm"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--probe-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args(argv)
    summary = convert_repository(
        args.repo,
        args.out,
        steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        probe_batches=args.probe_batches,
        log_every=args.log_every,
        config=RepoLMConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            seq_len=args.seq_len,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
