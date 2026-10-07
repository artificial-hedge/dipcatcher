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

The trainer core lives in ``fx1.train.tiny_lm`` — the same module the
``/v1/fine_tuning/jobs`` runner uses, so a fine-tune job trains exactly
the way this fixture was trained.

Determinism: fixed seed, fixed corpus file, fixed step schedule. CPU BLAS
reduction order is not byte-stable across machines, so regeneration may
produce a bit-different artifact with equivalent behavior; the committed
file is the pinned one.

Usage::

    uv run --extra nn python scripts/fx1_tiny_lm_train.py [--out-dir artifacts/fx1_tiny_lm]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fx1.train.tiny_lm import (  # noqa: E402
    DEFAULT_CTX,
    chat_jsonl_to_text,
    save_weights,
    sha256_file,
    train_tiny_lm,
    write_modelcard,
    write_weights_manifest,
)

_STEPS = 3000
_LR = 3e-3
_SEED = 0
_CORPUS_FILE = "fx1_seed_corpus.jsonl"


def _corpus_text(repo_root: Path) -> str:
    """The training corpus: every message string in the committed seed corpus."""
    path = repo_root / _CORPUS_FILE
    return chat_jsonl_to_text(path)


def _train(repo_root: Path):
    """Seconds-scale CPU training; returns (model, final_loss, corpus_sha256)."""
    import hashlib

    text = _corpus_text(repo_root)
    corpus_sha = hashlib.sha256(text.encode()).hexdigest()
    model, final_loss, _trained_tokens, _val_loss = train_tiny_lm(
        text, steps=_STEPS, lr=_LR, seed=_SEED
    )
    return model, final_loss, corpus_sha


def _write_checkpoint(out_dir: Path, model, final_loss: float, corpus_sha: str) -> dict[str, str]:
    """Save the real artifact set; returns the headline facts."""
    root = _REPO_ROOT.resolve()
    resolved = out_dir.resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(
            f"refusing to write a checkpoint outside the repo: {resolved} (repo root is {root})"
        )
    weights_sha, n_params = save_weights(resolved, model)

    manifest_path = write_weights_manifest(
        resolved,
        weights_sha256=weights_sha,
        n_params=n_params,
        training={
            "corpus": _CORPUS_FILE,
            "corpus_sha256": corpus_sha,
            "steps": _STEPS,
            "seq_len": DEFAULT_CTX,
            "optimizer": f"adamw lr={_LR}",
            "seed": _SEED,
            "final_loss": final_loss,
            "trained_tokens": _STEPS * DEFAULT_CTX,
        },
        generator="scripts/fx1_tiny_lm_train.py",
        generator_path=Path(__file__),
    )

    write_modelcard(
        resolved,
        corpus_sha256=corpus_sha,
        corpus_receipt_range=_CORPUS_FILE,
        training_manifest_sha256=sha256_file(manifest_path),
        known_limits=[
            "fixture-scale byte-level LM (~31k params) trained for seconds on CPU over "
            "fx1_seed_corpus.jsonl — exists so the weights-direct path loads a real "
            "artifact; not an fx-1 release candidate",
            "eval_delta values are synthetic ship-gate placeholders, not measured evals",
        ],
    )
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
