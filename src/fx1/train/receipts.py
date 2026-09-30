"""Immutable training receipts for fx-1 — the lab's philosophy, applied to ML.

Every training run emits a receipt binding: git revision, dirty-worktree
state, config hash, corpus + split-manifest hashes, eval hashes (base and
candidate), seed, and environment. ``verify_training_receipt`` re-checks the
hashes against the files on disk. fx-1 inherits the harness's evidence
culture: a run without a verifiable receipt never happened.

Two-phase sealing. The receipt is *issued* before training (``issue_receipt``)
because the config, corpus, split, and base-eval evidence must exist before a
single GPU-hour is spent. The *promotion decision* can only exist after the
candidate eval, so ``seal_promotion`` performs a single, one-way seal that
binds the candidate-eval / comparison / gate digests and the ``promoted``
verdict into the same file. A sealed receipt cannot be resealed with
different evidence — that is what makes ``promoted`` durable rather than a
claim. ``promoted=True`` without bound eval digests fails verification: a
promotion with no evidence is not a promotion.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(repo_root: Path) -> tuple[str, bool]:
    """(HEAD sha, dirty?) — unknown/True when git is unavailable (fail-closed)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


class TrainingReceipt(BaseModel):
    run_name: str
    created_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_revision: str
    dirty_worktree: bool
    config_sha256: str = Field(min_length=64, max_length=64)
    corpus_sha256: str = Field(min_length=64, max_length=64)
    split_manifest_sha256: str = Field(min_length=64, max_length=64)
    eval_base_sha256: str = Field(min_length=64, max_length=64)
    seed: int
    python: str
    platform: str
    env_fingerprint: str = Field(
        description="SHA-256 of sorted relevant environment variable names"
    )
    live_pnl_claim: bool = False
    research_only: bool = True
    # --- promotion seal (written once, after the candidate eval) -------------
    promoted: bool = Field(
        default=False,
        description="True only when the fail-closed promotion gate passed and "
        "the eval evidence below is bound into this receipt.",
    )
    promotion_reasons: list[str] = Field(
        default_factory=list,
        description="Gate failure reasons; empty exactly when promoted is true.",
    )
    sealed_utc: str | None = None
    eval_candidate_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    comparison_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    promotion_gate_sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @property
    def is_sealed(self) -> bool:
        """True once the promotion decision has been bound in."""
        return self.eval_candidate_sha256 is not None


def issue_receipt(
    *,
    run_name: str,
    repo_root: str | Path,
    config_path: str | Path,
    corpus_path: str | Path,
    split_manifest_path: str | Path,
    eval_base_path: str | Path,
    seed: int,
    out_path: str | Path,
) -> TrainingReceipt:
    root = Path(repo_root)
    sha, dirty = _git_revision(root)
    env_names = sorted(k for k in os.environ if k.startswith(("CUDA", "NCCL", "TORCH", "FX1")))
    receipt = TrainingReceipt(
        run_name=run_name,
        git_revision=sha,
        dirty_worktree=dirty,
        config_sha256=_sha256_file(Path(config_path)),
        corpus_sha256=_sha256_file(Path(corpus_path)),
        split_manifest_sha256=_sha256_file(Path(split_manifest_path)),
        eval_base_sha256=_sha256_file(Path(eval_base_path)),
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        env_fingerprint=hashlib.sha256("\n".join(env_names).encode()).hexdigest(),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return receipt


def seal_promotion(
    *,
    receipt_path: str | Path,
    eval_candidate_path: str | Path,
    comparison_path: str | Path,
    promotion_gate_path: str | Path,
    promoted: bool,
    reasons: list[str] | None = None,
) -> TrainingReceipt:
    """One-way seal binding the promotion verdict + eval digests into a receipt.

    Idempotent on identical evidence (a retried pipeline run re-seals to the
    same bytes); raises on any attempt to overwrite a sealed decision with
    different evidence or a different verdict. Fail-closed by construction:
    ``promoted=True`` requires the gate file to agree.
    """
    path = Path(receipt_path)
    receipt = loads_receipt(path)
    candidate_sha = _sha256_file(Path(eval_candidate_path))
    comparison_sha = _sha256_file(Path(comparison_path))
    gate_sha = _sha256_file(Path(promotion_gate_path))
    gate = json.loads(Path(promotion_gate_path).read_text(encoding="utf-8"))
    if bool(gate.get("promoted")) != promoted:
        raise ValueError(
            "promotion seal: gate verdict disagrees with the receipt verdict — "
            "refusing to record a promotion the gate did not grant"
        )
    if promoted and bool(gate.get("reasons")):
        raise ValueError("promotion seal: promoted=True with unresolved gate reasons")
    if receipt.is_sealed:
        same = (
            receipt.eval_candidate_sha256 == candidate_sha
            and receipt.comparison_sha256 == comparison_sha
            and receipt.promotion_gate_sha256 == gate_sha
            and receipt.promoted == promoted
            and list(receipt.promotion_reasons) == list(reasons or [])
        )
        if not same:
            raise ValueError(
                "promotion seal: receipt is already sealed — the promotion "
                "decision is immutable and cannot be rewritten"
            )
        return receipt
    sealed = receipt.model_copy(
        update={
            "promoted": promoted,
            "promotion_reasons": list(reasons or []),
            "sealed_utc": datetime.now(UTC).isoformat(),
            "eval_candidate_sha256": candidate_sha,
            "comparison_sha256": comparison_sha,
            "promotion_gate_sha256": gate_sha,
        }
    )
    path.write_text(sealed.model_dump_json(indent=2), encoding="utf-8")
    return sealed


def verify_training_receipt(
    receipt_path: str | Path,
    *,
    config_path: str | Path,
    corpus_path: str | Path,
    split_manifest_path: str | Path,
    eval_base_path: str | Path,
    eval_candidate_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    promotion_gate_path: str | Path | None = None,
) -> bool:
    """Re-verify a receipt against the files on disk. Fail-closed.

    The four pre-training hashes are always checked. Bound promotion evidence
    (``eval_candidate``/``comparison``/``promotion_gate``) is checked when the
    receipt carries those digests — and a receipt claiming ``promoted=True``
    must carry all three plus an agreeing gate file, or verification fails.
    """
    receipt = TrainingReceipt.model_validate_json(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.live_pnl_claim or not receipt.research_only:
        return False
    checks = {
        "config_sha256": config_path,
        "corpus_sha256": corpus_path,
        "split_manifest_sha256": split_manifest_path,
        "eval_base_sha256": eval_base_path,
    }
    sealed_checks: dict[str, str | Path | None] = {
        "eval_candidate_sha256": eval_candidate_path,
        "comparison_sha256": comparison_path,
        "promotion_gate_sha256": promotion_gate_path,
    }
    for field, path in {**checks, **sealed_checks}.items():
        expected: str | None = getattr(receipt, field)
        if expected is None:
            continue
        if path is None:
            return False  # receipt binds evidence the caller cannot present
        p = Path(path)
        if not p.exists():
            return False
        if _sha256_file(p) != expected:
            return False
    if receipt.promoted:
        # A promotion is a claim about evidence; no bound evidence, no claim.
        if not all(getattr(receipt, f) for f in sealed_checks):
            return False
        if receipt.promotion_reasons:
            return False
        if promotion_gate_path is not None and Path(promotion_gate_path).exists():
            gate = json.loads(Path(promotion_gate_path).read_text(encoding="utf-8"))
            if not bool(gate.get("promoted")) or gate.get("reasons"):
                return False
    return True


def loads_receipt(path: str | Path) -> TrainingReceipt:
    return TrainingReceipt.model_validate_json(Path(path).read_text(encoding="utf-8"))
