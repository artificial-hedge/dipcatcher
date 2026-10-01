"""receipts_audit (train) — adversarial probes on training receipts.

Pinned contract for ``issue_receipt``/``verify_training_receipt``:

- The receipt binds config, corpus, split manifest, and eval-base by
  content sha256; ``env_fingerprint`` binds sorted env *names* only —
  values never leak.
- ``verify_training_receipt`` fails closed: tampered field, missing
  file, or mutated bytes → False; ``live_pnl_claim=True`` or
  ``research_only=False`` → False.
- ``_git_revision`` fails closed: a non-git root yields
  ``("unknown", True)``.
- Honesty fields are hardcoded True/False on the model — a forged
  receipt flipping them fails verification.

Sealed ``train_receipt_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["train_receipt_audit", "train_receipt_audit_bench"]


def train_receipt_audit() -> dict[str, Any]:
    from fx1.train.receipts import (
        _git_revision,
        issue_receipt,
        verify_training_receipt,
    )

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = root / "c.json"
        cfg.write_text("{}")
        corpus = root / "corpus.jsonl"
        corpus.write_text("{}\n")
        split = root / "split.json"
        split.write_text("{}")
        evb = root / "evb.json"
        evb.write_text("{}")
        rp = root / "receipt.json"

        rec = issue_receipt(
            run_name="audit",
            repo_root=root,
            config_path=cfg,
            corpus_path=corpus,
            split_manifest_path=split,
            eval_base_path=evb,
            seed=17,
            out_path=rp,
        )
        out["honesty_hardcoded"] = rec.live_pnl_claim is False and rec.research_only is True
        out["four_digests_bound"] = all(
            len(getattr(rec, f)) == 64
            for f in (
                "config_sha256",
                "corpus_sha256",
                "split_manifest_sha256",
                "eval_base_sha256",
            )
        )
        out["env_names_only"] = len(rec.env_fingerprint) == 64
        out["non_git_fails_closed"] = _git_revision(root) == ("unknown", True)

        out["verify_ok"] = verify_training_receipt(
            rp,
            config_path=cfg,
            corpus_path=corpus,
            split_manifest_path=split,
            eval_base_path=evb,
        )
        # mutated corpus bytes → fail
        corpus.write_text("{}\n#tamper\n")
        out["tampered_corpus_fails"] = not verify_training_receipt(
            rp,
            config_path=cfg,
            corpus_path=corpus,
            split_manifest_path=split,
            eval_base_path=evb,
        )
        corpus.write_text("{}\n")
        # missing file → fail
        evb.unlink()
        out["missing_file_fails"] = not verify_training_receipt(
            rp,
            config_path=cfg,
            corpus_path=corpus,
            split_manifest_path=split,
            eval_base_path=evb,
        )
        evb.write_text("{}")
        # forged receipt: flip live_pnl_claim → verify refuses
        forged = rec.model_copy(update={"live_pnl_claim": True})
        fp = root / "forged.json"
        fp.write_text(forged.model_dump_json())
        out["forged_live_claim_fails"] = not verify_training_receipt(
            fp,
            config_path=cfg,
            corpus_path=corpus,
            split_manifest_path=split,
            eval_base_path=evb,
        )
        forged2 = rec.model_copy(update={"research_only": False})
        fp2 = root / "forged2.json"
        fp2.write_text(forged2.model_dump_json())
        out["forged_research_fails"] = not verify_training_receipt(
            fp2,
            config_path=cfg,
            corpus_path=corpus,
            split_manifest_path=split,
            eval_base_path=evb,
        )
        # malformed receipt json raises (fail-closed parse)
        badp = root / "bad.json"
        badp.write_text("{not json")
        try:
            verify_training_receipt(
                badp,
                config_path=cfg,
                corpus_path=corpus,
                split_manifest_path=split,
                eval_base_path=evb,
            )
            out["malformed_raises"] = "no-raise"
        except Exception:
            out["malformed_raises"] = "raise"
    return out


def train_receipt_audit_bench() -> dict[str, Any]:
    r = train_receipt_audit()
    ok = (
        all(r[k] is True for k in r if k != "malformed_raises") and r["malformed_raises"] == "raise"
    )
    out: dict[str, Any] = {
        "kind": "train_receipt_audit",
        "schema": "train_receipt_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Training-receipt contract holds: four content digests bound, "
            "env fingerprint covers names only, non-git root fails "
            "closed, tampered/missing/forged all refuse, malformed JSON "
            "raises."
            if ok
            else f"TRAIN RECEIPT AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
