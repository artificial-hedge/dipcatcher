"""run_audit — adversarial probes on the training-manifest builder.

Pinned contract:

- ``_validate_corpus``: every line must carry ``receipt_sha256``
  provenance; an unprovenanced line or an empty corpus refuses.
- ``_validate_eval_gate``: the flag is re-derived — a bare
  ``honesty_gate_passed: true`` with no honesty results refuses, as do
  failed honesty tasks and ``honesty:`` violations on ANY task
  (a domain-task violation also blocks).
- ``build_training_manifest``: missing corpus/eval raise
  FileNotFoundError; the manifest binds both files by sha256 and
  hardcodes ``live_pnl_claim=False`` / ``research_only=True``.

Sealed ``run_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["run_audit", "run_audit_bench"]

_HONESTY_OK = {
    "results": [{"kind": "honesty", "passed": True, "failures": []}],
    "honesty_gate_passed": True,
}


def _raises(fn: Any) -> str:
    try:
        fn()
        return "no-raise"
    except Exception as e:
        return type(e).__name__


def _corpus(lines: list[dict[str, Any]], p: Path) -> Path:
    p.write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
    return p


def run_audit() -> dict[str, Any]:
    from fx1.train.run import _validate_corpus, _validate_eval_gate, build_training_manifest

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        good_corpus = _corpus(
            [
                {"receipt_sha256": "a" * 64, "messages": []},
                {"receipt_sha256": "b" * 64, "negative": True},
            ],
            d / "c.jsonl",
        )
        stats = _validate_corpus(good_corpus)
        out["corpus_stats"] = stats == {"lines": 2, "positive": 1, "negative": 1}
        out["no_provenance_refuses"] = (
            _raises(lambda: _validate_corpus(_corpus([{"messages": []}], d / "x.jsonl")))
            == "ValueError"
        )
        out["empty_refuses"] = (
            _raises(lambda: _validate_corpus(_corpus([], d / "e.jsonl"))) == "ValueError"
        )

        eval_ok = d / "ev.json"
        eval_ok.write_text(json.dumps(_HONESTY_OK), encoding="utf-8")
        _validate_eval_gate(eval_ok)
        out["gate_ok"] = True

        no_flag = d / "nf.json"
        no_flag.write_text(json.dumps({"results": _HONESTY_OK["results"]}))
        flag_only = d / "fo.json"
        flag_only.write_text(json.dumps({"results": [], "honesty_gate_passed": True}))
        failing = d / "fa.json"
        failing.write_text(
            json.dumps(
                {
                    "results": [{"kind": "honesty", "passed": False, "failures": []}],
                    "honesty_gate_passed": True,
                }
            )
        )
        domain_viol = d / "dv.json"
        domain_viol.write_text(
            json.dumps(
                {
                    "results": [
                        {"kind": "honesty", "passed": True, "failures": []},
                        {"kind": "domain", "passed": True, "failures": ["honesty:x"]},
                    ],
                    "honesty_gate_passed": True,
                }
            )
        )
        out["flag_alone_fails"] = _raises(lambda: _validate_eval_gate(flag_only)) == "ValueError"
        out["no_flag_fails"] = _raises(lambda: _validate_eval_gate(no_flag)) == "ValueError"
        out["failing_honesty_fails"] = _raises(lambda: _validate_eval_gate(failing)) == "ValueError"
        out["domain_violation_fails"] = (
            _raises(lambda: _validate_eval_gate(domain_viol)) == "ValueError"
        )

        cfg = _mkcfg(good_corpus, eval_ok)
        out["missing_corpus_fails"] = (
            _raises(
                lambda: build_training_manifest(_mkcfg(d / "gone.jsonl", eval_ok), d / "m.json")
            )
            == "FileNotFoundError"
        )
        out["missing_eval_fails"] = (
            _raises(
                lambda: build_training_manifest(_mkcfg(good_corpus, d / "gone.json"), d / "m.json")
            )
            == "FileNotFoundError"
        )

        manifest_path = d / "manifest.json"
        m = build_training_manifest(cfg, manifest_path)
        out["binds_corpus"] = (
            m["corpus_sha256"] == hashlib.sha256(good_corpus.read_bytes()).hexdigest()
        )
        out["binds_eval"] = (
            m["eval_results_sha256"] == hashlib.sha256(eval_ok.read_bytes()).hexdigest()
        )
        out["honesty_hardcoded"] = m["live_pnl_claim"] is False and m["research_only"] is True
        out["written_matches"] = json.loads(manifest_path.read_text())["run_name"] == "x"
    return out


def _mkcfg(corpus: Path, ev: Path) -> Any:
    from fx1.train.config import LadderStage, TrainConfig

    return TrainConfig(
        run_name="x",
        stage=LadderStage.PROXY,
        corpus_jsonl=str(corpus),
        eval_results_json=str(ev),
        estimated_nodes=1,
        estimated_gpu_hours=0.5,
        estimated_cost_usd=0.0,
    )


def run_audit_bench() -> dict[str, Any]:
    r = run_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "run_audit",
        "schema": "run_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Manifest-builder contract holds: corpus provenance + non-empty "
            "are enforced, the honesty gate is re-derived (flag alone, no "
            "honesty results, failing honesty, and domain-task violations "
            "all refuse), and the manifest binds corpus+eval by sha256 "
            "with hardcoded honesty fields."
            if ok
            else f"RUN AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
