"""Bundle-fuzz: adversarial mutations of an exported evidence bundle must all
fail closed under ``scripts/verify_evidence_bundle.py``.

The auditor is a third-party oracle — an attacker controlling the export
boundary can lie in ``BUNDLE.json``, drop files outside corpora, roll the
heads pin back, or inject members. Every mutation class is pinned to exit 1;
the untouched baseline pins exit 0, so a mutation that starts passing is a
finding, not a flake.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from quant_fund.research.evidence_export import BUNDLE_MANIFEST, export_evidence_bundle
from quant_fund.research.repo_integrity import verify_repo

from .test_evidence_export import _minimal_evidence_tree

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "verify_evidence_bundle.py"


def _audit(bundle: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(bundle)],
        capture_output=True,
        text=True,
    )


def _bundle(tmp_path: Path) -> Path:
    root = _minimal_evidence_tree(tmp_path)
    out = tmp_path / "bundle"
    export_evidence_bundle(root, out)
    # The minimal tree is unsigned: a signature file without a committed
    # pubkey is a *signed-state* error, not the neutral ``unsigned`` this
    # fixture exercises. Drop it so the baseline is the unsigned bundle.
    (out / "gate_pins.sig").unlink()
    return out


def _manifest(bundle: Path) -> dict:
    return json.loads((bundle / BUNDLE_MANIFEST).read_text())


def _write_manifest(bundle: Path, m: dict) -> None:
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def _member(bundle: Path, corpus_dir: str) -> Path:
    return next(
        p
        for p in sorted((bundle / corpus_dir).iterdir())
        if p.is_file() and not p.name.startswith("corpus_epoch_")
    )


def test_baseline_passes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    proc = _audit(bundle)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = verify_repo(bundle, evidence_only=True)
    assert result["ok"], {k: g["errors"] for k, g in result["gates"].items() if not g["ok"]}


def _mut_schema(b: Path) -> None:
    m = _manifest(b)
    m["schema"] = "evidence_bundle.v0"
    _write_manifest(b, m)


def _mut_corpora_add(b: Path) -> None:
    m = _manifest(b)
    m["corpora"]["exfil"] = {"glob": "*", "members": 1, "git_members": {}}
    (b / "exfil").mkdir()
    (b / "exfil" / "loot.json").write_text("{}")
    _write_manifest(b, m)


def _mut_glob_drift(b: Path) -> None:
    m = _manifest(b)
    m["corpora"]["receipts"]["glob"] = "*.yaml"
    _write_manifest(b, m)


def _mut_member_tamper(b: Path) -> None:
    _member(b, "receipts").write_text('{"forged": true}\n')


def _mut_member_delete(b: Path) -> None:
    _member(b, "verifier").unlink()


def _mut_member_inject_glob(b: Path) -> None:
    (b / "receipts" / "smuggled.json").write_text('{"smuggled": true}\n')


def _mut_member_inject_nonglob(b: Path) -> None:
    (b / "receipts" / "evil.yaml").write_text("evil: true\n")


def _mut_foreign_root(b: Path) -> None:
    (b / "payload.exe").write_bytes(b"MZ")


def _mut_foreign_nested_parent(b: Path) -> None:
    # ``.github`` is only the *parent* of the workflows corpus — a file
    # beside it is foreign, not a member of ``.github/workflows``.
    (b / ".github" / "extra.yml").write_text("on: push\n")


def _mut_heads_pin_delete(b: Path) -> None:
    (b / "quality" / "epoch_heads.json").unlink()


def _mut_heads_pin_rollback(b: Path) -> None:
    pin = b / "quality" / "epoch_heads.json"
    heads = json.loads(pin.read_text())
    key = sorted(heads.get("heads", {}))[0]
    heads["heads"][key] = "0" * 64
    pin.write_text(json.dumps(heads, indent=2, sort_keys=True) + "\n")


def _mut_corpus_dir_delete(b: Path) -> None:
    shutil.rmtree(b / "artifacts")


def _mut_epoch_receipt_delete(b: Path) -> None:
    next((b / "configs").glob("corpus_epoch_*.json")).unlink()


def _mut_epoch_receipt_forge(b: Path) -> None:
    # A fabricated chain record: prev points at genesis but no member
    # set can make the recorded head digest real.
    victim = next((b / "configs").glob("corpus_epoch_*.json"))
    forged = json.loads(victim.read_text())
    forged["members"].append({"name": "injected.json", "sha256": "0" * 64})
    (b / "configs" / "corpus_epoch_forgedforgedfor.json").write_text(
        json.dumps(forged, indent=2, sort_keys=True)
    )


def _mut_manifest_delete(b: Path) -> None:
    (b / BUNDLE_MANIFEST).unlink()


MUTATIONS = {
    "manifest_schema_swap": _mut_schema,
    "manifest_corpora_add": _mut_corpora_add,
    "manifest_glob_drift": _mut_glob_drift,
    "member_tamper": _mut_member_tamper,
    "member_delete": _mut_member_delete,
    "member_inject_glob_matched": _mut_member_inject_glob,
    "member_inject_nonglob": _mut_member_inject_nonglob,
    "foreign_member_root": _mut_foreign_root,
    "foreign_member_nested_parent": _mut_foreign_nested_parent,
    "heads_pin_delete": _mut_heads_pin_delete,
    "heads_pin_rollback": _mut_heads_pin_rollback,
    "corpus_dir_delete": _mut_corpus_dir_delete,
    "epoch_receipt_delete": _mut_epoch_receipt_delete,
    "epoch_receipt_forge": _mut_epoch_receipt_forge,
    "manifest_delete": _mut_manifest_delete,
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_bundle_mutation_fails_closed(tmp_path: Path, mutation: str) -> None:
    bundle = _bundle(tmp_path)
    MUTATIONS[mutation](bundle)
    proc = _audit(bundle)
    assert proc.returncode != 0, (
        f"{mutation} escaped: exit {proc.returncode}\n{proc.stdout}{proc.stderr}"
    )
    # The library's own evidence-only audit must agree — the standalone
    # script exists to *cross-check* verify_repo; a mutation caught by one
    # oracle but not the other is a divergence finding of its own.
    result = verify_repo(bundle, evidence_only=True)
    assert not result["ok"], (
        f"{mutation} escaped the library auditor: {[k for k, g in result['gates'].items() if not g['ok']]}"
    )
