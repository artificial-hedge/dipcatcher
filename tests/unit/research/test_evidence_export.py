"""Evidence-export: the producer half of ``verify-repo --evidence-only``.

The strongest test uses the live repo itself: export the real evidence
store and verify the bundle — if the export's member set diverges from the
pinned epoch state (a missing member, a stray extra file), the bundle
fails its own gates.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quant_fund.research.evidence_export import (
    BUNDLE_MANIFEST,
    EVIDENCE_BUNDLE_SCHEMA,
    export_evidence_bundle,
)
from quant_fund.research.repo_integrity import EVIDENCE_CORPORA, verify_repo

REPO_ROOT = Path(__file__).resolve().parents[3]


def _has_git(root: Path) -> bool:
    try:
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _minimal_evidence_tree(tmp_path: Path) -> Path:
    """A tree with just the evidence corpora + signature file."""
    from quant_fund.research.corpus_epoch import (
        corpus_epoch,
        update_heads_pin,
        write_epoch_receipt,
    )
    from quant_fund.research.repo_integrity import CORPORA

    root = tmp_path / "repo"
    pin = root / "quality" / "epoch_heads.json"
    # Stamp quality last: epoch_heads.json is itself a quality member, so the
    # quality epoch must see the final pin.
    for corpus_dir in sorted(EVIDENCE_CORPORA - {"quality"}) + ["quality"]:
        d = root / corpus_dir
        d.mkdir(parents=True, exist_ok=True)
        pattern = next(c[1] for c in CORPORA if c[0] == corpus_dir)
        ext = pattern.lstrip("*") or ".json"
        (d / f"evidence_0{ext}").write_text('{"n": 0}\n', encoding="utf-8")
        ep = write_epoch_receipt(corpus_epoch(d, pattern=pattern), d)
        update_heads_pin(pin, corpus_dir, pattern, ep)
    (root / "gate_pins.sig").write_text("sig-bytes\n", encoding="utf-8")
    return root


def test_export_reproduces_pinned_member_set(tmp_path: Path) -> None:
    root = _minimal_evidence_tree(tmp_path)
    manifest = export_evidence_bundle(root, tmp_path / "bundle")
    assert manifest["schema"] == EVIDENCE_BUNDLE_SCHEMA
    bundle = tmp_path / "bundle"
    for corpus_dir in EVIDENCE_CORPORA:
        assert (bundle / corpus_dir).is_dir(), corpus_dir
        # member count: the file + its epoch receipt only when the glob admits .json
        assert manifest["corpora"][corpus_dir]["members"] >= 1
    assert (bundle / "gate_pins.sig").is_file()
    assert (bundle / BUNDLE_MANIFEST).is_file()
    # The bundle's epoch chains re-derive the same heads — verify the corpus
    # state directly rather than through the full gate set (no crown jewels
    # or signature infrastructure exist in the synthetic tree).
    from quant_fund.research.corpus_epoch import member_digests
    from quant_fund.research.repo_integrity import CORPORA

    for corpus_dir in EVIDENCE_CORPORA:
        pattern = next(c[1] for c in CORPORA if c[0] == corpus_dir)
        assert member_digests(bundle / corpus_dir, pattern=pattern) == member_digests(
            root / corpus_dir, pattern=pattern
        )


def test_export_refuses_nonempty_target(tmp_path: Path) -> None:
    root = _minimal_evidence_tree(tmp_path)
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        export_evidence_bundle(root, out)


def test_export_fails_closed_on_missing_corpus(tmp_path: Path) -> None:
    root = _minimal_evidence_tree(tmp_path)
    import shutil

    shutil.rmtree(root / sorted(EVIDENCE_CORPORA)[0])
    with pytest.raises(FileNotFoundError, match="corpus missing"):
        export_evidence_bundle(root, tmp_path / "bundle")


def test_export_manifest_records_source_revision(tmp_path: Path) -> None:
    root = _minimal_evidence_tree(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=root,
        check=True,
    )
    manifest = export_evidence_bundle(root, tmp_path / "bundle")
    assert len(manifest["source_revision"]) == 40


@pytest.mark.skipif(
    not (REPO_ROOT / "quality" / "epoch_heads.json").is_file() or not _has_git(REPO_ROOT),
    reason="requires the real repo tree with committed integrity state",
)
def test_real_tree_export_verifies_evidence_only(tmp_path: Path) -> None:
    """Export the live repo's evidence store and run the real gate set on it:
    every epoch chain must re-derive its pinned head; crown_jewels reports
    the skipped marker; signature/witness gates verify offline."""
    bundle = tmp_path / "bundle"
    manifest = export_evidence_bundle(REPO_ROOT, bundle)
    assert manifest["source_revision"]

    result = verify_repo(bundle, evidence_only=True)
    fails = {
        name: gate["errors"]
        for name, gate in result["gates"].items()
        if not gate["ok"] and not gate.get("skipped")
    }
    assert fails == {}, fails
    assert result["gates"]["crown_jewels"].get("skipped") == "evidence_only"


@pytest.mark.skipif(
    not (REPO_ROOT / "quality" / "epoch_heads.json").is_file() or not _has_git(REPO_ROOT),
    reason="requires the real repo tree with committed integrity state",
)
def test_corrupted_bundle_member_fails_verification(tmp_path: Path) -> None:
    """A bundle with a tampered member must not verify — the export is only
    as trustworthy as the gates that re-check it."""
    bundle = tmp_path / "bundle"
    export_evidence_bundle(REPO_ROOT, bundle)
    victim = bundle / "receipts"
    target = next(p for p in victim.glob("*.json") if not p.name.startswith("corpus_epoch_"))
    target.write_bytes(b'{"forged": true}\n')
    result = verify_repo(bundle, evidence_only=True)
    epoch_gate = result["gates"]["epoch:receipts"]
    assert not epoch_gate["ok"]
    assert not result["ok"]
