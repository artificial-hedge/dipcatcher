"""Kernel-lineage seals and witnesses fail closed when the kernels change.

The fixtures are copies of the real kernel sources. Nothing here is a
market result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.audit.kernel_lineage import (
    KERNEL_PATHS,
    compare_kernel_lineage,
    encode_kernel_witness,
    seal_kernel_lineage,
    structural_digest,
    verify_kernel_lineage,
    verify_kernel_witness,
)

_REPO = Path(__file__).resolve().parents[3]


def _stage(tmp_path: Path) -> Path:
    for rel in KERNEL_PATHS:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((_REPO / rel).read_bytes())
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'staged'\n", encoding="utf-8")
    return tmp_path


def test_structural_digest_ignores_docstring_and_sees_a_constant() -> None:
    plain = "def f(x):\n    return x + 1\n"
    documented = 'def f(x):\n    """note"""\n    return x + 1\n'
    changed = "def f(x):\n    return x + 2\n"
    assert structural_digest(plain) == structural_digest(documented)
    assert structural_digest(plain) != structural_digest(changed)


def test_seal_verifies_and_witness_hides_source(tmp_path: Path) -> None:
    root = _stage(tmp_path)
    seal = seal_kernel_lineage(root)
    assert verify_kernel_lineage(seal, root)["ok"]
    witness = encode_kernel_witness(root)
    assert verify_kernel_witness(witness, root)["ok"]
    assert b"gammaln" not in witness
    assert b"Nelder-Mead" not in witness


def test_comment_changes_bytes_not_syntax(tmp_path: Path) -> None:
    root = _stage(tmp_path)
    before = seal_kernel_lineage(root)
    target = root / "src/quant_fund/models/realized.py"
    target.write_bytes(target.read_bytes() + b"\n# layout note\n")
    after = seal_kernel_lineage(root)
    report = compare_kernel_lineage(before, after)
    assert report["same_root"] is False
    assert report["byte_drift"] == ["src/quant_fund/models/realized.py"]
    assert report["structural_drift"] == []
    checked = verify_kernel_lineage(before, root)
    assert checked["ok"] is False
    assert "src/quant_fund/models/realized.py" in checked["drifted"]


def test_logic_edit_moves_the_structural_digest(tmp_path: Path) -> None:
    root = _stage(tmp_path)
    before = seal_kernel_lineage(root)
    target = root / "src/quant_fund/flowbars/realized.py"
    target.write_text(target.read_text(encoding="utf-8") + "\nSENTINEL = 1\n", encoding="utf-8")
    after = seal_kernel_lineage(root)
    report = compare_kernel_lineage(before, after)
    assert "src/quant_fund/flowbars/realized.py" in report["structural_drift"]
    assert verify_kernel_witness(encode_kernel_witness(root), root)["ok"]
    assert (
        verify_kernel_witness(encode_kernel_witness(root), _stage(tmp_path / "other"))["ok"]
        is False
    )


def test_tampered_trailer_and_missing_member_fail(tmp_path: Path) -> None:
    root = _stage(tmp_path)
    witness = bytearray(encode_kernel_witness(root))
    witness[-1] ^= 0x01
    failed = verify_kernel_witness(bytes(witness), root)
    assert failed["ok"] is False
    assert "trailer mismatch" in failed["errors"][0]
    (root / "src/quant_fund/metrics/scoring.py").unlink()
    with pytest.raises(ValueError, match="missing"):
        seal_kernel_lineage(root)


def test_symlink_member_is_refused(tmp_path: Path) -> None:
    root = _stage(tmp_path)
    target = root / "src/quant_fund/metrics/scoring.py"
    payload = target.read_bytes()
    target.unlink()
    outside = tmp_path / "outside.py"
    outside.write_bytes(payload)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        seal_kernel_lineage(root)


def test_utils_default_repo_root_delegates_to_the_audit_library(monkeypatch) -> None:
    sentinel = Path("/tmp/kernel-lineage-root")

    def _fake() -> Path:
        return sentinel

    monkeypatch.setattr("quant_fund.audit.kernel_lineage.default_repo_root", _fake)
    from quant_fund.utils.kernel_lineage import default_repo_root

    assert default_repo_root() == sentinel


def test_receipt_hash_catches_a_rewritten_interpretation(tmp_path: Path) -> None:
    root = _stage(tmp_path)
    seal = seal_kernel_lineage(root)
    seal["interpretation"] = "rewritten"
    checked = verify_kernel_lineage(seal, root)
    assert checked["ok"] is False
    assert "receipt_sha256" in checked["errors"]
