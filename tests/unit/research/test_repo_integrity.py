from __future__ import annotations

import json
from pathlib import Path

from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.research.repo_integrity import (
    REPO_INTEGRITY_SCHEMA,
    verify_repo,
    write_repo_integrity_receipt,
)


def _git_repo(tmp_path: Path) -> Path:
    """Minimal tree satisfying every gate: 9 jewels + 4 stamped corpora."""
    import subprocess

    from quant_fund.research.corpus_epoch import (
        HEADS_PIN_BASENAME,
        corpus_epoch,
        update_heads_pin,
        write_epoch_receipt,
    )
    from quant_fund.research.crown_jewels import (
        DEFAULT_JEWELS,
        write_crown_jewels_pin,
    )

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for jewel in DEFAULT_JEWELS:
        p = root / jewel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {jewel}\n")
    pin = root / "quality" / HEADS_PIN_BASENAME
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps({"schema": "epoch_heads.v1", "heads": {}}))
    for corpus_dir, pattern in (
        ("receipts", "*.json"),
        ("verifier", "*.md"),
        ("quality", "*.json"),
        (".github/workflows", "*.yml"),
    ):
        d = root / corpus_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / f"seed.{pattern[2:]}").write_text("seed")
        ep = write_epoch_receipt(corpus_epoch(d, pattern=pattern), d)
        update_heads_pin(pin, corpus_dir, pattern, ep)
    write_crown_jewels_pin(root)
    # Re-stamp quality: the pin and jewels file now exist as members.
    ep = write_epoch_receipt(corpus_epoch(root / "quality"), root / "quality")
    update_heads_pin(pin, "quality", "*.json", ep)
    return root


def test_verify_repo_all_gates_green(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    res = verify_repo(root)
    assert res["ok"], res
    assert set(res["gates"]) == {
        "crown_jewels",
        "epoch:receipts",
        "epoch:verifier",
        "epoch:quality",
        "epoch:.github/workflows",
    }


def test_verify_repo_fails_on_any_gate(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    (root / "pyproject.toml").write_text("tampered")
    res = verify_repo(root)
    assert not res["ok"]
    assert res["gates"]["crown_jewels"]["errors"] == ["jewel_mutated:pyproject.toml"]
    # Epochs untouched -> other gates still report clean.
    assert res["gates"]["epoch:receipts"]["ok"]


def test_repo_integrity_receipt_seals_and_verifies(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    out = root / "quality" / "repo_integrity.json"
    write_repo_integrity_receipt(out, root)
    verdict = verify_receipt_file(out)
    assert verdict["valid"], verdict["errors"]
    body = json.loads(out.read_text())
    assert body["schema"] == REPO_INTEGRITY_SCHEMA
    assert body["ok"] is True
    assert body["pins"]["epoch_heads_sha256"]
    assert body["pins"]["crown_jewels_sha256"]


def test_verify_repo_missing_pin_fails_closed(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    (root / "quality" / "epoch_heads.json").unlink()
    res = verify_repo(root)
    assert not res["ok"]
    # Quality chain loses its member + the pin is gone: jewel gate fine,
    # epoch gates fail closed.
    assert res["gates"]["crown_jewels"]["ok"]
    assert not res["gates"]["epoch:receipts"]["ok"]
    assert not res["gates"]["epoch:quality"]["ok"]
