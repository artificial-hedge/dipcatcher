"""Epoch closed world must report tracked evidence that no chain covers."""

from __future__ import annotations

from pathlib import Path

from quant_fund.research.repo_integrity import unchained_surface_errors, verify_repo

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_verify_repo_fails_on_unchained_manifest(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    orphan = root / "data" / "manifests" / "replay" / "orphan.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("{}\n")
    res = verify_repo(root)
    gate = res["gates"]["coverage_closure"]
    assert gate["ok"] is False
    assert "uncovered_member:'data/manifests/replay/orphan.json'" in gate["errors"]
    assert res["ok"] is False


def test_live_manifests_and_packaging_are_unchained() -> None:
    errors = unchained_surface_errors(REPO_ROOT)
    for rel in (
        "data/manifests/replay/corpus.json",
        "data/manifests/replay_corpus_members.json",
        "data/manifests/yahoo_eod.json",
        "data/manifests/yahoo_eod_us_wide.json",
        "packaging/homebrew/fxi.rb",
    ):
        assert f"uncovered_member:{rel!a}" in errors
