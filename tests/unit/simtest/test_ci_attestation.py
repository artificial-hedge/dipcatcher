"""pr_test_suite reads an external cursor-ci attestation and does not write one."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from quant_fund.simtest.prod_readiness import (
    _ci_attestation_row,
    _judge_status_payload,
    _pr_test_suite,
    _research_api_guard,
    _slug_from_remote,
)

_SOURCE = Path("src/quant_fund/simtest/prod_readiness.py")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "attestation@example.com")
    _git(repo, "config", "user.name", "Attestation Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "README").write_text("checkout\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "init")
    return repo, _git(repo, "rev-parse", "HEAD")


def _signed(sha: str, result: str = "pass") -> tuple[bytes, bytes, bytes]:
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    body = json.dumps(
        {"context": "cursor-ci", "result": result, "schema": "cursor_ci_result.v1", "sha": sha},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return public, body, key.sign(body)


def _write_attestation(directory: Path, sha: str, result: str = "pass") -> tuple[Path, str]:
    public, body, signature = _signed(sha, result)
    path = directory / "cursor-ci-result.json"
    path.write_bytes(body)
    path.with_name(path.name + ".sig").write_text(signature.hex() + "\n", encoding="utf-8")
    return path, public.hex()


def test_absent_attestation_stays_needs_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sha = _repo(tmp_path)
    monkeypatch.delenv("CURSOR_CI_RESULT_FILE", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    row = _pr_test_suite(repo, tmp_path / "work")
    assert row["status"] == "needs-owner"
    assert row["evidence"]["sha"] == sha
    assert row["evidence"]["ok"] is False
    assert "no cursor-ci attestation" in row["summary"]


def test_commit_status_pass_and_fail_are_observations() -> None:
    passed = _judge_status_payload(
        {
            "statuses": [
                {
                    "context": "cursor-ci",
                    "state": "success",
                    "description": "CURSOR_CI_RESULT: pass",
                    "updated_at": "2026-11-02T00:00:00Z",
                }
            ]
        }
    )
    failed = _judge_status_payload(
        {
            "statuses": [
                {
                    "context": "other",
                    "state": "success",
                    "description": "CURSOR_CI_RESULT: pass",
                },
                {
                    "context": "cursor-ci",
                    "state": "failure",
                    "description": "CURSOR_CI_RESULT: fail",
                    "updated_at": "2026-11-02T00:00:01Z",
                },
            ]
        }
    )
    pending = _judge_status_payload(
        {"statuses": [{"context": "cursor-ci", "state": "pending", "description": ""}]}
    )
    assert passed == "pass"
    assert failed == "fail"
    assert pending is None
    assert _ci_attestation_row("pass", None, "a" * 40)["status"] == "pass"
    assert _ci_attestation_row("fail", None, "a" * 40)["status"] == "fail"
    assert _ci_attestation_row("pass", "fail", "a" * 40)["status"] == "fail"
    assert _ci_attestation_row(None, "bad-signature", "a" * 40)["status"] == "fail"


def test_signed_file_for_this_sha_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, sha = _repo(tmp_path)
    outside = tmp_path / "attest"
    outside.mkdir()
    path, public = _write_attestation(outside, sha)
    monkeypatch.setenv("CURSOR_CI_RESULT_FILE", str(path))
    monkeypatch.setenv("CURSOR_CI_ATTESTATION_PUBKEY", public)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    row = _pr_test_suite(repo, tmp_path / "work")
    assert row["status"] == "pass"
    assert row["evidence"]["sha"] == sha


def test_signed_file_for_another_sha_stays_needs_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _sha = _repo(tmp_path)
    outside = tmp_path / "attest"
    outside.mkdir()
    path, public = _write_attestation(outside, "b" * 40)
    monkeypatch.setenv("CURSOR_CI_RESULT_FILE", str(path))
    monkeypatch.setenv("CURSOR_CI_ATTESTATION_PUBKEY", public)
    row = _pr_test_suite(repo, tmp_path / "work")
    assert row["status"] == "needs-owner"


def test_bad_signature_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, sha = _repo(tmp_path)
    outside = tmp_path / "attest"
    outside.mkdir()
    path, public = _write_attestation(outside, sha, result="fail")
    path.with_name(path.name + ".sig").write_text("00" * 64 + "\n", encoding="utf-8")
    monkeypatch.setenv("CURSOR_CI_RESULT_FILE", str(path))
    monkeypatch.setenv("CURSOR_CI_ATTESTATION_PUBKEY", public)
    row = _pr_test_suite(repo, tmp_path / "work")
    assert row["status"] == "fail"


def test_tracked_work_and_symlink_paths_do_not_attest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sha = _repo(tmp_path)
    tracked, public = _write_attestation(repo, sha)
    _git(repo, "add", tracked.name, tracked.name + ".sig")
    monkeypatch.setenv("CURSOR_CI_RESULT_FILE", str(tracked))
    monkeypatch.setenv("CURSOR_CI_ATTESTATION_PUBKEY", public)
    assert _pr_test_suite(repo, tmp_path / "work")["status"] == "needs-owner"

    outside = tmp_path / "attest"
    outside.mkdir()
    external, public = _write_attestation(outside, sha)
    link = tmp_path / "linked.json"
    link.symlink_to(external)
    monkeypatch.setenv("CURSOR_CI_RESULT_FILE", str(link))
    monkeypatch.setenv("CURSOR_CI_ATTESTATION_PUBKEY", public)
    assert _pr_test_suite(repo, tmp_path / "work")["status"] == "needs-owner"

    work = tmp_path / "work"
    work.mkdir()
    dropped, public = _write_attestation(work, sha)
    monkeypatch.setenv("CURSOR_CI_RESULT_FILE", str(dropped))
    monkeypatch.setenv("CURSOR_CI_ATTESTATION_PUBKEY", public)
    assert _pr_test_suite(repo, work)["status"] == "needs-owner"


def test_slug_parser_drops_userinfo() -> None:
    slug = _slug_from_remote("https://user:s3cret@github.com/acme/widgets.git")
    assert slug == "acme/widgets"
    assert _slug_from_remote("git@github.com:acme/widgets.git") == "acme/widgets"
    assert "s3cret" not in (slug or "")


def test_research_api_guard_follows_the_extracted_compare() -> None:
    row = _research_api_guard(Path("."))
    assert row["status"] == "pass"
    text = (Path("src/quant_fund/api/http_controls.py")).read_text(encoding="utf-8")
    assert "hmac.compare_digest" in text
    assert "def keys_match" in text


def test_scorecard_source_does_not_sign_or_post() -> None:
    text = _SOURCE.read_text(encoding="utf-8")
    assert "Ed25519PrivateKey" not in text
    assert "generate_keypair" not in text
    assert "method POST" not in text
    assert "--method" not in text
    assert "def _pr_test_suite" in text
    assert "def _ed25519_verify" in text
