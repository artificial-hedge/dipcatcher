"""Base for admission and epoch gates is the last green evidence run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from quant_fund.research.evidence_base import (
    CommitEvidence,
    collect_commit_evidence,
    main,
    resolve_evidence_base,
)


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name + "\n")
    subprocess.run(["git", "add", name], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", name], cwd=repo, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "ledger@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Ledger"], cwd=repo, check=True)
    green = _commit(repo, "green")
    cancelled = _commit(repo, "cancelled")
    head = _commit(repo, "head")
    return repo, green, cancelled, head


def test_cancelled_push_is_gated_back_to_last_green(tmp_path: Path) -> None:
    """Two synthetic commits after a green run: the cancelled tip and the new push."""
    _repo_path, green, cancelled, head = _repo(tmp_path)
    assert (_repo_path / "head").is_file()
    commits = [
        CommitEvidence(head, cancelled, None),
        CommitEvidence(cancelled, green, "cancelled"),
        CommitEvidence(green, None, "success"),
    ]
    result = resolve_evidence_base("push", head, cancelled, None, commits)
    assert result["ok"] is True
    assert result["sha"] == green
    assert result["reason"] == "last_green"
    assert result["unverified"] == [cancelled]


def test_completed_previous_run_stays_the_base(tmp_path: Path) -> None:
    _repo_path, green, cancelled, head = _repo(tmp_path)
    commits = [
        CommitEvidence(head, cancelled, None),
        CommitEvidence(cancelled, green, "success"),
        CommitEvidence(green, None, "success"),
    ]
    result = resolve_evidence_base("push", head, cancelled, None, commits)
    assert result == {
        "ok": True,
        "sha": cancelled,
        "reason": "last_green",
        "unverified": [],
    }


def test_no_green_fails_closed_over_the_cancelled_commit(tmp_path: Path) -> None:
    repo, green, cancelled, head = _repo(tmp_path)
    commits = [
        CommitEvidence(head, cancelled, None),
        CommitEvidence(cancelled, green, "cancelled"),
    ]
    result = resolve_evidence_base("push", head, cancelled, None, commits)
    assert result["ok"] is False
    assert result["sha"] == green
    assert cancelled in result["unverified"]
    assert result["reason"] == "no_green_evidence"
    assert repo.is_dir()


def test_pull_request_uses_pr_base() -> None:
    result = resolve_evidence_base("pull_request", "a" * 40, "b" * 40, "c" * 40, [])
    assert result == {"ok": True, "sha": "c" * 40, "reason": "pr_base", "unverified": []}


def test_branch_create_skips() -> None:
    result = resolve_evidence_base("push", "a" * 40, "0" * 40, None, [])
    assert result["ok"] is True
    assert result["sha"] is None
    assert result["reason"] == "branch_created"


def test_cli_reads_commits_json(tmp_path: Path, capsys: object) -> None:
    _repo_path, green, cancelled, head = _repo(tmp_path)
    assert (_repo_path / "cancelled").is_file()
    path = tmp_path / "commits.json"
    path.write_text(
        json.dumps(
            [
                {"sha": head, "parent": cancelled, "evidence": None},
                {"sha": cancelled, "parent": green, "evidence": "cancelled"},
                {"sha": green, "parent": None, "evidence": "success"},
            ]
        )
    )
    code = main(
        [
            "--event",
            "push",
            "--head",
            head,
            "--push-before",
            cancelled,
            "--commits-json",
            str(path),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    body = json.loads(captured.out)
    assert body["sha"] == green
    assert body["unverified"] == [cancelled]


def test_collector_walks_past_cancelled_to_artifact_green() -> None:
    head, cancelled, green, parent = ("h" * 40, "c" * 40, "g" * 40, "p" * 40)

    def fetch(path: str) -> object:
        if path.startswith("/repos/o/r/commits?"):
            return [
                {"sha": head, "parents": [{"sha": cancelled}]},
                {"sha": cancelled, "parents": [{"sha": green}]},
                {"sha": green, "parents": [{"sha": parent}]},
            ]
        if "artifacts" in path:
            return {"artifacts": [{"name": "evidence-green", "workflow_run": {"head_sha": green}}]}
        if "workflows/ci.yml/runs" in path:
            return {
                "workflow_runs": [
                    {
                        "id": 1,
                        "head_sha": cancelled,
                        "conclusion": "cancelled",
                        "created_at": "2026-10-18T00:00:00Z",
                    }
                ]
            }
        if path.endswith("/jobs?per_page=100"):
            return {"jobs": [{"name": "evidence-audit", "conclusion": "cancelled"}]}
        raise AssertionError(path)

    chain = collect_commit_evidence(head, repo="o/r", push_before=cancelled, fetch=fetch)
    result = resolve_evidence_base("push", head, cancelled, None, chain)
    assert result["ok"] is True
    assert result["sha"] == green
    assert result["unverified"] == [cancelled]
