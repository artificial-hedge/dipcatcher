"""Resolve the commit a push must be evidence-gated against.

CI used to diff only ``github.event.before``..HEAD. A run that was
cancelled while pending never became a base, so the commits it carried
were never admission-, ack-, or chain-gated. The base is the nearest
first-parent ancestor of HEAD with a completed green evidence run.
Cancelled, failed, and unrecorded ancestors stay inside the range.

Pull requests gate the whole PR against ``github.base_ref``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ZERO_SHA = "0" * 40

_EVIDENCE = frozenset({"success", "failure", "cancelled"})


@dataclass(frozen=True)
class CommitEvidence:
    """One first-parent commit and the evidence-audit conclusion recorded for it."""

    sha: str
    parent: str | None
    evidence: str | None


def _blank(ref: str | None) -> bool:
    return not ref or ref == ZERO_SHA


def resolve_evidence_base(
    event: str,
    head: str,
    push_before: str | None,
    pr_base: str | None,
    commits: Sequence[CommitEvidence],
) -> dict[str, Any]:
    """Return ``{ok, sha, reason, unverified}``.

    ``sha`` is the commit the diff starts at (exclusive). ``unverified``
    lists ancestors strictly between that commit and ``head`` whose
    evidence run was not a completed success, oldest first. ``ok`` is
    false when no green ancestor was found inside the lookback — the
    caller still receives the widest known sha so the comparison steps
    run, and must exit non-zero.
    """
    if event == "pull_request":
        if _blank(pr_base):
            return {"ok": True, "sha": None, "reason": "branch_created", "unverified": []}
        return {"ok": True, "sha": pr_base, "reason": "pr_base", "unverified": []}
    if event != "push":
        return {"ok": False, "sha": None, "reason": "unsupported_event", "unverified": []}
    if _blank(push_before):
        return {"ok": True, "sha": None, "reason": "branch_created", "unverified": []}

    by_sha = {item.sha: item for item in commits}
    cursor = by_sha[head].parent if head in by_sha else (commits[0].sha if commits else None)
    chain: list[CommitEvidence] = []
    seen: set[str] = set()
    while cursor and cursor in by_sha and cursor not in seen:
        seen.add(cursor)
        chain.append(by_sha[cursor])
        cursor = by_sha[cursor].parent

    unverified: list[str] = []
    for item in chain:
        if item.evidence == "success":
            return {
                "ok": True,
                "sha": item.sha,
                "reason": "last_green",
                "unverified": list(reversed(unverified)),
            }
        unverified.append(item.sha)

    furthest = chain[-1] if chain else None
    if furthest is not None and furthest.parent:
        sha: str | None = furthest.parent
    elif furthest is not None:
        sha = furthest.sha
    else:
        sha = None
    reached_before = push_before in seen or push_before == sha
    reason = "no_green_evidence" if reached_before else "lookback_missed_push_before"
    return {
        "ok": False,
        "sha": sha,
        "reason": reason,
        "unverified": list(reversed(unverified)),
    }


def _paged(
    fetch: Callable[[str], Any],
    path: str,
    key: str,
    pages: int,
) -> list[Any]:
    items: list[Any] = []
    for page in range(1, pages + 1):
        sep = "&" if "?" in path else "?"
        doc = fetch(f"{path}{sep}per_page=100&page={page}")
        if isinstance(doc, list):
            batch = doc
        elif isinstance(doc, dict) and isinstance(doc.get(key), list):
            batch = doc[key]
        else:
            break
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
    return items


def _evidence_for(
    sha: str,
    *,
    green: set[str],
    runs_by_sha: Mapping[str, list[Mapping[str, Any]]],
    fetch: Callable[[str], Any],
    repo: str,
    budget: list[int],
) -> str | None:
    if sha in green:
        return "success"
    runs = runs_by_sha.get(sha, [])
    if any(run.get("conclusion") == "success" for run in runs):
        return "success"
    saw_failure = False
    saw_cancelled = False
    ordered = sorted(runs, key=lambda run: str(run.get("created_at") or ""), reverse=True)
    for run in ordered:
        if budget[0] <= 0:
            break
        run_id = run.get("id")
        if not isinstance(run_id, int):
            continue
        budget[0] -= 1
        doc = fetch(f"/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100")
        jobs = doc.get("jobs") if isinstance(doc, dict) else None
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if not isinstance(job, dict) or job.get("name") != "evidence-audit":
                continue
            conclusion = job.get("conclusion")
            if conclusion == "success":
                return "success"
            if conclusion == "failure":
                saw_failure = True
            elif conclusion == "cancelled":
                saw_cancelled = True
    if saw_failure:
        return "failure"
    if saw_cancelled or any(run.get("conclusion") == "cancelled" for run in runs):
        return "cancelled"
    if any(run.get("conclusion") == "failure" for run in runs):
        return "failure"
    return None


def collect_commit_evidence(
    head: str,
    *,
    repo: str,
    push_before: str | None,
    fetch: Callable[[str], Any],
    max_commit_pages: int = 5,
    max_job_fetches: int = 40,
) -> list[CommitEvidence]:
    """First-parent chain from ``head`` back through the last green run.

    ``fetch`` is ``gh api`` (or a test double) and receives a path beginning
    with ``/repos/...``. Workflow success and an ``evidence-green`` artifact
    both count as a completed green evidence run. HEAD itself is never the
    base; a success on an ancestor stops the walk.
    """
    commits = _paged(
        fetch,
        f"/repos/{repo}/commits?sha={head}",
        "commits",
        max_commit_pages,
    )
    by_sha: dict[str, Mapping[str, Any]] = {}
    for item in commits:
        if isinstance(item, dict) and isinstance(item.get("sha"), str):
            by_sha[item["sha"]] = item
    artifacts = _paged(
        fetch,
        f"/repos/{repo}/actions/artifacts?name=evidence-green",
        "artifacts",
        2,
    )
    green: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        run = artifact.get("workflow_run")
        if isinstance(run, dict) and isinstance(run.get("head_sha"), str):
            green.add(run["head_sha"])
    runs = _paged(
        fetch,
        f"/repos/{repo}/actions/workflows/ci.yml/runs",
        "workflow_runs",
        max_commit_pages,
    )
    runs_by_sha: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("head_sha"), str):
            continue
        runs_by_sha.setdefault(run["head_sha"], []).append(run)

    budget = [max_job_fetches]
    chain: list[CommitEvidence] = []
    cursor: str | None = head
    seen: set[str] = set()
    while cursor and cursor in by_sha and cursor not in seen and len(chain) < 500:
        seen.add(cursor)
        item = by_sha[cursor]
        parents = item.get("parents")
        parent = None
        if isinstance(parents, list) and parents and isinstance(parents[0], dict):
            raw_parent = parents[0].get("sha")
            parent = raw_parent if isinstance(raw_parent, str) else None
        evidence = _evidence_for(
            cursor,
            green=green,
            runs_by_sha=runs_by_sha,
            fetch=fetch,
            repo=repo,
            budget=budget,
        )
        if evidence not in _EVIDENCE:
            evidence = None
        chain.append(CommitEvidence(cursor, parent, evidence))
        if cursor != head and evidence == "success":
            break
        if cursor == push_before and cursor != head and evidence == "success":
            break
        cursor = parent
    return chain


def _gh_json(path: str) -> Any:
    proc = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "gh api failed").strip()
        raise RuntimeError(err)
    return json.loads(proc.stdout)


def _load_commits(path: Path) -> list[CommitEvidence]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("commits-json must be a list")
    commits: list[CommitEvidence] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("sha"), str):
            raise ValueError("commits-json entry needs a sha")
        parent = item.get("parent")
        evidence = item.get("evidence")
        commits.append(
            CommitEvidence(
                sha=item["sha"],
                parent=parent if isinstance(parent, str) else None,
                evidence=evidence if evidence in _EVIDENCE else None,
            )
        )
    return commits


def _emit(result: Mapping[str, Any]) -> None:
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant_fund.research.evidence_base")
    parser.add_argument("--event", default="push")
    parser.add_argument("--head", required=True)
    parser.add_argument("--push-before", default="")
    parser.add_argument("--pr-base", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--commits-json", default="")
    args = parser.parse_args(argv)
    try:
        if args.commits_json:
            commits = _load_commits(Path(args.commits_json))
        elif args.event == "pull_request":
            commits = []
        else:
            repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
            if not repo:
                raise RuntimeError("GITHUB_REPOSITORY is required to resolve evidence runs")
            commits = collect_commit_evidence(
                args.head,
                repo=repo,
                push_before=args.push_before or None,
                fetch=_gh_json,
            )
        result = resolve_evidence_base(
            args.event,
            args.head,
            args.push_before or None,
            args.pr_base or None,
            commits,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        _emit(
            {
                "detail": str(exc),
                "ok": False,
                "reason": "evidence_base_error",
                "sha": None,
                "unverified": [],
            }
        )
        return 1
    _emit(result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
