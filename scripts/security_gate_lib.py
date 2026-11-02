"""Pure checks for the security gate. The CLI lives in security_gate.py."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

SCHEMA = "dipcatcher.security_gate/1"
TOOLS_SCHEMA = "dipcatcher.security_tools/1"
ALLOW_SCHEMA = "dipcatcher.trivy_allow/1"
ACCEPTED_SCHEMA = "dipcatcher.security_accepted/1"
REQUIRED_TOOLS = (
    "bandit",
    "gitleaks",
    "hadolint",
    "osv-scanner",
    "pip-audit",
    "service",
    "trivy",
    "zap",
)
_HIGH = frozenset({"HIGH", "CRITICAL"})


@dataclass(frozen=True)
class TrivyVuln:
    image: str
    vuln_id: str
    severity: str
    fixable: bool


def utc_today() -> date:
    return datetime.now(UTC).date()


def digest_recorded(inspected_id: str, digest: str) -> bool:
    """Return whether ``docker image inspect`` recorded ``digest``."""
    token = digest.removeprefix("sha256:").strip()
    return bool(token) and token in inspected_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit_download(partial: Path, dest: Path, expected: str) -> None:
    """Move ``partial`` onto ``dest`` when its sha256 matches ``expected``."""
    digest = sha256_file(partial)
    if digest != expected:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"sha256 mismatch for {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(dest)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dated_entries(payload: dict[str, Any], key: str, schema: str) -> list[dict[str, Any]]:
    if payload.get("schema") != schema:
        raise SystemExit(f"{key} schema")
    entries = payload.get(key)
    if not isinstance(entries, list):
        raise SystemExit(f"{key} must be a list")
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("expiry"):
            raise SystemExit(f"{key} entry needs id and expiry")
        date.fromisoformat(str(entry["expiry"]))
    return entries


def finding_accepted(finding_id: str, entries: list[dict[str, Any]], today: date) -> bool:
    for entry in entries:
        if entry.get("id") != finding_id:
            continue
        return date.fromisoformat(str(entry["expiry"])) >= today
    return False


def allow_map(entries: list[dict[str, Any]]) -> dict[str, date]:
    return {str(entry["id"]): date.fromisoformat(str(entry["expiry"])) for entry in entries}


def parse_trivy(report: dict[str, Any], image: str) -> list[TrivyVuln]:
    grouped: dict[tuple[str, str], TrivyVuln] = {}
    for result in report.get("Results") or []:
        grouped.update(_trivy_result(result, image, grouped))
    return list(grouped.values())


def _trivy_result(
    result: dict[str, Any],
    image: str,
    grouped: dict[tuple[str, str], TrivyVuln],
) -> dict[tuple[str, str], TrivyVuln]:
    for vuln in result.get("Vulnerabilities") or []:
        severity = str(vuln.get("Severity") or "")
        if severity not in _HIGH:
            continue
        vuln_id = str(vuln.get("VulnerabilityID") or "")
        fixable = bool(str(vuln.get("FixedVersion") or "").strip())
        key = (image, vuln_id)
        current = grouped.get(key)
        if current is None or fixable:
            grouped[key] = TrivyVuln(image, vuln_id, severity, fixable)
    return grouped


def trivy_failures(vulns: list[TrivyVuln], allow: dict[str, date], today: date) -> list[str]:
    failing: list[str] = []
    for vuln in vulns:
        if vuln.severity not in _HIGH:
            continue
        failing.extend(_trivy_one(vuln, allow, today))
    return sorted(set(failing))


def _trivy_one(vuln: TrivyVuln, allow: dict[str, date], today: date) -> list[str]:
    if vuln.fixable:
        return [f"trivy:{vuln.image}:fixable:{vuln.vuln_id}"]
    expiry = allow.get(vuln.vuln_id)
    if expiry is None:
        return [f"trivy:{vuln.image}:unlisted:{vuln.vuln_id}"]
    if expiry < today:
        return [f"trivy:{vuln.image}:expired:{vuln.vuln_id}"]
    return []


def new_ids(ids: list[str], accepted: list[dict[str, Any]], today: date) -> list[str]:
    return sorted(item for item in set(ids) if not finding_accepted(item, accepted, today))


def tool_record(
    version: str,
    digest: str,
    ids: list[str],
    accepted: list[dict[str, Any]],
    today: date,
    error: str | None = None,
) -> dict[str, Any]:
    fresh = new_ids(ids, accepted, today)
    return {
        "error": error,
        "finding_ids": sorted(set(ids)),
        "new_ids": fresh,
        "ok": error is None and not fresh,
        "sha256": digest,
        "version": version,
    }


def parse_bandit(report: dict[str, Any], root: Path) -> list[str]:
    ids: list[str] = []
    for item in report.get("results") or []:
        ids.append(_bandit_id(item, root))
    return sorted(set(ids))


def _bandit_id(item: dict[str, Any], root: Path) -> str:
    path = Path(str(item.get("filename") or ""))
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = path
    return f"bandit:{item.get('test_id')}:{relative}:{item.get('line_number')}"


def parse_hadolint(items: list[dict[str, Any]]) -> list[str]:
    return sorted(
        f"hadolint:{item.get('code')}:{item.get('file')}:{item.get('line')}" for item in items
    )


def parse_pip_audit(report: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for dep in report.get("dependencies") or []:
        name = dep.get("name")
        for vuln in dep.get("vulns") or []:
            ids.append(f"pip-audit:{vuln.get('id')}:{name}")
    return sorted(set(ids))


def parse_osv(report: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for result in report.get("results") or []:
        for package in result.get("packages") or []:
            name = (package.get("package") or {}).get("name")
            for vuln in package.get("vulnerabilities") or []:
                ids.append(f"osv:{vuln.get('id')}:{name}")
    return sorted(set(ids))


def parse_gitleaks(report: list[Any]) -> list[str]:
    ids: list[str] = []
    for item in report:
        if not isinstance(item, dict):
            continue
        ids.append(f"gitleaks:{item.get('RuleID')}:{item.get('File')}:{item.get('StartLine')}")
    return sorted(set(ids))


def scrub_gitleaks(path: Path) -> list[str]:
    """Read a gitleaks report and delete it. The secret fields are not kept."""
    if not path.exists() or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)
    if not isinstance(data, list):
        return []
    return parse_gitleaks(data)


def parse_zap(report: dict[str, Any]) -> list[str]:
    sites = report.get("site") or []
    if isinstance(sites, dict):
        sites = [sites]
    ids: list[str] = []
    for site in sites:
        for alert in site.get("alerts") or []:
            ids.append(f"zap:{alert.get('pluginid')}:{alert.get('name')}")
    return sorted(set(ids))


def strip_markers(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        marker = line.find(" ;")
        lines.append(line if marker < 0 else line[:marker])
    return "\n".join(lines) + "\n"


def evaluate_summary(summary: dict[str, Any], head: str) -> list[str]:
    errors: list[str] = []
    if summary.get("schema") != SCHEMA:
        errors.append("schema")
    if summary.get("head") != head:
        errors.append("head")
    if summary.get("ok") is not True:
        errors.append("ok")
    if summary.get("dirty") is not False:
        errors.append("dirty")
    if summary.get("new_findings"):
        errors.append("new_findings")
    tools = summary.get("tools")
    if not isinstance(tools, dict):
        errors.append("tools")
        return errors
    for name in REQUIRED_TOOLS:
        tool = tools.get(name)
        if not isinstance(tool, dict) or tool.get("ok") is not True:
            errors.append(f"tool:{name}")
    return errors


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def worktree_dirty(root: Path) -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(proc.stdout.strip())
