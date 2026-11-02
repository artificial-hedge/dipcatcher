"""Run the pinned security gate and write ``.tmp_security/summary.json``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from security_gate_lib import (
    ACCEPTED_SCHEMA,
    ALLOW_SCHEMA,
    SCHEMA,
    TOOLS_SCHEMA,
    TrivyVuln,
    allow_map,
    commit_download,
    dated_entries,
    digest_recorded,
    git_head,
    load_json,
    parse_bandit,
    parse_hadolint,
    parse_osv,
    parse_pip_audit,
    parse_trivy,
    parse_zap,
    scrub_gitleaks,
    sha256_file,
    strip_markers,
    tool_record,
    trivy_failures,
    utc_today,
    worktree_dirty,
)

LAB_IMAGE = "dipcatcher:supply-hardened"
RESEARCH_IMAGE = "dipcatcher-research-api:supply"
DOCKERFILES = ("Dockerfile", "deploy/lab.Dockerfile", "docker/research-api.Dockerfile")
_ZAP_OK = frozenset({0, 1, 2})


def download_verified(url: str, dest: Path, expected: str) -> None:
    if dest.is_file() and sha256_file(dest) == expected:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "-o", str(partial), url],
        check=True,
    )
    commit_download(partial, dest, expected)


def materialize(spec: dict[str, Any], archive: Path, bindir: Path) -> Path:
    kind = str(spec["kind"])
    binary = bindir / str(spec["binary"])
    if kind == "wheel":
        return archive
    if kind == "file":
        shutil.copyfile(archive, binary)
        binary.chmod(0o755)
        return binary
    if kind != "tar":
        raise SystemExit(f"unknown tool kind {kind}")
    subprocess.run(
        ["tar", "-xzf", str(archive), "-C", str(bindir), str(spec["binary"])],
        check=True,
    )
    binary.chmod(0o755)
    return binary


def require_image_digest(ref: str, digest: str) -> None:
    subprocess.run(["docker", "pull", ref], check=True)
    inspected = subprocess.check_output(
        ["docker", "image", "inspect", ref, "--format", "{{.Id}}"],
        text=True,
    )
    if not digest_recorded(inspected, digest):
        raise SystemExit("zap image digest mismatch")


def install_tools(root: Path, lock: dict[str, Any]) -> dict[str, Path]:
    if lock.get("schema") != TOOLS_SCHEMA:
        raise SystemExit("tools lock schema")
    bindir = root / ".tmp_security" / "bin"
    distdir = root / ".tmp_security" / "dist"
    bindir.mkdir(parents=True, exist_ok=True)
    distdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, spec in lock["tools"].items():
        if spec["kind"] == "image":
            require_image_digest(str(spec["image"]), str(spec["sha256"]))
            paths[name] = Path(str(spec["image"]))
            continue
        dest = distdir / str(spec["filename"])
        download_verified(str(spec["url"]), dest, str(spec["sha256"]))
        paths[name] = materialize(spec, dest, bindir)
    return paths


def _record_ids(
    spec: dict[str, Any],
    ids: list[str],
    accepted: list[dict[str, Any]],
    today: Any,
) -> dict[str, Any]:
    return tool_record(str(spec["version"]), str(spec["sha256"]), ids, accepted, today)


def run_bandit(
    wheel: Path, root: Path, spec: dict[str, Any], accepted: list[dict[str, Any]], today: Any
) -> dict[str, Any]:
    report = root / ".tmp_security" / "bandit.json"
    proc = subprocess.run(
        [
            "uvx",
            "--from",
            str(wheel),
            "bandit",
            "-q",
            "-r",
            "src",
            "-f",
            "json",
            "-o",
            str(report),
            "--severity-level",
            "medium",
            "--confidence-level",
            "medium",
        ],
        cwd=root,
        capture_output=True,
    )
    if proc.returncode not in {0, 1} or not report.is_file():
        return tool_record(
            str(spec["version"]), str(spec["sha256"]), [], accepted, today, "bandit failed"
        )
    return _record_ids(spec, parse_bandit(load_json(report), root), accepted, today)


def export_requirements(root: Path) -> Path:
    dest = root / ".tmp_security" / "requirements.txt"
    raw = root / ".tmp_security" / "requirements.raw.txt"
    subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--quiet",
            "--format",
            "requirements.txt",
            "--no-hashes",
            "--no-emit-project",
            "--all-extras",
            "--all-groups",
            "-o",
            str(raw),
        ],
        cwd=root,
        check=True,
    )
    dest.write_text(strip_markers(raw.read_text(encoding="utf-8")), encoding="utf-8")
    return dest


def run_pip_audit(
    wheel: Path,
    root: Path,
    spec: dict[str, Any],
    accepted: list[dict[str, Any]],
    today: Any,
) -> dict[str, Any]:
    requirements = export_requirements(root)
    report = root / ".tmp_security" / "pip-audit.json"
    proc = subprocess.run(
        [
            "uvx",
            "--from",
            str(wheel),
            "pip-audit",
            "--format",
            "json",
            "--disable-pip",
            "--no-deps",
            "-r",
            str(requirements),
            "-o",
            str(report),
        ],
        cwd=root,
        capture_output=True,
    )
    if proc.returncode not in {0, 1} or not report.is_file():
        return tool_record(
            str(spec["version"]), str(spec["sha256"]), [], accepted, today, "pip-audit failed"
        )
    return _record_ids(spec, parse_pip_audit(load_json(report)), accepted, today)


def run_osv(
    binary: Path, root: Path, spec: dict[str, Any], accepted: list[dict[str, Any]], today: Any
) -> dict[str, Any]:
    report = root / ".tmp_security" / "osv.json"
    proc = subprocess.run(
        [
            str(binary),
            "scan",
            "source",
            "-L",
            "uv.lock",
            "--format",
            "json",
            "--output-file",
            str(report),
            "--verbosity",
            "error",
        ],
        cwd=root,
        capture_output=True,
    )
    if proc.returncode not in {0, 1} or not report.is_file():
        return tool_record(
            str(spec["version"]), str(spec["sha256"]), [], accepted, today, "osv-scanner failed"
        )
    return _record_ids(spec, parse_osv(load_json(report)), accepted, today)


def _gitleaks(
    binary: Path, root: Path, report: Path, extra: list[str], data: bytes | None
) -> list[str]:
    cmd = [
        str(binary),
        "detect",
        "--source",
        str(root),
        "--config",
        str(root / ".gitleaks.toml"),
        "--redact",
        "--no-banner",
        "--report-format",
        "json",
        "--report-path",
        str(report),
        *extra,
    ]
    proc = subprocess.run(cmd, cwd=root, input=data, capture_output=True)
    try:
        if proc.returncode not in {0, 1}:
            raise SystemExit("gitleaks failed")
        if report.is_file():
            return scrub_gitleaks(report)
        if proc.returncode == 0:
            return []
        raise SystemExit("gitleaks failed")
    finally:
        report.unlink(missing_ok=True)


def _gitleaks_pipe(binary: Path, root: Path, report: Path, diff: bytes) -> list[str]:
    cmd = [
        str(binary),
        "detect",
        "--pipe",
        "--no-git",
        "--config",
        str(root / ".gitleaks.toml"),
        "--redact",
        "--no-banner",
        "--report-format",
        "json",
        "--report-path",
        str(report),
    ]
    proc = subprocess.run(cmd, cwd=report.parent, input=diff, capture_output=True)
    try:
        if proc.returncode not in {0, 1}:
            raise SystemExit("gitleaks diff failed")
        if report.is_file():
            return scrub_gitleaks(report)
        return []
    finally:
        report.unlink(missing_ok=True)


def _untracked_ids(binary: Path, root: Path, work: Path) -> list[str]:
    listed = subprocess.check_output(
        ["git", "ls-files", "-o", "--exclude-standard"],
        cwd=root,
        text=True,
    )
    names = [line for line in listed.splitlines() if line]
    if not names:
        return []
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        dest_root = Path(tmp)
        _copy_named(root, dest_root, names)
        report = work / "untracked.json"
        return _gitleaks(binary, dest_root, report, ["--no-git"], None)


def _copy_named(root: Path, dest_root: Path, names: list[str]) -> None:
    for name in names:
        src = root / name
        if not src.is_file():
            continue
        dest = dest_root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


def run_gitleaks(
    binary: Path, root: Path, spec: dict[str, Any], accepted: list[dict[str, Any]], today: Any
) -> dict[str, Any]:
    work = root / ".tmp_security" / "gitleaks"
    work.mkdir(parents=True, exist_ok=True)
    ids = _gitleaks(binary, root, work / "history.json", ["--log-opts=--full-history"], None)
    base = os.environ.get("SECURITY_GATE_BASE", "").strip()
    if base:
        ids.extend(_gitleaks(binary, root, work / "range.json", [f"--log-opts={base}..HEAD"], None))
    diff = subprocess.check_output(["git", "diff", base or "HEAD"], cwd=root)
    if diff.strip():
        ids.extend(_gitleaks_pipe(binary, root, work / "diff.json", diff))
    ids.extend(_untracked_ids(binary, root, work))
    return _record_ids(spec, ids, accepted, today)


def run_hadolint(
    binary: Path, root: Path, spec: dict[str, Any], accepted: list[dict[str, Any]], today: Any
) -> dict[str, Any]:
    ids: list[str] = []
    for name in DOCKERFILES:
        proc = subprocess.run(
            [str(binary), "-f", "json", name],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if proc.returncode not in {0, 1}:
            return tool_record(
                str(spec["version"]), str(spec["sha256"]), [], accepted, today, "hadolint failed"
            )
        ids.extend(parse_hadolint(_hadolint_items(proc.stdout)))
    return _record_ids(spec, ids, accepted, today)


def _hadolint_items(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    data = json.loads(stripped)
    if not isinstance(data, list):
        return []
    return data


def build_images(root: Path) -> None:
    subprocess.run(
        ["docker", "build", "-t", "dipcatcher:supply", "-f", str(root / "Dockerfile"), str(root)],
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            LAB_IMAGE,
            "--build-arg",
            "BASE=dipcatcher:supply",
            "-f",
            str(root / "deploy" / "lab.Dockerfile"),
            str(root),
        ],
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            RESEARCH_IMAGE,
            "-f",
            str(root / "docker" / "research-api.Dockerfile"),
            str(root),
        ],
        check=True,
    )


def run_trivy(
    binary: Path, root: Path, spec: dict[str, Any], allow: dict[str, Any], today: Any
) -> dict[str, Any]:
    vulns: list[TrivyVuln] = []
    for image_name, image in (("lab", LAB_IMAGE), ("research", RESEARCH_IMAGE)):
        report = root / ".tmp_security" / f"trivy-{image_name}.json"
        proc = subprocess.run(
            [
                str(binary),
                "image",
                "--quiet",
                "--severity",
                "HIGH,CRITICAL",
                "--format",
                "json",
                "--output",
                str(report),
                image,
            ],
            cwd=root,
            capture_output=True,
        )
        if proc.returncode != 0 or not report.is_file():
            return tool_record(
                str(spec["version"]), str(spec["sha256"]), [], [], today, "trivy failed"
            )
        vulns.extend(parse_trivy(load_json(report), image_name))
    failing = trivy_failures(vulns, allow, today)
    observed = [f"trivy:{item.image}:{item.vuln_id}" for item in vulns]
    record = tool_record(str(spec["version"]), str(spec["sha256"]), observed, [], today)
    record["new_ids"] = failing
    record["ok"] = not failing
    return record


def _request(
    url: str, data: bytes | None, headers: dict[str, str], timeout: int
) -> tuple[int, dict[str, str]]:
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status, {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        exc.read()
        return exc.code, {key.lower(): value for key, value in exc.headers.items()}
    except (urllib.error.URLError, OSError):
        # Connection reset while uvicorn is still binding is a down service.
        return 0, {}


def health_findings(name: str, url: str) -> list[str]:
    status, headers = _request(url, None, {}, 10)
    ids: list[str] = []
    if status != 200:
        ids.append(f"service:{name}:status:{status}")
    if headers.get("cross-origin-resource-policy") != "same-origin":
        ids.append(f"service:{name}:corp")
    if headers.get("cross-origin-opener-policy") != "same-origin":
        ids.append(f"service:{name}:coop")
    if headers.get("server"):
        ids.append(f"service:{name}:server")
    return ids


def post_findings(name: str, url: str, key: str) -> list[str]:
    status, _headers = _request(
        url,
        b"{}",
        {"X-API-Key": key, "Content-Type": "application/json"},
        180,
    )
    if status >= 500 or status == 0:
        return [f"service:{name}:{status}"]
    return []


def service_ids(key: str) -> list[str]:
    ids = health_findings("lab", "http://127.0.0.1:8000/health")
    ids.extend(health_findings("research", "http://127.0.0.1:8010/health"))
    ids.extend(post_findings("lab-backtest", "http://127.0.0.1:8000/backtest", key))
    ids.extend(post_findings("lab-optimize", "http://127.0.0.1:8000/portfolio/optimize", key))
    return ids


def wait_healthy() -> None:
    for _ in range(90):
        lab, _headers = _request("http://127.0.0.1:8000/health", None, {}, 2)
        research, _ignored = _request("http://127.0.0.1:8010/health", None, {}, 2)
        if lab == 200 and research == 200:
            return
        subprocess.run(["sleep", "1"], check=False)
    raise SystemExit("services did not become healthy")


def start_services(root: Path) -> str:
    key = os.urandom(32).hex()
    env_path = root / ".tmp_security" / "compose.env"
    env_path.write_text(f"QUANT_API_KEY={key}\nRESEARCH_API_KEY={key}\n", encoding="utf-8")
    env_path.chmod(0o600)
    env = os.environ.copy()
    env["QUANT_API_KEY"] = key
    env["RESEARCH_API_KEY"] = key
    proc = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            "dipcatcher-supply",
            "-f",
            str(root / "docker-compose.yml"),
            "-f",
            str(root / "deploy" / "compose.hardened.yml"),
            "--env-file",
            str(env_path),
            "up",
            "-d",
            "--force-recreate",
            "--no-deps",
            "api",
            "research-api",
        ],
        cwd=root,
        env=env,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise SystemExit("compose up failed")
    wait_healthy()
    return key


def stop_env_file(root: Path) -> None:
    (root / ".tmp_security" / "compose.env").unlink(missing_ok=True)


def one_baseline(image: str, target: str, work: Path, report_name: str, port: int) -> list[str]:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    work.chmod(0o777)
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "-v",
            f"{work}:/zap/wrk:rw",
            image,
            "zap-baseline.py",
            "-P",
            str(port),
            "-t",
            target,
            "-J",
            report_name,
            "-I",
            "-m",
            "1",
            "--autooff",
        ],
        capture_output=True,
    )
    report = work / report_name
    if proc.returncode not in _ZAP_OK or not report.is_file():
        raise SystemExit("zap baseline failed")
    return parse_zap(load_json(report))


def run_baselines(image: str, root: Path) -> list[str]:
    work = root / ".tmp_security" / "zap"
    ids = one_baseline(image, "http://127.0.0.1:8000", work / "lab", "lab-baseline.json", 18091)
    ids.extend(
        one_baseline(
            image, "http://127.0.0.1:8010", work / "research", "research-baseline.json", 18092
        )
    )
    return sorted(set(ids))


def run_runtime(
    root: Path,
    zap_spec: dict[str, Any],
    accepted: list[dict[str, Any]],
    today: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    build_images(root)
    try:
        key = start_services(root)
        zap_ids = run_baselines(str(zap_spec["image"]), root)
        service = service_ids(key)
    finally:
        stop_env_file(root)
    zap = _record_ids(zap_spec, zap_ids, accepted, today)
    service_spec = {"version": "runtime", "sha256": zap_spec["sha256"]}
    return zap, _record_ids(service_spec, service, accepted, today)


def collect_new(tools: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for tool in tools.values():
        found.extend(tool.get("new_ids") or [])
    return sorted(set(found))


def execute(root: Path, summary: dict[str, Any]) -> int:
    today = utc_today()
    lock = load_json(root / "deploy" / "security" / "tools.lock.json")
    allow_entries = dated_entries(
        load_json(root / "deploy" / "security" / "trivy-allow.json"),
        "entries",
        ALLOW_SCHEMA,
    )
    accepted = dated_entries(
        load_json(root / "deploy" / "security" / "accepted-findings.json"),
        "findings",
        ACCEPTED_SCHEMA,
    )
    paths = install_tools(root, lock)
    specs = lock["tools"]
    tools: dict[str, Any] = {}
    tools["bandit"] = run_bandit(paths["bandit"], root, specs["bandit"], accepted, today)
    tools["pip-audit"] = run_pip_audit(
        paths["pip-audit"], root, specs["pip-audit"], accepted, today
    )
    tools["osv-scanner"] = run_osv(
        paths["osv-scanner"], root, specs["osv-scanner"], accepted, today
    )
    tools["gitleaks"] = run_gitleaks(paths["gitleaks"], root, specs["gitleaks"], accepted, today)
    tools["hadolint"] = run_hadolint(paths["hadolint"], root, specs["hadolint"], accepted, today)
    tools["trivy"] = run_trivy(
        paths["trivy"], root, specs["trivy"], allow_map(allow_entries), today
    )
    tools["zap"], tools["service"] = run_runtime(root, specs["zap"], accepted, today)
    summary["tools"] = tools
    summary["new_findings"] = collect_new(tools)
    summary["dirty"] = worktree_dirty(root)
    summary["ok"] = not summary["new_findings"] and not summary["dirty"] and _tools_ok(tools)
    return 0 if summary["ok"] else 1


def _tools_ok(tools: dict[str, Any]) -> bool:
    return all(bool(tool.get("ok")) for tool in tools.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    summary: dict[str, Any] = {
        "dirty": True,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": "",
        "new_findings": [],
        "ok": False,
        "schema": SCHEMA,
        "tools": {},
    }
    code = 1
    try:
        summary["head"] = git_head(root)
        code = execute(root, summary)
    except (Exception, SystemExit) as exc:
        summary["error"] = type(exc).__name__
        summary["ok"] = False
        code = 1
    out = root / ".tmp_security" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"security-gate ok={summary['ok']} new={len(summary['new_findings'])}")
    for item in summary["new_findings"]:
        print(item)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
