"""Owner restamp, the kill-audit note, the restore drill, and host hardening."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import socket
import subprocess
from pathlib import Path

import pytest

from quant_fund.monitoring.kill_audit import kill_switch_audit
from quant_fund.research.corpus_epoch import load_heads_pin, member_digests, member_tree_root

_ROOT = Path(__file__).resolve().parents[3]
_KEY = "planted-test-key-value-0123456789"
_OTHER = "metrics-test-key-0123456789"
_RESEAL = (
    "uv run python -c 'import json; from pathlib import Path; "
    "from quant_fund.monitoring.kill_audit import kill_audit_bench; "
    'Path("receipts/kill_audit.json").write_text(json.dumps(kill_audit_bench(), indent=2) '
    '+ "\\n", encoding="utf-8")\''
)
_ADDED = (
    "FLATTEN|auto=False:orders",
    "FLATTEN|auto=False:cancel",
    "FLATTEN|auto=False:flatten(human=False)",
    "FLATTEN|auto=False:flatten(human=True)",
    "FLATTEN|auto=True:orders",
    "FLATTEN|auto=True:cancel",
    "FLATTEN|auto=True:flatten(human=False)",
    "FLATTEN|auto=True:flatten(human=True)",
    "transition:ENABLED->FLATTEN",
)
_REMOVED = (
    "unknown['FLATTEN']:orders",
    "unknown['FLATTEN']:cancel+flatten",
    "set_state['FLATTEN']",
)
_UFW = """Status: active
Default: deny (incoming), allow (outgoing), deny (routed)
443/tcp                    ALLOW IN    Anywhere
8443/tcp                   ALLOW IN    Anywhere
8444/tcp                   ALLOW IN    Anywhere
443/tcp (v6)               ALLOW IN    Anywhere (v6)
8443/tcp (v6)              ALLOW IN    Anywhere (v6)
8444/tcp (v6)              ALLOW IN    Anywhere (v6)
"""
_IPTABLES = """-P INPUT DROP
-P FORWARD DROP
-P OUTPUT ACCEPT
-A INPUT -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
-A INPUT -p tcp -m tcp --dport 443 -j ACCEPT
-A INPUT -p tcp -m multiport --dports 8443,8444 -j ACCEPT
"""
_SSHD = """Port 22
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
# PasswordAuthentication yes
"""
_APT = 'APT::Periodic::Unattended-Upgrade "1";\n'
_BASE_CHECKS = {"env", "certs", "ports", "disk", "write-owner", "clock"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _restamp():
    return _load(_ROOT / "scripts" / "owner_restamp.py", "dipcatcher_owner_restamp_ops11")


def _drill():
    return _load(_ROOT / "scripts" / "prod_dr_drill.py", "dipcatcher_prod_dr_drill_ops11")


def _preflight():
    return _load(_ROOT / "scripts" / "prod_preflight.py", "dipcatcher_prod_preflight_ops11")


def _pins() -> dict[str, dict[str, str]]:
    script = _restamp()
    return {
        f"{name}/*": {"receipt": f"{name}.json", "sha256": "b" * 64, "tree_root": "a" * 64}
        for name in script.CORPORA
    }


def _trees() -> dict[str, str]:
    script = _restamp()
    return {name: "a" * 64 for name in script.CORPORA}


def _no_subprocess(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "subprocess" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "subprocess"


def _mint(directory: Path) -> None:
    directory.mkdir()
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            "30",
            "-nodes",
            "-keyout",
            str(directory / "dev.key"),
            "-out",
            str(directory / "dev.crt"),
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    (directory / "dev.key").chmod(0o600)


def _free_ports(count: int) -> list[int]:
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tls = tmp_path / "tls"
    lab = tmp_path / "lab"
    research = tmp_path / "research"
    _mint(tls)
    lab.mkdir()
    research.mkdir()
    monkeypatch.setenv("DIPCATCHER_TLS_DIR", str(tls))
    monkeypatch.setenv("DIPCATCHER_LAB_DATA", str(lab))
    monkeypatch.setenv("DIPCATCHER_RESEARCH_DATA", str(research))
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    monkeypatch.setenv("RESEARCH_API_KEY", _OTHER)
    free = _free_ports(4)
    monkeypatch.setenv("LAB_TLS_PORT", str(free[0]))
    monkeypatch.setenv("RESEARCH_TLS_PORT", str(free[1]))
    monkeypatch.setenv("PROMETHEUS_PUBLISH_PORT", str(free[2]))
    monkeypatch.setenv("GRAFANA_PUBLISH_PORT", str(free[3]))


def _good_probes() -> dict[str, str]:
    return {
        "apt": _APT,
        "firewall": _UFW,
        "firewall6": "",
        "service": "enabled\n",
        "sshd": _SSHD,
        "time": "yes\n",
    }


def test_owner_steps_are_the_documented_order() -> None:
    script = _restamp()
    note = (_ROOT / "docs" / "KILL_AUDIT_RECONCILIATION.md").read_text(encoding="utf-8")
    runbook = (_ROOT / "docs" / "OPS_RUNBOOK_PROD.md").read_text(encoding="utf-8")
    assert list(script.OWNER_STEPS) == [
        "make stamp-epochs",
        "make sign-pins",
        "make anchor-pins",
        "make checkpoint",
    ]
    positions = [note.index(command) for command in script.OWNER_STEPS]
    assert positions == sorted(positions)
    runbook_positions = [runbook.index(command) for command in script.OWNER_STEPS]
    assert runbook_positions == sorted(runbook_positions)
    _no_subprocess(_ROOT / "scripts" / "owner_restamp.py")


def test_drift_report_names_only_moved_corpora_in_corpus_order() -> None:
    script = _restamp()
    pins = _pins()
    trees = _trees()
    trees["docs"] = "c" * 64
    trees["src"] = "d" * 64
    report = script.drift_report(pins, trees)
    assert report["drifted"] == ["docs", "src"]
    assert report["corpora"] == list(script.CORPORA)
    assert report["owner_steps"] == list(script.OWNER_STEPS)
    assert report["read_only"] is True
    assert report["signed"] is False
    docs = next(item for item in report["epochs"] if item["corpus"] == "docs")
    assert docs["receipt"] == "docs.json"
    assert docs["drifted"] is True


def test_matching_trees_still_print_owner_steps() -> None:
    script = _restamp()
    report = script.drift_report(_pins(), _trees())
    assert report["drifted"] == []
    assert report["owner_steps"] == list(script.OWNER_STEPS)
    assert script.drift_report({}, _trees())["drifted"] == list(script.CORPORA)


def test_live_report_is_read_only(capsys: pytest.CaptureFixture[str]) -> None:
    pin_path = _ROOT / "quality" / "epoch_heads.json"
    before = pin_path.read_bytes()
    script = _restamp()
    assert script.main(["owner_restamp.py"]) == 0
    assert pin_path.read_bytes() == before
    payload = json.loads(capsys.readouterr().out)
    assert payload["signed"] is False
    assert payload["read_only"] is True
    assert payload["owner_steps"] == list(script.OWNER_STEPS)
    assert payload["drifted"] == [item["corpus"] for item in payload["epochs"] if item["drifted"]]
    for item in payload["epochs"]:
        moved = (not item["pinned_tree_root"]) or item["pinned_tree_root"] != item["tree_root"]
        assert item["drifted"] is moved
    pins = load_heads_pin(pin_path)
    deploy = next(item for item in payload["epochs"] if item["corpus"] == "deploy")
    expected = member_tree_root(member_digests(_ROOT / "deploy", pattern="*"))
    assert deploy["tree_root"] == expected
    assert deploy["receipt"] == pins["deploy/*"]["receipt"]
    assert script.main(["owner_restamp.py", "--sign"]) == 1
    assert json.loads(capsys.readouterr().out)["signed"] is False


def test_kill_audit_note_names_the_gap_and_the_receipt_stays_sealed() -> None:
    receipt = json.loads((_ROOT / "receipts" / "kill_audit.json").read_text(encoding="utf-8"))
    note = (_ROOT / "docs" / "KILL_AUDIT_RECONCILIATION.md").read_text(encoding="utf-8")
    assert receipt["claim"]["n_checks"] == 57
    assert receipt["interpretation"]["n_checks"] == 57
    assert (
        receipt["receipt_sha256"]
        == "493cd55611dd273c687b0e94c6b3b6e11ed233ee9a3381cd498b7c38eb280e7c"
    )
    assert kill_switch_audit()["n_checks"] == 63
    assert "57 - 3 + 9 = 63" in note
    assert _RESEAL in note
    for label in _ADDED:
        assert label in note
    for label in _REMOVED:
        assert label in note
    assert "ca9f305b4" in note
    assert "94836f167" in note


def test_dr_target_stays_out_of_the_root_makefile() -> None:
    prod = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    root = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "prod-dr-drill:" in prod
    assert "scripts/prod_dr_drill.py" in prod
    assert "prod-dr-drill:" not in root
    assert "prod-up: prod-certs prod-data prod-preflight prod-build" in prod
    runbook = (_ROOT / "docs" / "OPS_RUNBOOK_PROD.md").read_text(encoding="utf-8")
    assert "## Disaster recovery" in runbook
    assert "docs/HOST_HARDENING.md" in runbook
    assert "docs/KILL_AUDIT_RECONCILIATION.md" in (
        _ROOT / "docs" / "PRODUCTION_RUNBOOK.md"
    ).read_text(encoding="utf-8")


def test_ledger_digest_and_restore_round_trip(tmp_path: Path) -> None:
    script = _drill()
    one = "a" * 64
    relative = "metadata/paper/dr-drill/broker_state.json"
    assert script.combine_ledger_digest([(relative, one)]) == one
    rows = [("b/broker_state.json", "b" * 64), ("a/broker_state.json", "a" * 64)]
    blob = f"a/broker_state.json {'a' * 64}\nb/broker_state.json {'b' * 64}".encode()
    assert (
        script.combine_ledger_digest([*rows, ("kill_switch", "c" * 64)])
        == hashlib.sha256(blob).hexdigest()
    )
    body = b'{"data_label":"SYNTHETIC","schema":"dr-drill-ledger","step":0}\n'
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    entries = [(relative, digest, len(body))]
    assert script.restored_digest(tmp_path, entries) == digest
    target.write_bytes(body + b"x")
    assert script.restored_digest(tmp_path, entries) == ""
    assert script.elapsed_seconds(10.0, 12.3454) == 2.345
    assert script.removable(Path("/tmp/dipcatcher-ops7-data/lab"))
    assert script.removable(Path("/tmp/dipcatcher-ops7-backup"))
    assert not script.removable(Path("/tmp"))
    assert not script.removable(Path("/tmp/other"))
    assert not script.removable(Path("/workspace/dipcatcher-ops7"))


def test_report_ok_needs_every_check_and_a_measured_rto() -> None:
    script = _drill()
    checks = [{"name": name, "ok": True} for name in script._REQUIRED]
    report = script._report(checks, "ab", 1.5, {"up": 1.0})
    assert report["ok"] is True
    assert report["rto_seconds"] == 1.5
    assert report["rto_includes"].startswith("destroy")
    assert script._report(checks[:-1], "ab", 1.5, {})["ok"] is False
    failed = [{**item, "ok": item["name"] != "ledger"} for item in checks]
    assert script._report(failed, "ab", 1.5, {})["ok"] is False
    assert script._report(checks, "ab", None, {})["ok"] is False


def test_drill_does_not_make_without_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _drill()
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCH_API_KEY", raising=False)

    def boom(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("make")

    monkeypatch.setattr(script, "_make", boom)
    monkeypatch.setattr(script, "_run", boom)
    assert script.main(["prod_dr_drill.py"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "required" in payload["error"]
    assert script.main(["prod_dr_drill.py", "--destroy"]) == 1


def test_manifest_rows_reject_a_foreign_schema(tmp_path: Path) -> None:
    script = _drill()
    path = tmp_path / "manifest.json"
    path.write_text('{"schema":"other","files":[]}\n', encoding="utf-8")
    assert script.manifest_rows(path) == []
    path.write_text(
        json.dumps(
            {
                "files": [{"path": "broker_state.json", "sha256": "ab", "size": 1}],
                "schema": "paper_state_backup.v1",
            }
        ),
        encoding="utf-8",
    )
    assert script.manifest_rows(path) == [("broker_state.json", "ab", 1)]


def test_firewall_ssh_and_upgrades() -> None:
    script = _preflight()
    assert script.firewall_ok(script.parse_ufw_ports(_UFW))
    assert script.parse_ufw_ports(_UFW.replace("deny (incoming)", "allow (incoming)")) is None
    assert script.firewall_ok(script.parse_iptables_ports(_IPTABLES))
    assert (
        script.parse_iptables_ports(_IPTABLES.replace("-P INPUT DROP", "-P INPUT ACCEPT")) is None
    )
    wide = _IPTABLES + "-A INPUT -p tcp -j ACCEPT\n"
    assert script.parse_iptables_ports(wide) is None
    extra = _UFW + "22/tcp                     ALLOW IN    Anywhere\n"
    assert not script.firewall_ok(script.parse_ufw_ports(extra))
    assert script.firewall_probes_ok(_IPTABLES, "")
    assert not script.firewall_probes_ok(_IPTABLES, _IPTABLES.replace("--dport 443", "--dport 22"))
    assert script.ssh_key_only(_SSHD)
    assert not script.ssh_key_only(_SSHD + "\nPasswordAuthentication yes\n")
    assert not script.ssh_key_only("PubkeyAuthentication yes\n")
    assert script.upgrades_ok(_APT, "enabled")
    assert not script.upgrades_ok('// APT::Periodic::Unattended-Upgrade "1";\n', "enabled")
    assert not script.upgrades_ok(_APT, "disabled")
    assert script.time_sync_ok("yes")
    assert not script.time_sync_ok("no")
    assert [item["name"] for item in script.hardening_warnings()] == list(script.HARDENING_NAMES)
    assert all(item["warning"] == "not root" for item in script.hardening_warnings())


def test_preflight_warns_when_not_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _preflight()
    _ready(monkeypatch, tmp_path)
    monkeypatch.setattr(script.os, "geteuid", lambda: 1000)

    def forbidden() -> dict[str, str]:
        raise AssertionError("read_hardening")

    monkeypatch.setattr(script, "read_hardening", forbidden)
    assert script.main(["prod_preflight.py"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert {item["name"] for item in payload["checks"]} == _BASE_CHECKS
    assert [item["warning"] for item in payload["warnings"]] == ["not root"] * 4
    blob = json.dumps(payload)
    assert _KEY not in blob
    assert _OTHER not in blob


def test_preflight_fails_closed_on_hardening_when_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _preflight()
    _ready(monkeypatch, tmp_path)
    monkeypatch.setattr(script.os, "geteuid", lambda: 0)
    monkeypatch.setattr(script, "read_hardening", _good_probes)
    assert script.main(["prod_preflight.py"]) == 0
    passed = json.loads(capsys.readouterr().out)
    assert passed["ok"] is True
    assert passed["warnings"] == []
    assert {item["name"] for item in passed["checks"]} > _BASE_CHECKS
    probes = _good_probes()
    probes["firewall"] = _UFW + "22/tcp                     ALLOW IN    Anywhere\n"
    monkeypatch.setattr(script, "read_hardening", lambda: probes)
    assert script.main(["prod_preflight.py"]) == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed["ok"] is False
    firewall = next(item for item in failed["checks"] if item["name"] == "firewall")
    assert firewall["ok"] is False
    assert _KEY not in json.dumps(failed)
