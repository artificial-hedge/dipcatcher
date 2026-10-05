"""Regression checks for observability audit evidence and isolation."""

from pathlib import Path

import pytest

from fx1.serve import api
from fx1.serve import observe_audit as audit


def test_empty_audit_does_not_mint_a_passing_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "observe_audit", lambda: {})
    result = audit.observe_audit_bench()
    assert result["claim"]["ok"] is False
    assert result["claim"]["defects"] == ["no_probes"]
    assert "no_probes" in result["interpretation"]


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["# HELP item demo", "# TYPE item gauge", "item 1"], True),
        (["item 1", "# HELP item demo", "# TYPE item gauge"], False),
        (["# HELP item demo", "item 1", "# TYPE item gauge"], False),
        (["# TYPE item gauge", "# HELP item demo", "item 1"], False),
    ],
)
def test_prometheus_declarations_precede_samples(lines: list[str], expected: bool) -> None:
    parsed = audit._Prom("\n".join(lines))
    assert parsed.ordered() is expected


@pytest.mark.parametrize("probe", ["_probe_dev_and_remote", "_probe_error_envelope"])
def test_probe_apps_ignore_ambient_state_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, probe: str
) -> None:
    ambient = tmp_path / "ambient-state"
    monkeypatch.setenv("FX1_API_STATE_DIR", str(ambient))
    monkeypatch.delenv("FX1_API_KEY", raising=False)
    with audit._audit_resources():
        getattr(audit, probe)()
    assert not ambient.exists(), "audit app initialization touched ambient persistent state"


def test_drain_probe_detects_missing_blocking_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api._Metrics, "wait_idle", lambda self, timeout_s: True)
    with audit._audit_resources():
        result = audit._probe_drain()
    assert result["drain_wait_blocks_until_idle"] is False


@pytest.mark.parametrize("fail", [False, True])
def test_clients_and_workers_close_when_audit_finishes(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    clients = []
    executors = []

    def probe() -> dict[str, bool]:
        client, _ = audit._client()
        clients.append(client)
        executor = client.app.state.jobs_executor
        executors.append(executor)
        executor.submit(lambda: None).result(timeout=2)
        if fail:
            raise RuntimeError("forced probe failure")
        return {"cleanup_case": True}

    for name in vars(audit):
        if name.startswith("_probe_"):
            monkeypatch.setattr(audit, name, lambda: {})
    monkeypatch.setattr(audit, "_probe_liveness", probe)
    monkeypatch.setenv("FX1_API_KEY", "ambient-test-key")
    if fail:
        with pytest.raises(RuntimeError, match="forced probe failure"):
            audit.observe_audit()
    else:
        assert audit.observe_audit() == {"cleanup_case": True}
    assert clients and all(client.is_closed for client in clients)
    for executor in executors:
        with pytest.raises(RuntimeError, match="shutdown"):
            executor.submit(lambda: None)
        assert all(not worker.is_alive() for worker in executor._threads)
    import os

    assert os.environ["FX1_API_KEY"] == "ambient-test-key"
