"""Native ownership and environment regressions for the upload audit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import fx1.serve.uploads_audit as audit


def _stub_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in vars(audit):
        if name.startswith("_probe_"):
            monkeypatch.setattr(audit, name, lambda *a, **k: {})


@pytest.mark.parametrize("fail", [False, True])
def test_audit_owns_clients_and_workers(monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    _stub_probes(monkeypatch)
    clients: list[Any] = []
    pools: list[Any] = []
    original = audit._client

    def track(**kw: Any) -> Any:
        client = original(**kw)
        clients.append(client)
        pool = client.app.state.jobs_executor
        pools.append(pool)
        assert pool.submit(lambda: "worker-started").result(timeout=5) == "worker-started"
        return client

    def probe(*a: Any) -> dict[str, bool]:
        if fail:
            raise RuntimeError("probe failure")
        return {"owned": True}

    monkeypatch.setattr(audit, "_client", track)
    monkeypatch.setattr(audit, "_probe_happy_path", probe)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="probe failure"):
                audit.uploads_audit()
        else:
            assert audit.uploads_audit() == {"owned": True}
        assert len(clients) == 3
        assert all(client.is_closed for client in clients)
        for pool in pools:
            with pytest.raises(RuntimeError, match="shutdown"):
                pool.submit(lambda: None)
    finally:
        for client in clients:
            client.close()
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("fail", [False, True])
def test_environment_and_private_state_restored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: bool
) -> None:
    import fx1.serve.api as api

    _stub_probes(monkeypatch)
    ambient = tmp_path / "ambient"
    monkeypatch.setenv("FX1_API_STATE_DIR", str(ambient / "state"))
    monkeypatch.setenv("FX1_FT_DIR", str(ambient / "ft"))
    monkeypatch.setenv("FX1_API_RECEIPTS_DIR", str(ambient / "receipts"))
    monkeypatch.setenv("FX1_FUTURE_CONFIG", "restore-me")
    monkeypatch.setenv("MOONSHOT_API_KEY", "synthetic-ambient")
    saved = {k: v for k, v in os.environ.items() if k.startswith("FX1_") or k == "MOONSHOT_API_KEY"}
    paths: list[Path] = []
    original = api.create_app

    def create(**kw: Any) -> Any:
        assert not any(k.startswith("FX1_") or k == "MOONSHOT_API_KEY" for k in os.environ)
        paths.extend(Path(kw[k]) for k in ("state_dir", "ft_dir", "receipts_dir"))
        assert all(ambient not in path.parents for path in paths)
        return original(**kw)

    def probe(*a: Any) -> dict[str, bool]:
        os.environ["FX1_NEW_PROBE_CONFIG"] = "temporary"
        if fail:
            raise RuntimeError("probe failure")
        return {"isolated": True}

    monkeypatch.setattr(api, "create_app", create)
    monkeypatch.setattr(audit, "_probe_happy_path", probe)
    if fail:
        with pytest.raises(RuntimeError, match="probe failure"):
            audit.uploads_audit()
    else:
        assert audit.uploads_audit() == {"isolated": True}
    assert len(paths) == 9
    assert not ambient.exists()
    assert all(not path.exists() for path in paths)
    assert {
        k: v for k, v in os.environ.items() if k.startswith("FX1_") or k == "MOONSHOT_API_KEY"
    } == saved


def test_thread_worker_exceptions_reach_audit_caller() -> None:
    def fail(index: int) -> None:
        if index == 1:
            raise RuntimeError("worker failure")

    with pytest.raises(RuntimeError, match="worker failure"):
        audit._run_threads(fail, n=2)


def test_durability_probe_closes_each_app_before_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[Any] = []
    original = audit._client

    def track(**kw: Any) -> Any:
        assert all(client.is_closed for client in clients)
        for prior in clients:
            with pytest.raises(RuntimeError, match="shutdown"):
                prior.app.state.jobs_executor.submit(lambda: None)
        client = original(**kw)
        clients.append(client)
        return client

    monkeypatch.setattr(audit, "_client", track)
    with audit._audit_context():
        out = audit._probe_durability(audit._temporary_directory())
        assert len(out) == 6
        assert all(value is True for value in out.values())
        assert len(clients) == 4
        assert all(client.is_closed for client in clients)
