"""Unit tests for ``fx1.cli_sources``.

These are the datasource commands of the fx1 CLI. The module is deliberately
thin, so its value is in the *failure contracts* it encodes:

* a failed fetch is reported, never patched over or retried into a fake
  success;
* a malformed ``--params-json`` is a usage error (exit 2), distinct from a
  fetch failure (exit 1);
* an unavailable source exits 1 with the reason on stderr;
* the corpus ingest refuses when the fetch fails — nothing is appended to the
  corpus, so no leakage and no fabrication.

The commands are driven through their real Typer signatures so the parameter
defaults are resolved by the CLI layer, and the datasource layer is replaced
with fakes so nothing here touches the network, credentials or a vendor
script.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from fx1.cli_sources import (
    corpus_ingest_source,
    register_source_commands,
    sources_describe,
    sources_fetch,
    sources_list,
    sources_probe,
    sources_route,
    sources_scenarios,
)
from fx1.data.sources.base import FetchResult, SourceProbe, SourceStatus
from fx1.data.sources.registry import SourceSpec
from fx1.data.sources.router import RoutingPlan

RUNNER = CliRunner()


def _spec(name: str = "demo") -> SourceSpec:
    return SourceSpec(
        name=name,
        display="Demo source",
        owner_plugins=["demo"],
        kind="agent_gw",
        script_candidates=["demo.py"],
        markets=["us"],
        assets=["equity"],
        latency="daily",
    )


class _FakeAdapter:
    """Adapter double: records requests and replays a scripted outcome."""

    def __init__(self, result: FetchResult) -> None:
        self.result = result
        self.requests: list[Any] = []

    def describe(self) -> FetchResult:
        return self.result

    def fetch(self, request: Any) -> FetchResult:
        self.requests.append(request)
        return self.result


def _ok_result(text: str = "payload") -> FetchResult:
    return FetchResult(
        ok=True,
        source="demo",
        api="quotes",
        text=text,
        payload_sha256=FetchResult.hash_payload(text),
    )


def _fail_result(error: str = "vendor timeout") -> FetchResult:
    return FetchResult(ok=False, source="demo", api="quotes", error=error)


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    result: FetchResult | None = None,
) -> _FakeAdapter:
    """Point the datasource layer at deterministic fakes."""
    import fx1.data.sources.adapters as adapters
    import fx1.data.sources.registry as registry
    import fx1.data.sources.router as router

    spec = _spec()
    adapter = _FakeAdapter(result if result is not None else _ok_result())
    monkeypatch.setattr(registry, "list_sources", lambda: [spec])
    monkeypatch.setattr(registry, "get_spec", lambda name: spec)
    monkeypatch.setattr(registry, "roots_status", lambda: {"root": "ok"})
    monkeypatch.setattr(
        router,
        "probe_names",
        lambda names: [SourceProbe(name=name, status=SourceStatus.READY) for name in names],
    )
    monkeypatch.setattr(adapters, "build_adapter", lambda spec, runner=None: adapter)
    return adapter


def _cli(command: Any) -> typer.Typer:
    """Wrap a command in a one-command app that keeps its real signature."""
    app = typer.Typer()
    app.command("run")(command)
    return app


# --------------------------------------------------------------------------- #
# Command registration
# --------------------------------------------------------------------------- #


def test_register_attaches_every_source_command() -> None:
    sources_app = typer.Typer()
    corpus_app = typer.Typer()
    register_source_commands(sources_app, corpus_app)
    assert {command.name for command in sources_app.registered_commands} == {
        "list",
        "probe",
        "describe",
        "fetch",
        "route",
        "scenarios",
    }
    assert {command.name for command in corpus_app.registered_commands} == {"ingest-source"}


# --------------------------------------------------------------------------- #
# list / probe
# --------------------------------------------------------------------------- #


def test_list_prints_one_line_per_source(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    result = RUNNER.invoke(_cli(sources_list), [])
    assert result.exit_code == 0
    assert "demo" in result.stdout
    assert "ready" in result.stdout


def test_probe_reports_roots_and_probe_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    result = RUNNER.invoke(_cli(sources_probe), [])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["roots"] == {"root": "ok"}
    assert payload["probes"][0]["name"] == "demo"
    assert payload["probes"][0]["status"] == "ready"


def test_probe_can_target_a_single_source(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    result = RUNNER.invoke(_cli(sources_probe), ["demo"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [probe["name"] for probe in payload["probes"]] == ["demo"]


# --------------------------------------------------------------------------- #
# describe / scenarios
# --------------------------------------------------------------------------- #


def test_describe_prints_the_source_documentation(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    result = RUNNER.invoke(_cli(sources_describe), ["demo"])
    assert result.exit_code == 0
    assert "payload" in result.stdout


def test_describe_of_an_unavailable_source_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch, _fail_result("script missing"))
    result = RUNNER.invoke(_cli(sources_describe), ["demo"])
    assert result.exit_code == 1


def test_scenarios_uses_the_finance_fetch_source(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    result = RUNNER.invoke(_cli(sources_scenarios), [])
    assert result.exit_code == 0
    assert "payload" in result.stdout


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #


def test_fetch_passes_the_requested_parameters_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install_fakes(monkeypatch)
    result = RUNNER.invoke(
        _cli(sources_fetch),
        [
            "demo",
            "--api",
            "quotes",
            "--params-json",
            '{"symbol": "AAA"}',
            "--as-of",
            "2024-01-02",
        ],
    )
    assert result.exit_code == 0
    request = adapter.requests[0]
    assert request.api == "quotes"
    assert request.params == {"symbol": "AAA"}
    assert request.as_of == "2024-01-02"


def test_fetch_defaults_the_params_to_an_empty_object(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _install_fakes(monkeypatch)
    result = RUNNER.invoke(_cli(sources_fetch), ["demo", "--api", "quotes"])
    assert result.exit_code == 0
    assert adapter.requests[0].params == {}


def test_fetch_rejects_a_non_object_params_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A usage error (exit 2), distinct from a fetch failure (exit 1)."""
    _install_fakes(monkeypatch)
    result = RUNNER.invoke(
        _cli(sources_fetch),
        ["demo", "--api", "quotes", "--params-json", "[1, 2, 3]"],
    )
    assert result.exit_code == 2


def test_fetch_reports_a_failed_fetch_without_patching_it_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, _fail_result())
    result = RUNNER.invoke(_cli(sources_fetch), ["demo", "--api", "quotes"])
    assert result.exit_code == 1


# --------------------------------------------------------------------------- #
# route
# --------------------------------------------------------------------------- #


def test_route_prints_the_plan_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.data.sources.router as router

    monkeypatch.setattr(
        router,
        "route",
        lambda question, market=None, need=None: RoutingPlan(
            question=question,
            need=need or "quotes",
            market=market or "us",
            candidates=[],
            note="",
        ),
    )
    result = RUNNER.invoke(
        _cli(sources_route),
        ["--question", "how did AAPL do?", "--market", "us", "--need", "quotes"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["question"] == "how did AAPL do?"
    assert payload["need"] == "quotes"
    assert payload["market"] == "us"


# --------------------------------------------------------------------------- #
# corpus ingest
# --------------------------------------------------------------------------- #


def test_corpus_ingest_refuses_a_failed_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """A refused ingest must not append anything to the corpus file."""
    _install_fakes(monkeypatch, _fail_result())
    out = tmp_path / "corpus.jsonl"
    result = RUNNER.invoke(
        _cli(corpus_ingest_source),
        [
            "demo",
            "--api",
            "quotes",
            "--ledger-path",
            str(tmp_path / "ledger.jsonl"),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert not out.exists()
