"""fxi — model endpoints, orbs, wizard onboarding, REPL, and the Typer app.

Presence-only hygiene is tested the same way as fx1.doctor: raw keys must
never appear in listings, doctor output, or error paths.
"""

import io
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fx1.interactive import actions, orb, profiles, shell, wizard
from fx1.interactive.app import app as fxi_app
from fx1.serve.backends import MOONSHOT_API_URL, HostedK3Backend

RUNNER = CliRunner()
SECRET_FX1 = "sk-fx1-super-secret-value"
SECRET_LITE = "lite-super-secret-value"
ALT_BASE = "https://fx-lite.example.test/v1/chat/completions"


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    store_dir = tmp_path / "fx1-config"
    monkeypatch.setenv("FX1_CONFIG_DIR", str(store_dir))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    return store_dir


# ---------------------------------------------------------------------------
# profiles: model endpoints, mapping, presence-only reporting
# ---------------------------------------------------------------------------


def test_model_map_and_injected_env_names() -> None:
    assert profiles.MODELS == ("fx1", "fx1-lite")
    assert profiles.DEFAULT_MODEL == "fx1"
    assert profiles.ENV_API_KEY == "MOONSHOT_API_KEY"
    assert profiles.ENV_BASE_URL == "FX1_BASE_URL"


def test_set_endpoint_roundtrip_and_mode(isolated_store: Path) -> None:
    path = profiles.set_endpoint("fx1", SECRET_FX1)
    assert path == isolated_store / "credentials.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert profiles.resolve_endpoint("fx1") == (SECRET_FX1, MOONSHOT_API_URL)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["profiles"]["fx1"]["base_url"] == MOONSHOT_API_URL
    assert loaded["profiles"]["fx1"]["set_at"]


def test_set_endpoint_explicit_base_url() -> None:
    profiles.set_endpoint("fx1-lite", SECRET_LITE, ALT_BASE)
    assert profiles.resolve_endpoint("fx1-lite") == (SECRET_LITE, ALT_BASE)
    assert profiles.host_of(ALT_BASE) == "fx-lite.example.test"
    rows = {row["model"]: row for row in profiles.list_models()}
    assert rows["fx1-lite"]["host"] == "fx-lite.example.test"
    assert rows["fx1"]["status"] == "unset"


def test_set_endpoint_validation() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        profiles.set_endpoint("fx1-nope", "x")
    with pytest.raises(ValueError, match="empty API key"):
        profiles.set_endpoint("fx1", "   ")
    with pytest.raises(ValueError, match="invalid base URL"):
        profiles.set_endpoint("fx1", SECRET_FX1, "not-a-url")


def test_legacy_key_schema_read_compat(isolated_store: Path) -> None:
    store = isolated_store / "credentials.json"
    isolated_store.mkdir(parents=True)
    store.write_text(
        json.dumps({"profiles": {"fx1": {"key": SECRET_FX1, "env_var": "MOONSHOT_API_KEY"}}}),
        encoding="utf-8",
    )
    assert profiles.resolve_endpoint("fx1") == (SECRET_FX1, MOONSHOT_API_URL)
    rows = {row["model"]: row for row in profiles.list_models()}
    assert rows["fx1"]["status"] == "set"


def test_remove_endpoint() -> None:
    profiles.set_endpoint("fx1", SECRET_FX1)
    assert profiles.remove_endpoint("fx1") is True
    assert profiles.remove_endpoint("fx1") is False


def test_list_models_presence_only(isolated_store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profiles.set_endpoint("fx1", SECRET_FX1)
    monkeypatch.setenv("MOONSHOT_API_KEY", SECRET_LITE)
    rows = {row["model"]: row for row in profiles.list_models()}
    assert rows["fx1"]["status"] == "set"
    assert rows["fx1"]["fingerprint"] == "...alue"
    assert rows["fx1-lite"]["status"] == "env"
    assert rows["fx1-lite"]["fingerprint"] == "-"
    blob = json.dumps(rows)
    assert SECRET_FX1 not in blob
    assert SECRET_LITE not in blob


def test_resolve_endpoint_stored_wins_over_env(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles.set_endpoint("fx1", SECRET_FX1)
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-key-should-lose")
    assert profiles.resolve_endpoint("fx1") == (SECRET_FX1, MOONSHOT_API_URL)


def test_resolve_endpoint_env_fallback(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-key")
    assert profiles.resolve_endpoint("fx1") == ("env-key", MOONSHOT_API_URL)
    monkeypatch.setenv("FX1_BASE_URL", ALT_BASE)
    assert profiles.resolve_endpoint("fx1") == ("env-key", ALT_BASE)


def test_apply_profile_injects_endpoint() -> None:
    profiles.set_endpoint("fx1-lite", SECRET_LITE, ALT_BASE)
    target: dict[str, str] = {}
    names = profiles.apply_profile("fx1-lite", environ=target)
    assert names == ["MOONSHOT_API_KEY", "FX1_BASE_URL"]
    assert target == {"MOONSHOT_API_KEY": SECRET_LITE, "FX1_BASE_URL": ALT_BASE}


def test_apply_profile_fail_closed_names_model() -> None:
    with pytest.raises(RuntimeError, match="fx1-lite"):
        profiles.apply_profile("fx1-lite", environ={})


# ---------------------------------------------------------------------------
# orb: thinking-orbs terminal port
# ---------------------------------------------------------------------------


def test_orb_captions_cover_the_six_states() -> None:
    for state in orb.OrbState:
        assert orb.caption(state).endswith("…")
    assert orb.caption(orb.OrbState.LISTENING) == "Agent listening…"


def test_orb_render_frame_shape_and_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLORFGBG", raising=False)
    frame_plain = orb.render_frame(orb.OrbState.WORKING, t=1.0, color=False)
    assert len(frame_plain.split("\n")) == 4  # avatar preset
    assert "\x1b" not in frame_plain
    frame_color = orb.render_frame(orb.OrbState.WORKING, t=1.0, color=True)
    assert "\x1b[" in frame_color
    inline = orb.render_frame(orb.OrbState.LISTENING, t=0.0, preset="inline", color=False)
    assert len(inline.split("\n")) == 2


def test_orb_states_are_distinct_choreographies() -> None:
    frames = {
        state: orb.render_frame(state, t=1.25, color=False)
        for state in (orb.OrbState.LISTENING, orb.OrbState.WORKING, orb.OrbState.SEARCHING)
    }
    assert len(set(frames.values())) == 3


def test_orb_disabled_by_no_color_and_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    assert not orb.orbs_enabled(stream)
    monkeypatch.setenv("NO_COLOR", "1")

    class _Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert not orb.orbs_enabled(_Tty())
    monkeypatch.delenv("NO_COLOR")
    assert orb.orbs_enabled(_Tty())
    monkeypatch.setenv("FXI_ORBS", "off")
    assert not orb.orbs_enabled(_Tty())


def test_orb_animator_no_op_off_tty() -> None:
    stream = io.StringIO()
    with orb.OrbAnimator(orb.OrbState.SOLVING, stream=stream):
        pass
    assert stream.getvalue() == ""


def test_orb_banner_mentions_model_and_host() -> None:
    text = orb.banner("fx1", "api.moonshot.ai", color=False)
    assert "fx1" in text and "api.moonshot.ai" in text
    assert "╭" in text and "╰" in text


# ---------------------------------------------------------------------------
# hosted backend: base URL resolution (core, additive)
# ---------------------------------------------------------------------------


def test_hosted_backend_base_url_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    monkeypatch.setenv("FX1_BASE_URL", ALT_BASE)
    assert HostedK3Backend()._api_url == ALT_BASE  # noqa: SLF001 — asserting the resolved endpoint
    assert (
        HostedK3Backend(api_url="https://explicit.test/v1")._api_url == "https://explicit.test/v1"
    )  # noqa: SLF001
    monkeypatch.delenv("FX1_BASE_URL")
    assert HostedK3Backend()._api_url == MOONSHOT_API_URL  # noqa: SLF001


# ---------------------------------------------------------------------------
# actions: rendering, harness metadata, subprocess env injection, chat
# ---------------------------------------------------------------------------


def test_render_table_formats_rows() -> None:
    rows = [{"name": "doctor", "role": "verification"}, {"name": "ingest", "role": "data_engine"}]
    out = actions.render_table(rows)
    assert "name" in out and "role" in out
    assert "doctor" in out and "data_engine" in out


def test_harness_rows_and_get() -> None:
    rows = actions.harness_rows()
    names = {row["name"] for row in rows}
    assert {"doctor", "verify-research", "backtest"} <= names
    meta = actions.harness_get("doctor")
    assert meta["role"] == "verification"
    with pytest.raises(KeyError, match="not a registered harness command"):
        actions.harness_get("nope")


def test_run_tool_passes_endpoint_via_env_not_argv(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles.set_endpoint("fx1", SECRET_FX1, ALT_BASE)
    calls: list[dict[str, object]] = []

    def fake_run(argv: list[str], **kwargs: object) -> object:
        calls.append({"argv": list(argv), **kwargs})
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(actions.subprocess, "run", fake_run)
    monkeypatch.setattr(actions.shutil, "which", lambda name: f"/usr/bin/{name}")
    code = actions.run_tool("fx1", ["doctor"], "fx1")
    assert code == 0
    (call,) = calls
    argv = call["argv"]
    assert isinstance(argv, list)
    assert argv[:2] == ["/usr/bin/fx1", "doctor"]
    assert SECRET_FX1 not in " ".join(str(a) for a in argv)  # env only, never argv
    # The endpoint reaches the child via os.environ mutation (inherited), not env=.
    assert os.environ["MOONSHOT_API_KEY"] == SECRET_FX1
    assert os.environ["FX1_BASE_URL"] == ALT_BASE


def test_run_tool_rejects_unknown_tool() -> None:
    with pytest.raises(KeyError, match="unknown tool"):
        actions.run_tool("rm", ["-rf"], "fx1")


def test_run_eval_builds_fx1_eval_argv(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles.set_endpoint("fx1", SECRET_FX1)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> object:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(actions.subprocess, "run", fake_run)
    monkeypatch.setattr(actions.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert actions.run_eval(Path("out.json"), "fx1") == 0
    assert calls[0] == [
        "/usr/bin/fx1",
        "eval",
        "--backend",
        "hosted_k3",
        "--model",
        "fx1",
        "--out",
        "out.json",
    ]


class _FakeBackend:
    def __init__(
        self, api_key: str | None = None, model: str = "kimi-k3", api_url: str | None = None
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append([dict(m) for m in messages])
        return "fake-reply"


def test_chat_one_shot_uses_stored_endpoint() -> None:
    profiles.set_endpoint("fx1-lite", SECRET_LITE, ALT_BASE)
    outputs: list[str] = []
    made: list[_FakeBackend] = []

    def factory(**kwargs: object) -> _FakeBackend:
        backend = _FakeBackend(**kwargs)
        made.append(backend)
        return backend

    code = actions.chat(
        "fx1-lite",
        "hello",
        backend_factory=factory,
        output_fn=outputs.append,
        animated=False,
    )
    assert code == 0
    assert outputs == ["fake-reply"]
    assert made[0].api_key == SECRET_LITE
    assert made[0].model == "fx1-lite"  # fx1/fx1-lite are model names
    assert made[0].api_url == ALT_BASE
    assert made[0].calls == [[{"role": "user", "content": "hello"}]]


def test_chat_fail_closed_without_key() -> None:
    with pytest.raises(RuntimeError, match="fx1.*MOONSHOT_API_KEY"):
        actions.chat("fx1", "hi", backend_factory=lambda **kwargs: _FakeBackend())


def test_chat_interactive_loop() -> None:
    profiles.set_endpoint("fx1", SECRET_FX1)
    prompts = iter(["what is fx-1?", "exit"])
    outputs: list[str] = []
    made: list[_FakeBackend] = []

    def factory(**kwargs: object) -> _FakeBackend:
        backend = _FakeBackend(**kwargs)
        made.append(backend)
        return backend

    actions.chat(
        "fx1",
        backend_factory=factory,
        input_fn=lambda _: next(prompts),
        output_fn=outputs.append,
        animated=False,
    )
    turns = [o for o in outputs if o not in ("",)]
    assert turns[-1] == "fake-reply"
    assert len(made[0].calls) == 1  # the 'exit' line never reaches the backend


def test_doctor_presence_only(isolated_store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profiles.set_endpoint("fx1", SECRET_FX1, ALT_BASE)
    monkeypatch.setattr(actions, "_probe", lambda url, timeout=3.0: "reachable")
    status = actions.doctor("fx1")
    assert status["active_model"] == "fx1"
    assert status["api_host"] == "fx-lite.example.test"
    assert status["api_reachable"] == "reachable"
    assert status["models"][0]["fingerprint"] == "...alue"
    blob = json.dumps(status, default=str)
    assert SECRET_FX1 not in blob


# ---------------------------------------------------------------------------
# wizard onboarding
# ---------------------------------------------------------------------------


def _drive_wizard(
    answers: list[str],
    *,
    password: str = SECRET_FX1,
    prober_result: str = "reachable",
    force: bool = True,
) -> tuple[list[str], list[str]]:
    outputs: list[str] = []
    shells: list[str] = []
    answers_iter = iter(answers)

    def input_fn(prompt: str) -> str:
        outputs.append(prompt)
        try:
            return next(answers_iter)
        except StopIteration:
            raise EOFError from None

    wizard.enter(
        force=force,
        input_fn=input_fn,
        password_fn=lambda prompt: (outputs.append(prompt), password)[1],
        output_fn=outputs.append,
        animations=False,
        shell_factory=lambda model="fx1": shells.append(model),
        prober=lambda url: prober_result,
    )
    return outputs, shells


def test_wizard_full_onboarding_flow() -> None:
    # model prompt → default, base URL prompt → default, no second model.
    outputs, shells = _drive_wizard(["", "", "n"], password=SECRET_FX1)
    assert profiles.resolve_endpoint("fx1") == (SECRET_FX1, MOONSHOT_API_URL)
    assert shells == ["fx1"]
    assert any("reachable" in o for o in outputs)
    prompts = [o for o in outputs if o.startswith("›")]
    assert any("API key" in p for p in prompts)
    assert any("base URL" in p for p in prompts)
    blob = json.dumps(outputs)
    assert SECRET_FX1 not in blob  # the key never appears in output, only the prompts


def test_wizard_configures_chosen_model_with_base_url() -> None:
    _, shells = _drive_wizard(["fx1-lite", "", "n"], password=SECRET_LITE)
    assert profiles.resolve_endpoint("fx1-lite") == (SECRET_LITE, MOONSHOT_API_URL)
    assert shells == ["fx1-lite"]


def test_wizard_skips_when_configured_without_force() -> None:
    profiles.set_endpoint("fx1", SECRET_FX1)
    outputs: list[str] = []
    shells: list[str] = []
    wizard.enter(
        force=False,
        input_fn=lambda prompt: pytest.fail("must not prompt when configured"),
        password_fn=lambda prompt: pytest.fail("must not prompt when configured"),
        output_fn=outputs.append,
        animations=False,
        shell_factory=lambda model="fx1": shells.append(model),
    )
    assert shells == ["fx1"]
    assert outputs == []


def test_wizard_aborts_on_eof() -> None:
    outputs: list[str] = []
    shells: list[str] = []
    wizard.enter(
        force=True,
        input_fn=lambda prompt: (_ for _ in ()).throw(EOFError()),
        password_fn=lambda prompt: SECRET_FX1,
        output_fn=outputs.append,
        animations=False,
        shell_factory=lambda model="fx1": shells.append(model),
    )
    assert shells == []
    assert profiles.resolve_endpoint("fx1") is None


# ---------------------------------------------------------------------------
# REPL dispatch
# ---------------------------------------------------------------------------


def test_shell_keys_table(isolated_store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    profiles.set_endpoint("fx1", SECRET_FX1)
    sh = shell.FxiShell()
    sh.do_keys("")
    out = capsys.readouterr().out
    assert "fx1" in out and "api.moonshot.ai" in out and "...alue" in out
    assert SECRET_FX1 not in out


def test_shell_use_injects_session_env(isolated_store: Path) -> None:
    profiles.set_endpoint("fx1-lite", SECRET_LITE, ALT_BASE)
    sh = shell.FxiShell()
    old_key = os.environ.get("MOONSHOT_API_KEY")
    old_url = os.environ.get("FX1_BASE_URL")
    try:
        sh.do_use("fx1-lite")
        assert sh.model == "fx1-lite"
        assert os.environ["MOONSHOT_API_KEY"] == SECRET_LITE
        assert os.environ["FX1_BASE_URL"] == ALT_BASE
        assert "fx1-lite@fx-lite.example.test" in sh.prompt
    finally:
        for var, old in (("MOONSHOT_API_KEY", old_key), ("FX1_BASE_URL", old_url)):
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old


def test_shell_use_fail_closed(isolated_store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sh = shell.FxiShell()
    sh.do_use("fx1-lite")
    err = capsys.readouterr().err
    assert "error:" in err and "fxi setup" in err
    assert sh.model == "fx1"


def test_shell_model_and_exit(isolated_store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sh = shell.FxiShell(model="fx1-lite")
    sh.do_model("")
    assert "fx1-lite" in capsys.readouterr().out
    assert sh.do_exit("") is True
    assert sh.do_quit("") is True
    assert sh.do_EOF("") is True


def test_shell_unknown_command_hint(
    isolated_store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sh = shell.FxiShell()
    sh.default("bogus")
    err = capsys.readouterr().err
    assert "unknown command" in err and "help" in err


# ---------------------------------------------------------------------------
# fxi Typer app
# ---------------------------------------------------------------------------


def test_cli_version() -> None:
    result = RUNNER.invoke(fxi_app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.startswith("fxi ")


def test_cli_keys_set_value_and_list(isolated_store: Path) -> None:
    result = RUNNER.invoke(
        fxi_app, ["keys", "set", "fx1", "--value", SECRET_FX1, "--base-url", ALT_BASE]
    )
    assert result.exit_code == 0
    assert "mode 0600" in result.stdout
    result = RUNNER.invoke(fxi_app, ["keys", "list"])
    assert result.exit_code == 0
    assert "fx-lite.example.test" in result.stdout
    assert "...alue" in result.stdout
    assert SECRET_FX1 not in result.stdout


def test_cli_keys_set_unknown_model(isolated_store: Path) -> None:
    result = RUNNER.invoke(fxi_app, ["keys", "set", "nope", "--value", "x"])
    assert result.exit_code == 2
    assert "unknown model" in result.output


def test_cli_keys_remove(isolated_store: Path) -> None:
    profiles.set_endpoint("fx1", SECRET_FX1)
    result = RUNNER.invoke(fxi_app, ["keys", "remove", "fx1"])
    assert result.exit_code == 0
    assert profiles.resolve_endpoint("fx1") is None


def test_cli_doctor_json(isolated_store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(actions, "_probe", lambda url, timeout=3.0: "reachable")
    result = RUNNER.invoke(fxi_app, ["doctor"])
    assert result.exit_code == 0
    status = json.loads(result.stdout)
    assert status["fx1_version"]
    assert status["api_reachable"] == "reachable"
    assert "models" in status


def test_cli_bare_invocation_opens_shell(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.interactive import concierge

    calls: list[str] = []
    monkeypatch.setattr(concierge, "run_console", lambda model="fx1": calls.append(model))
    result = RUNNER.invoke(fxi_app, [])
    assert result.exit_code == 0
    assert calls == ["fx1"]


def test_cli_global_model_selects_shell_model(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.interactive import concierge

    calls: list[str] = []
    monkeypatch.setattr(concierge, "run_console", lambda model="fx1": calls.append(model))
    result = RUNNER.invoke(fxi_app, ["--model", "fx1-lite"])
    assert result.exit_code == 0
    assert calls == ["fx1-lite"]


def test_cli_eval_fail_closed_without_key(isolated_store: Path) -> None:
    result = RUNNER.invoke(fxi_app, ["eval"])
    assert result.exit_code == 2
    assert "MOONSHOT_API_KEY" in result.output


def test_cli_harness_list(isolated_store: Path) -> None:
    result = RUNNER.invoke(fxi_app, ["harness", "list"])
    assert result.exit_code == 0
    assert "verify-research" in result.stdout


def test_cli_bad_model_flag(isolated_store: Path) -> None:
    result = RUNNER.invoke(fxi_app, ["--model", "nope", "doctor"])
    assert result.exit_code == 2
    assert "unknown model" in result.output


# ---------------------------------------------------------------------------
# fx1 eval --model flag (core CLI, additive)
# ---------------------------------------------------------------------------


def test_fx1_eval_accepts_model_flag(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fx1.cli import app as fx1_cli_app
    from fx1.serve import backends as serve_backends

    seen: dict[str, object] = {}

    def fake_backend(**kwargs: object) -> _FakeBackend:
        seen.update(kwargs)
        return _FakeBackend(**kwargs)

    def fake_run_suite(complete: object, tasks: list[object]) -> dict[str, object]:
        return {
            "by_kind": {},
            "honesty_gate_passed": True,
            "eval_bank_sha256": "x",
            "n": len(tasks),
        }

    # Patch both bindings: `fx1.eval.run_suite` is the package re-export that
    # `from fx1.eval import run_suite` reads once any test has touched it; the
    # suite-module binding covers a lazy-export first touch.
    monkeypatch.setattr(serve_backends, "HostedK3Backend", fake_backend)
    monkeypatch.setattr("fx1.eval.run_suite", fake_run_suite)
    monkeypatch.setattr("fx1.eval.suite.run_suite", fake_run_suite)
    monkeypatch.setenv("MOONSHOT_API_KEY", SECRET_FX1)
    out = tmp_path / "eval.json"
    result = CliRunner().invoke(
        fx1_cli_app, ["eval", "--backend", "hosted_k3", "--model", "fx1", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert seen["model"] == "fx1"
    assert json.loads(out.read_text(encoding="utf-8"))["honesty_gate_passed"] is True


# ---------------------------------------------------------------------------
# bare `dipcatcher` startup hook
# ---------------------------------------------------------------------------


def test_bare_dipcatcher_shows_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare ``dipcatcher`` prints help and exits 0.

    The fx-1 interactive front door is the ``fxi`` console script; the harness
    CLI deliberately does not import fx1 (see ``src/quant_fund/cli/_app.py``).
    """
    from quant_fund.cli.main import app as quant_app

    calls: list[bool] = []
    monkeypatch.setattr(wizard, "enter", lambda **kwargs: calls.append(True))
    result = CliRunner().invoke(quant_app, [])
    assert result.exit_code == 0
    assert "Dipcatcher" in result.output
    assert not calls


def test_dipcatcher_subcommand_skips_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.cli.main import app as quant_app

    monkeypatch.setattr(
        wizard, "enter", lambda **kwargs: pytest.fail("wizard must not run with a subcommand")
    )
    result = CliRunner().invoke(quant_app, ["--help"])
    assert result.exit_code == 0
    assert "Dipcatcher" in result.output
