"""RT-2: end-to-end secret leakage across the dipcatcher harness.

Planted values use the marker ``rt2leak``. The credentials store is the only
file allowed to contain that marker. Everything else an operator can read,
export, zip, or hand to a child must not.
"""

from __future__ import annotations

import json
import sys
import traceback
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from fx1.harness import Harness, OwnedRunner
from fx1.interactive.approvals import ApprovalDecision, ApprovalGate, ApprovalRequest
from fx1.interactive.concierge import BackgroundJob, Concierge
from fx1.interactive.headless import ExecRequest, run_exec
from fx1.interactive.model_client import ModelClient, ModelError, RetryPolicy
from fx1.interactive.procs import ProcessRegistry, ProcLimits, run_owned, scrub_env
from fx1.interactive.runctx import new_session_id
from fx1.interactive.sessions import SessionStore
from fx1.interactive.telemetry import Telemetry
from tests.fx1.stub_model_server import Reply, StubModelServer

MARK = "rt2leak"
FX1_STORED = "rt2leakFx1StoredKey9f3c1a7e"
LITE_STORED = "rt2leakLiteStoredKey8e2b0d6c"
FX1_ENV = "rt2leakFx1EnvKey17ab44c0"
LITE_ENV = "rt2leakLiteEnvKey28bc55d1"
SHARED = "rt2leakSharedMoonshotKey39cd66e2"
URL_PW = "rt2leakUrlUserinfo6c0f99b5"
PROXY_PW = "rt2leakProxyUserinfo7d1a0ac6"
ECHO = "rt2leakHostileEcho8e2b1bd7"
SK = "sk-rt2leakShapeKey9f3c2ce8aa"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJrt2leakXX.SflKxwRJSMeKKF2QT4"
BEARER = "rt2leakBearerToken0a3c3df9bb"
ZW = "\u200b"

_SECRET_ENV = (
    "MOONSHOT_API_KEY",
    "FX1_BASE_URL",
    "MOONSHOT_API_URL",
    "DIPCATCHER_FX1_API_KEY",
    "DIPCATCHER_FX1_BASE_URL",
    "DIPCATCHER_FX1_LITE_API_KEY",
    "DIPCATCHER_FX1_LITE_BASE_URL",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "QUANT_LAB_NOTE",
)

_ERROR_CASES = (
    (401, "invalid_api_key", "auth"),
    (403, "permission_denied", "permission"),
    (404, "model_not_found", "unknown"),
    (404, "route_missing", "route"),
    (429, "rate_limit_exceeded", "rate"),
    (402, "insufficient_quota", "quota"),
    (400, "bad_request", "bad"),
    (413, "context_length_exceeded", "context"),
    (500, "internal_error", "server"),
    (502, "bad_gateway", "gateway"),
)


@pytest.fixture()
def root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg"
    sessions = tmp_path / "sessions"
    work = tmp_path / "work"
    for path in (cfg, sessions, work):
        path.mkdir()
    monkeypatch.setenv("FX1_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("FX1_SESSION_DIR", str(sessions))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.chdir(work)
    for name in _SECRET_ENV:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


class _Running:
    def __init__(self, keys: Mapping[str, str]) -> None:
        self._server = StubModelServer(dict(keys))

    def __enter__(self) -> StubModelServer:
        self._server.start()
        return self._server

    def __exit__(self, *_exc: object) -> None:
        self._server.close()


def _client(**kwargs: Any) -> ModelClient:
    timeout_s = kwargs.pop("timeout_s", 2.0)
    return ModelClient(
        timeout_s=timeout_s,
        retry=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0),
        sleep=lambda _delay: None,
        **kwargs,
    )


def _deny(request: ApprovalRequest) -> ApprovalDecision:
    del request
    return ApprovalDecision.DENY


def _store_both(url: str) -> None:
    from fx1.interactive.profiles import set_endpoint

    set_endpoint("fx1", FX1_STORED, url)
    set_endpoint("fx1-lite", LITE_STORED, url)


def _concierge(root: Path, out: list[str], **kwargs: Any) -> Concierge:
    tel = Telemetry(session_id=new_session_id(), directory=root / "cfg" / "logs")
    harness = kwargs.pop("harness", None)
    return Concierge(
        store=kwargs.pop("store", None),
        researcher=kwargs.pop("researcher", None),
        harness=harness,
        approvals=ApprovalGate(decide_fn=_deny),
        backend_factory=kwargs.pop("backend_factory", _client),
        output_fn=out.append,
        animated=False,
        input_fn=lambda _prompt: "",
        session_store=kwargs.pop("session_store", SessionStore()),
        telemetry_emit=tel.emit,
        telemetry=tel,
        **kwargs,
    )


def _files(root: Path, *, skip_credentials: bool = True) -> str:
    parts: list[str] = []
    if not root.exists():
        return ""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if skip_credentials and path.name == "credentials.json":
            continue
        parts.append(_file_text(path))
    return "\n".join(parts)


def _file_text(path: Path) -> str:
    if path.suffix == ".zip":
        return _zip_text(path)
    return path.read_bytes().decode("utf-8", errors="replace")


def _zip_text(path: Path) -> str:
    chunks: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            chunks.append(name)
            chunks.append(archive.read(name).decode("utf-8", errors="replace"))
    return "\n".join(chunks)


def _forbid(*blobs: str) -> None:
    blob = "\n".join(blobs)
    index = blob.find(MARK)
    if index < 0:
        return
    window = blob[max(0, index - 90) : index + 90]
    raise AssertionError(f"secret marker leaked near {window!r}")


def _surface(concierge: Concierge, out: list[str], err: str, root: Path) -> None:
    ident = concierge.last_identity
    _forbid(
        "\n".join(out),
        err,
        concierge.last_answer,
        json.dumps(concierge.messages(), default=str),
        "" if ident is None else repr(ident),
        "" if ident is None else str(ident),
        _files(root),
    )


def _captured(capsys: pytest.CaptureFixture[str]) -> str:
    got = capsys.readouterr()
    return f"{got.out}\n{got.err}"


def _echo_body(secret: str) -> str:
    return f"note {secret} pinball 0.2"


def _tool(name: str, arguments: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_stored_keys_for_both_models_stay_out_of_chat(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: list[str] = []
    with _Running({"fx1": FX1_STORED, "fx1-lite": LITE_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.ok(_echo_body(FX1_STORED), served_model="fx1"))
        server.queue(Reply.ok(_echo_body(LITE_STORED), served_model="fx1-lite"))
        concierge = _concierge(root, out)
        concierge.handle_line(f"keep {FX1_STORED} private")
        concierge.handle_line("/use fx1-lite")
        concierge.handle_line(f"and also {LITE_STORED}")
        concierge.handle_line("/models")
        _surface(concierge, out, _captured(capsys), root)


def test_per_model_env_keys_are_not_echoed(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DIPCATCHER_FX1_API_KEY", FX1_ENV)
    monkeypatch.setenv("DIPCATCHER_FX1_LITE_API_KEY", LITE_ENV)
    out: list[str] = []
    with _Running({"fx1": FX1_ENV, "fx1-lite": LITE_ENV}) as server:
        monkeypatch.setenv("DIPCATCHER_FX1_BASE_URL", server.url)
        monkeypatch.setenv("DIPCATCHER_FX1_LITE_BASE_URL", server.url)
        server.queue(Reply.ok(_echo_body(FX1_ENV), served_model="fx1"))
        server.queue(Reply.ok(_echo_body(LITE_ENV), served_model="fx1-lite"))
        concierge = _concierge(root, out)
        concierge.handle_line("hello fx1")
        concierge.handle_line("/use fx1-lite")
        concierge.handle_line("hello lite")
        _surface(concierge, out, _captured(capsys), root)


def test_shared_moonshot_key_is_not_echoed(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out: list[str] = []
    with _Running({"fx1": SHARED}) as server:
        monkeypatch.setenv("MOONSHOT_API_KEY", SHARED)
        monkeypatch.setenv("FX1_BASE_URL", server.url)
        server.queue(Reply.ok(_echo_body(SHARED), served_model="fx1"))
        concierge = _concierge(root, out)
        concierge.handle_line(f"do not repeat {SHARED}")
        _surface(concierge, out, _captured(capsys), root)


@pytest.mark.parametrize(("status", "code", "label"), _ERROR_CASES)
def test_error_status_does_not_leak(
    root: Path,
    capsys: pytest.CaptureFixture[str],
    status: int,
    code: str,
    label: str,
) -> None:
    del label
    message = f"{code} refused {FX1_STORED} {SK} {JWT} Bearer {BEARER}"
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(
            Reply.error(
                status,
                headers={"x-request-id": FX1_STORED, "www-authenticate": f"Bearer {BEARER}"},
                body={"error": {"message": message, "type": "server_error", "code": code}},
            )
        )
        concierge = _concierge(root, out)
        concierge.handle_line("hello")
        err = _captured(capsys)
        assert "Traceback" not in err
        _surface(concierge, out, err, root)


def test_non_json_error_body_and_raw_header_do_not_leak(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = f"plain failure {FX1_STORED} https://a:{URL_PW}@h.example/v Bearer {BEARER}".encode()
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(
            Reply(
                status=500,
                headers={"x-request-id": SK, "retry-after": FX1_STORED},
                body=raw,
            )
        )
        concierge = _concierge(root, out)
        concierge.handle_line("hello")
        _surface(concierge, out, _captured(capsys), root)


def test_malformed_success_body_does_not_leak(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.malformed(f"<<<{FX1_STORED} {JWT}>>>".encode()))
        concierge = _concierge(root, out)
        concierge.handle_line("hello")
        _surface(concierge, out, _captured(capsys), root)


def test_hostile_success_identity_content_and_models_table(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = f"answer {FX1_STORED} {SK} {JWT} Bearer {BEARER} https://a:{URL_PW}@h.example/v"
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(
            Reply.ok(
                content,
                served_model=FX1_STORED,
                system_fingerprint=JWT,
                headers={"x-request-id": FX1_STORED, "x-echo-secret": ECHO},
            )
        )
        concierge = _concierge(root, out)
        concierge.handle_line("hello")
        concierge.handle_line("/models")
        _surface(concierge, out, _captured(capsys), root)


def test_zero_width_split_of_a_registered_key_is_removed(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    split = FX1_STORED[:8] + ZW + FX1_STORED[8:]
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.ok(f"echo {split}", served_model="fx1"))
        concierge = _concierge(root, out)
        concierge.handle_line(split)
        _surface(concierge, out, _captured(capsys), root)


def test_direct_client_errors_keep_secrets_out_of_repr_and_traceback(root: Path) -> None:
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        client = _client(api_key=FX1_STORED, api_url=server.url, model="fx1")
        server.queue(
            Reply.error(
                401,
                body={"error": {"message": FX1_STORED, "type": "auth", "code": "invalid_api_key"}},
            )
        )
        with pytest.raises(ModelError) as caught:
            client.complete([{"role": "user", "content": "ping"}])
        text = "".join(traceback.format_exception(caught.value))
        _forbid(text, repr(caught.value), str(caught.value), repr(client), _files(root))


def test_deadline_and_cancel_do_not_leak(root: Path) -> None:
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        client = _client(api_key=FX1_STORED, api_url=server.url, model="fx1")
        import threading

        cancel = threading.Event()
        cancel.set()
        with pytest.raises(ModelError) as cancelled:
            client.complete([{"role": "user", "content": FX1_STORED}], cancel_event=cancel)
        with pytest.raises(ModelError) as deadline:
            client.complete(
                [{"role": "user", "content": FX1_STORED}],
                deadline=client._monotonic(),  # noqa: SLF001
            )
        _forbid(
            str(cancelled.value),
            repr(cancelled.value),
            str(deadline.value),
            repr(deadline.value),
            _files(root),
        )


def test_network_and_tls_failures_do_not_leak(root: Path) -> None:
    from fx1.interactive.profiles import set_endpoint

    set_endpoint("fx1", FX1_STORED, "https://127.0.0.1:9/v1/chat/completions")
    client = _client(
        api_key=FX1_STORED,
        api_url="https://127.0.0.1:9/v1/chat/completions",
        model="fx1",
        timeout_s=1.0,
    )
    with pytest.raises(ModelError) as tls_error:
        client.complete([{"role": "user", "content": "ping"}])
    refused = _client(
        api_key=FX1_STORED,
        api_url="http://127.0.0.1:1/v1/chat/completions",
        model="fx1",
        timeout_s=1.0,
    )
    with pytest.raises(ModelError) as net_error:
        refused.complete([{"role": "user", "content": "ping"}])
    _forbid(str(tls_error.value), str(net_error.value), repr(client), _files(root))


def test_typed_shapes_in_chat_are_scrubbed(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    line = f"see {FX1_STORED} {SK} {JWT} Bearer {BEARER} https://a:{URL_PW}@h.example/x"
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.ok(line, served_model="fx1"))
        concierge = _concierge(root, out)
        concierge.handle_line(line)
        _surface(concierge, out, _captured(capsys), root)


def test_flash_add_drops_pasted_keys_urls_and_shapes(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fx1.flash.store import FlashStore

    out: list[str] = []
    store = FlashStore(root / "cfg" / "flash" / "entries.jsonl")
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        concierge = _concierge(root, out, store=store)
        concierge.handle_line(
            f"/flash add {FX1_STORED} {JWT} Bearer {BEARER} "
            f"--source https://a:{URL_PW}@h.example/x --task {SK} --uncertainty low"
        )
        concierge.handle_line("/flash list")
        shown = store.all()
        assert shown
        concierge.handle_line(f"/flash show {shown[0].id}")
        concierge.handle_line(f"/flash search {FX1_STORED}")
        _surface(concierge, out, _captured(capsys), root)


def test_flash_correct_does_not_write_a_pasted_key(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fx1.flash.store import FlashStore

    out: list[str] = []
    store = FlashStore(root / "cfg" / "flash" / "entries.jsonl")
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        concierge = _concierge(root, out, store=store)
        concierge.handle_line("/flash add ordinary pinball note")
        entry_id = store.all()[0].id
        concierge.handle_line(
            f'/flash correct {entry_id} --text "{FX1_STORED} {JWT}" '
            f'--task "https://a:{URL_PW}@h.example/x"'
        )
        concierge.handle_line(f"/flash show {entry_id}")
        _surface(concierge, out, _captured(capsys), root)


def test_history_file_skips_typed_secrets(root: Path) -> None:
    from fx1.interactive import console

    out: list[str] = []
    split = FX1_STORED[:6] + ZW + FX1_STORED[6:]
    queued = ["pinball check", FX1_STORED, split, f"Bearer {BEARER} {JWT} {SK}", "/exit"]

    def read(_prompt: str) -> str:
        if not queued:
            raise EOFError
        return queued.pop(0)

    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.ok("one", served_model="fx1"))
        server.queue(Reply.ok("two", served_model="fx1"))
        server.queue(Reply.ok("three", served_model="fx1"))
        server.queue(Reply.ok("four", served_model="fx1"))
        concierge = _concierge(root, out)
        console.run_console(
            engine=True,
            input_fn=read,
            output_fn=out.append,
            concierge_factory=lambda: concierge,
            history_path=root / "cfg" / "history",
        )
        history = (root / "cfg" / "history").read_text(encoding="utf-8")
        assert "pinball check" in history
        _forbid("\n".join(out), history, _files(root))


def test_exec_plain_and_json_and_last_message_file(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    last = root / "work" / "last.txt"
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.ok(_echo_body(FX1_STORED) + f" {JWT}", served_model=FX1_STORED))
        plain = run_exec(
            ExecRequest(prompt=f"hide {FX1_STORED}", model="fx1", output_path=last),
            backend_factory=_client,
            install_signals=False,
            session_store=SessionStore(),
        )
        server.queue(Reply.ok(_echo_body(LITE_STORED), served_model="fx1-lite"))
        document = run_exec(
            ExecRequest(prompt=f"hide {LITE_STORED} {SK}", model="fx1-lite"),
            backend_factory=_client,
            install_signals=False,
            session_store=SessionStore(),
        )
        _forbid(
            plain.answer,
            "\n".join(plain.errors),
            "\n".join(plain.warnings),
            json.dumps(document.as_json()),
            last.read_text(encoding="utf-8") if last.exists() else "",
            _captured(capsys),
            _files(root),
        )


def test_exec_cli_stderr_and_json(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from dipcatcher_cli.exec_cmd import main as exec_main

    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.error(500, body={"error": {"message": FX1_STORED, "code": "error"}}))
        with pytest.raises(SystemExit):
            exec_main(["--json", "--model", "fx1", f"please leak {FX1_STORED}"])
        _forbid(_captured(capsys), _files(root))


def test_models_list_and_live_check_omit_keys(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fx1.interactive.models_check import check_models, format_check_report, format_model_list

    with _Running({"fx1": FX1_STORED, "fx1-lite": LITE_STORED}) as server:
        _store_both(server.url)
        server.queue(
            Reply.ok(
                "pong",
                served_model=FX1_STORED,
                system_fingerprint=JWT,
                headers={"x-request-id": SK},
            )
        )
        server.queue(Reply.ok("pong", served_model=LITE_STORED, system_fingerprint=SK))
        listed = format_model_list(
            *__import__(
                "fx1.interactive.models_check", fromlist=["presence_report"]
            ).presence_report()
        )
        report = check_models(("fx1", "fx1-lite"), live=True, timeout_s=2.0)
        _forbid(
            listed,
            format_check_report(report),
            json.dumps(report.as_json()),
            _captured(capsys),
            _files(root),
        )


def test_models_cli_writes_a_clean_record(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from dipcatcher_cli.models_cmd import main as models_main

    record = root / "work" / "models.json"
    with _Running({"fx1": FX1_STORED, "fx1-lite": LITE_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.ok("pong", served_model=FX1_STORED))
        server.queue(Reply.ok("pong", served_model="fx1-lite"))
        with pytest.raises(SystemExit):
            models_main(["check", "--live", "--json", "--timeout-s", "2", "--out", str(record)])
        body = record.read_text(encoding="utf-8") if record.exists() else ""
        _forbid(_captured(capsys), body, _files(root))


def test_sessions_export_slash_and_cli(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from dipcatcher_cli.sessions_cmd import sessions_main

    out: list[str] = []
    exported = root / "work" / "session.md"
    cli_out = root / "work" / "session.json"
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.ok(_echo_body(FX1_STORED) + f" {JWT}", served_model="fx1"))
        concierge = _concierge(root, out)
        concierge.handle_line(f"remember not to store {FX1_STORED}")
        concierge.handle_line(f"/export md {exported}")
        sessions_main(["export", "last", "--format", "json", "--out", str(cli_out)])
        sessions_main(["show", "last", "--messages"])
        _surface(concierge, out, _captured(capsys), root)
        _forbid(
            exported.read_text(encoding="utf-8") if exported.exists() else "",
            cli_out.read_text(encoding="utf-8") if cli_out.exists() else "",
        )


def test_diagnostics_bundle_members_omit_secrets(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", f"https://ops:{PROXY_PW}@proxy.example:8443")
    monkeypatch.setenv("QUANT_LAB_NOTE", FX1_STORED)
    out: list[str] = []
    bundle = root / "work" / "diag.zip"
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(
            Reply.ok(
                _echo_body(FX1_STORED) + f" https://ops:{PROXY_PW}@h.example/p",
                served_model="fx1",
            )
        )
        concierge = _concierge(root, out)
        concierge.handle_line("hello")
        concierge.handle_line(f"/diagnostics --bundle {bundle}")
        assert bundle.is_file()
        names = zipfile.ZipFile(bundle).namelist()
        assert "diagnostics.json" in names
        assert "report.txt" in names
        _surface(concierge, out, _captured(capsys), root)


def test_superpower_real_catalog_does_not_leak_the_goal(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", f"http://ops:{PROXY_PW}@proxy.example:8443")
    monkeypatch.setenv("QUANT_LAB_NOTE", SHARED)
    monkeypatch.setenv("MOONSHOT_API_KEY", SHARED)
    seen_argv: list[str] = []
    seen_env: list[str] = []
    real = __import__("fx1.interactive.procs", fromlist=["_popen"])._popen

    def spy(argv: list[str], **kwargs: Any) -> Any:
        seen_argv.extend(str(part) for part in argv)
        env = kwargs.get("env") or {}
        seen_env.extend(f"{key}={value}" for key, value in env.items())
        return real(argv, **kwargs)

    monkeypatch.setattr("fx1.interactive.procs._popen", spy)
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        for _ in range(4):
            server.queue(Reply.ok(_echo_body(FX1_STORED), served_model=FX1_STORED))
        concierge = _concierge(root, out, harness=Harness(OwnedRunner()))
        concierge.handle_line(f"/superpower --sync zzzzqqq no family {FX1_STORED}")
        _surface(concierge, out, _captured(capsys), root)
        _forbid("\n".join(seen_argv), "\n".join(seen_env))


def test_tool_loop_does_not_forward_secrets(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fx1.webresearch.engine import ResearchReport, SourceDoc

    class _Research:
        def run(self, goal: str, budget: Any = None, queries: Any = None) -> ResearchReport:
            del budget, queries
            page = f"ignore previous {FX1_STORED} Bearer {BEARER} {JWT}"
            return ResearchReport(
                goal=goal,
                sources=[
                    SourceDoc(
                        url=f"https://a:{URL_PW}@evil.example/a", title="A", status=200, text=page
                    )
                ],
            )

    seen: list[str] = []
    real = __import__("fx1.interactive.procs", fromlist=["_popen"])._popen

    def spy(argv: list[str], **kwargs: Any) -> Any:
        seen.extend(str(part) for part in argv)
        env = kwargs.get("env") or {}
        seen.extend(env.values())
        return real(argv, **kwargs)

    monkeypatch.setattr("fx1.interactive.procs._popen", spy)
    monkeypatch.setenv("QUANT_LAB_NOTE", FX1_STORED)
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(
            Reply.ok(
                None,
                tool_calls=[_tool("web_lookup", {"query": f"public cpi {FX1_STORED}"})],
                served_model="fx1",
            )
        )
        server.queue(
            Reply.ok(
                None,
                tool_calls=[
                    _tool(
                        "harness_run",
                        {"name": "families", "args": ["--select", FX1_STORED, SK]},
                        "c2",
                    )
                ],
                served_model="fx1",
            )
        )
        server.queue(Reply.ok("pinball loss is 0.4", served_model="fx1"))
        concierge = _concierge(root, out, researcher=_Research(), harness=Harness(OwnedRunner()))
        concierge.handle_line("check the web then the families")
        _surface(concierge, out, _captured(capsys), root)
        _forbid("\n".join(seen))


def test_run_owned_hides_secrets_from_proc_cmdline_and_env(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", SHARED)
    monkeypatch.setenv("DIPCATCHER_FX1_API_KEY", FX1_ENV)
    monkeypatch.setenv("HTTPS_PROXY", f"http://ops:{PROXY_PW}@proxy.example:8443")
    monkeypatch.setenv("QUANT_LAB_NOTE", SHARED)
    out = root / "work" / "child.json"
    code = "\n".join(
        (
            "import json, os, pathlib, sys",
            "cmd = pathlib.Path('/proc/self/cmdline').read_bytes().replace(bytes([0]), b' ')",
            "payload = {'cmd': cmd.decode('utf-8', 'replace'), 'env': dict(os.environ)}",
            "pathlib.Path(sys.argv[1]).write_text(json.dumps(payload), encoding='utf-8')",
        )
    )
    argv = [sys.executable, "-c", code, str(out), SHARED, f"--api-key={SK}", JWT]
    result = run_owned(
        argv,
        ProcLimits(timeout_s=3),
        env=scrub_env(__import__("os").environ),
        registry=ProcessRegistry(),
    )
    assert result.exit_code == 0, result.stderr
    _forbid(
        out.read_text(encoding="utf-8"), result.stdout, result.stderr, " ".join(result.argv_summary)
    )


def test_owned_runner_spawn_receives_neither_secret_env_nor_argv(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del root
    monkeypatch.setenv("MOONSHOT_API_KEY", SHARED)
    monkeypatch.setenv("QUANT_LAB_NOTE", SHARED)
    monkeypatch.setenv("HTTPS_PROXY", f"https://u:{PROXY_PW}@proxy.example:8443")
    seen: list[str] = []
    real = __import__("fx1.interactive.procs", fromlist=["_popen"])._popen

    def spy(argv: list[str], **kwargs: Any) -> Any:
        seen.extend(str(part) for part in argv)
        env = kwargs.get("env") or {}
        seen.extend(f"{key}={value}" for key, value in env.items())
        return real(argv, **kwargs)

    monkeypatch.setattr("fx1.interactive.procs._popen", spy)
    result = OwnedRunner(registry=ProcessRegistry())(["version", SHARED, f"--token={JWT}"], 10)
    _forbid("\n".join(seen), result.stdout, result.stderr, " ".join(result.argv_summary))


def test_repr_of_key_holding_objects_omits_values(root: Path) -> None:
    from fx1.interactive.profiles import ResolvedEndpoint, _EnvChoice, resolve_endpoint_info

    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        info = resolve_endpoint_info("fx1")
        assert info is not None
        choice = _EnvChoice(FX1_ENV, f"https://a:{URL_PW}@h.example/v1", "env:model", "X")
        client = _client(api_key=LITE_STORED, api_url=server.url, model="fx1-lite")
        _forbid(
            repr(info),
            str(info),
            repr(choice),
            str(choice),
            repr(client),
            repr(ResolvedEndpoint("hidden", info.completion_url, info.host, "stored", "....abcd")),
            _files(root),
        )


def test_generic_exception_and_traceback_do_not_leak(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: list[str] = []

    class _Boom:
        supports_cancel = False

        def complete_with_tools(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            del messages, kwargs
            raise RuntimeError(f"upstream returned {FX1_STORED} {JWT}")

    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        concierge = _concierge(root, out, backend_factory=lambda **_k: _Boom())
        concierge.handle_line("hello")
        err = _captured(capsys)
        assert "Traceback" not in err
        _surface(concierge, out, err, root)


def test_background_job_error_and_events_are_scrubbed(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fx1.interactive.redact import register_secret

    register_secret(FX1_STORED)
    register_secret(LITE_STORED)
    out: list[str] = []
    concierge = _concierge(root, out)

    def run(cancel: Any, emit: Any) -> None:
        del cancel
        emit("progress", f"using {FX1_STORED}")
        raise RuntimeError(f"failed with {LITE_STORED} {SK}")

    job = BackgroundJob(label="probe", run_fn=run)
    job.start()
    assert job.join(2.0)
    concierge._job = job  # noqa: SLF001
    concierge.tick()
    _forbid("\n".join(out), job.error, repr(job), _captured(capsys), _files(root))


def test_userinfo_url_is_refused_without_an_echo(root: Path) -> None:
    from fx1.interactive.profiles import set_endpoint

    url = f"http://alice:{URL_PW}@127.0.0.1:9/v1/chat/completions"
    with pytest.raises(ValueError) as caught:
        set_endpoint("fx1", FX1_STORED, url)
    text = "".join(traceback.format_exception(caught.value))
    _forbid(text, str(caught.value), repr(caught.value), _files(root, skip_credentials=False))


def test_credentials_file_is_the_only_stored_copy_and_is_private(root: Path) -> None:
    from fx1.interactive.profiles import set_endpoint, store_path

    with _Running({"fx1": FX1_STORED}) as server:
        set_endpoint("fx1", FX1_STORED, server.url)
        set_endpoint("fx1-lite", LITE_STORED, server.url)
        path = store_path()
        assert path.is_file()
        assert path.stat().st_mode & 0o777 == 0o600
        raw = path.read_text(encoding="utf-8")
        assert FX1_STORED in raw
        assert LITE_STORED in raw
        _forbid(_files(root))
        temps = [item.name for item in path.parent.iterdir() if item.name != path.name]
        assert not any(name.endswith(".tmp") for name in temps)


def test_unregistered_shapes_in_tool_arguments_do_not_persist(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(
            Reply.ok(
                None,
                tool_calls=[
                    _tool(
                        "flash_save",
                        {
                            "text": f"note {SK} {JWT} Bearer {BEARER}",
                            "task": f"https://a:{URL_PW}@h.example/x",
                        },
                    )
                ],
                served_model="fx1",
            )
        )
        server.queue(Reply.ok("not saved", served_model="fx1"))
        concierge = _concierge(root, out)
        concierge.handle_line("remember this")
        _surface(concierge, out, _captured(capsys), root)


def test_console_error_path_redacts_a_startup_failure(root: Path) -> None:
    from fx1.interactive import console

    out: list[str] = []

    class _Bad:
        model = "fx1"

        def startup_checks(self) -> list[str]:
            raise RuntimeError(f"store broke {FX1_STORED} {JWT}")

        def shutdown(self) -> None:
            return None

    def read(_prompt: str) -> str:
        raise EOFError

    console.run_console(
        engine=True,
        input_fn=read,
        output_fn=out.append,
        concierge_factory=lambda: _Bad(),
        history_path=root / "cfg" / "history",
    )
    _forbid("\n".join(out), _files(root))


def test_approval_prompt_does_not_keep_a_secret(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = ApprovalGate(decide_fn=_deny)
    question = f"run harness with {FX1_STORED} Bearer {BEARER}?"
    assert gate.ask(question, kind="consequential", key="harness:train") is False
    blob = json.dumps(gate.snapshot())
    _forbid(blob, repr(gate), _files(root), _captured(capsys))


def test_session_resume_does_not_restore_a_secret(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: list[str] = []
    with _Running({"fx1": FX1_STORED}) as server:
        _store_both(server.url)
        server.queue(Reply.ok(f"first {FX1_STORED}", served_model="fx1"))
        server.queue(Reply.ok("resumed pinball 0.1", served_model="fx1"))
        first = _concierge(root, out)
        first.handle_line(f"stash {FX1_STORED} {JWT}")
        session_id = first._session["id"]  # noqa: SLF001
        second = _concierge(root, out)
        second.handle_line(f"/resume {session_id}")
        second.handle_line("/sessions")
        _surface(second, out, _captured(capsys), root)


def test_env_only_proxy_userinfo_is_not_in_the_scrubbed_child(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del root
    monkeypatch.setenv("HTTPS_PROXY", f"http://alice:sec{URL_PW}@proxy.example:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    cleaned = scrub_env(__import__("os").environ)
    blob = "\n".join(f"{key}={value}" for key, value in cleaned.items())
    _forbid(blob)
    assert cleaned["HTTP_PROXY"] == "http://proxy.example:8080"


def test_key_holder_str_matches_repr_for_resolved_endpoint(root: Path) -> None:
    from fx1.interactive.profiles import resolve_endpoint_info

    with _Running({"fx1": FX1_STORED, "fx1-lite": LITE_STORED}) as server:
        _store_both(server.url)
        for model in ("fx1", "fx1-lite"):
            info = resolve_endpoint_info(model)
            assert info is not None
            assert str(info) == repr(info)
            _forbid(repr(info), info.completion_url, _files(root))
