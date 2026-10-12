"""Merge defects in /data, exec, and the chat dataset note."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fx1.interactive.chat_turn import _call_backend
from fx1.interactive.context_window import ContextBudget
from fx1.interactive.data_ux import (
    PIT_NOTE,
    Attachment,
    annotate_summary,
    dataset_refs,
    file_sha256,
    handle_data,
    merge_attachments,
    model_context,
    rehash_rows,
    restore_stored,
)
from fx1.interactive.headless import ExecRequest, run_exec
from fx1.interactive.model_client import ErrorKind, ModelError
from fx1.interactive.model_ux import build_outbound, render_models_table
from fx1.interactive.superpower import SuperpowerPlan, SuperpowerResult
from fx1.interactive.superpower_glue import operator_status

_KEY = "sk-review-r1-test-key-aaaa"


class ScriptHarness:
    def __init__(self, stdout: str = "", exit_code: int = 0, mutate: object | None = None) -> None:
        self.stdout = stdout
        self.exit_code = exit_code
        self.mutate = mutate

    def run(
        self, name: str, extra_args: list[str] | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del name, kwargs
        if callable(self.mutate):
            self.mutate()
        return SimpleNamespace(exit_code=self.exit_code, stdout=self.stdout, stderr="")


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def _csv(path: Path, text: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = text or "timestamp,realized\n2020-01-01T00:00:00+00:00,1.0\n"
    path.write_text(body, encoding="utf-8")
    return path


def _profile(path: Path, **overrides: object) -> str:
    payload: dict[str, object] = {
        "schema": "dataset_profile.v1",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "n_rows": 1,
        "columns": [
            {"name": "timestamp", "dtype": "String", "nulls": 0},
            {"name": "realized", "dtype": "Float64", "nulls": 0},
        ],
        "time": {"column": "timestamp", "parsed": True, "strictly_increasing": True},
        "suggested_roles": {"available_time": {"column": None}},
    }
    payload.update(overrides)
    return json.dumps(payload)


def _stored(path: Path, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "path": str(path),
        "kind": "csv",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
        "rows": 1,
        "columns": ["timestamp", "realized"],
        "stale": False,
        "has_time": True,
        "time_monotonic": True,
        "has_nulls": False,
        "has_available_time": False,
        "available_time_col": "",
    }
    row.update(overrides)
    return row


def _attachment(path: str) -> Attachment:
    return Attachment(
        path=path,
        kind="csv",
        label="file",
        sha256="ab" * 32,
        size=1,
        rows=1,
        columns=("timestamp",),
        stale=False,
        has_time=False,
        time_monotonic=False,
        has_nulls=False,
        has_available_time=False,
    )


def test_data_prefix_symlink_does_not_attach_a_nested_decoy(tmp_path: Path) -> None:
    nested = _csv(tmp_path / "data" / "data" / "escape.csv")
    outside = _csv(tmp_path / "outside.csv")
    link = tmp_path / "data" / "escape.csv"
    link.symlink_to(outside)
    rows, text = handle_data("attach data/escape.csv", [], ScriptHarness())
    assert rows == []
    assert "stay inside data" in text
    assert str(nested.resolve()) not in text
    bare, bare_text = handle_data("attach escape.csv", [], ScriptHarness())
    assert bare == []
    assert "stay inside data" in bare_text


def test_windows_style_attach_reaches_the_data_directory(tmp_path: Path) -> None:
    path = _csv(tmp_path / "data" / "panel.csv")
    harness = ScriptHarness(_profile(path))
    rows, text = handle_data(r"attach data\panel.csv", [], harness)
    assert rows
    assert rows[0].path == str(path.resolve())
    assert "not found" not in text
    quoted, _quoted_text = handle_data(r'attach "data\panel.csv"', [], harness)
    assert quoted[0].path == str(path.resolve())


def test_unterminated_quote_is_a_usage_error() -> None:
    with pytest.raises(ValueError, match="usage: /data"):
        handle_data('attach "unterminated', [], ScriptHarness())


def test_hostile_profile_cannot_bind_a_column_it_invented(tmp_path: Path) -> None:
    path = _csv(tmp_path / "data" / "panel.csv")
    invented = {
        "available_time": {"column": "not_a_column", "confidence": "exact_name"},
    }
    rows, _text = handle_data(
        f"attach {path}",
        [],
        ScriptHarness(_profile(path, suggested_roles=invented)),
    )
    assert dataset_refs(rows)[0].available_time_col == ""
    listed = _profile(
        path,
        columns=[
            {"name": "timestamp", "dtype": "String", "nulls": 0},
            {"name": "not_a_column", "dtype": "String", "nulls": 0},
        ],
        suggested_roles=invented,
    )
    poisoned, _shown = handle_data(f"attach {path}", [], ScriptHarness(listed))
    assert dataset_refs(poisoned)[0].available_time_col == ""
    assert "not_a_column" not in poisoned[0].columns
    assert "not_a_column" not in model_context(poisoned)


def test_shell_metachar_and_flag_columns_are_not_bound(tmp_path: Path) -> None:
    path = _csv(tmp_path / "data" / "panel.csv", "timestamp,a;b\n2020-01-01T00:00:00+00:00,1\n")
    roles = {"available_time": {"column": "a;b", "confidence": "exact_name"}}
    rows, _text = handle_data(
        f"attach {path}",
        [],
        ScriptHarness(
            _profile(
                path,
                columns=[
                    {"name": "timestamp", "dtype": "String", "nulls": 0},
                    {"name": "a;b", "dtype": "String", "nulls": 0},
                ],
                suggested_roles=roles,
            )
        ),
    )
    assert dataset_refs(rows)[0].available_time_col == ""
    exact = {"available_time": {"column": "timestamp", "confidence": "exact_name"}}
    bound, _shown = handle_data(
        f"attach {path}", [], ScriptHarness(_profile(path, suggested_roles=exact))
    )
    assert dataset_refs(bound)[0].available_time_col == "timestamp"


def test_relative_resume_keeps_the_file_and_rejects_bad_rows(tmp_path: Path) -> None:
    path = _csv(tmp_path / "data" / "panel.csv")
    other = _csv(tmp_path / "data" / "other.csv")
    kept, warnings = restore_stored([_stored(Path("data/panel.csv"))], cwd=tmp_path)
    assert kept
    assert Path(kept[0].path) == path.resolve()
    assert kept[0].stale is False
    assert warnings == ()
    windows_row = _stored(path)
    windows_row["path"] = r"C:\outside\panel.csv"
    windows, win_warnings = restore_stored([windows_row], cwd=tmp_path)
    assert windows == []
    assert win_warnings
    mismatch, mismatch_warnings = restore_stored([_stored(path, kind="parquet")], cwd=tmp_path)
    assert mismatch == []
    assert any("invalid" in note for note in mismatch_warnings)
    evil = _stored(path, columns=["timestamp", "--evil"], available_time_col="--evil")
    restored, _notes = restore_stored([evil], cwd=tmp_path)
    assert dataset_refs(restored)[0].available_time_col == ""
    missing_legacy = [{"path": str(path.resolve()), "kind": "csv", "label": "file"}]
    assert restore_stored([], cwd=tmp_path, legacy=missing_legacy)[0] == []
    legacy, _legacy_notes = restore_stored(None, cwd=tmp_path, legacy=missing_legacy)
    assert len(legacy) == 1
    outside = _stored(path)
    outside["path"] = str(tmp_path / "panel.csv")
    rows, _dropped = restore_stored([_stored(path), outside, _stored(other)], cwd=tmp_path)
    assert [Path(row.path).name for row in rows] == ["panel.csv", "other.csv"]
    removed, text = handle_data("remove 1", rows, ScriptHarness())
    assert text.startswith("removed ")
    assert [Path(row.path).name for row in removed] == ["other.csv"]


def test_resume_warns_when_extra_dataset_rows_are_dropped(tmp_path: Path) -> None:
    paths = [_csv(tmp_path / "data" / f"f{index}.csv") for index in range(9)]
    _kept, warnings = restore_stored([_stored(path) for path in paths], cwd=tmp_path)
    assert len(_kept) == 8
    assert any("extra records" in note for note in warnings)


def test_huge_row_count_is_not_kept() -> None:
    path = _csv(Path("data/panel.csv"))
    kept, _warnings = restore_stored([_stored(path, rows=2**80)], cwd=Path.cwd())
    assert kept[0].rows is None


def test_digest_tracks_bytes_changed_during_profile(tmp_path: Path) -> None:
    path = _csv(tmp_path / "data" / "panel.csv")

    def rewrite() -> None:
        path.write_text(path.read_text(encoding="utf-8") + "x\n", encoding="utf-8")

    rows, text = handle_data(f"attach {path}", [], ScriptHarness(_profile(path), mutate=rewrite))
    assert rows
    assert rows[0].sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rows[0].stale is False
    assert "changed during attach" in text
    _updated, reason = rehash_rows(rows)
    assert reason is None


def test_file_sha256_does_not_follow_a_symlink(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform has no O_NOFOLLOW")
    secret = _csv(tmp_path / "secret.csv", "secret,value\n1,2\n")
    link = tmp_path / "data" / "link.csv"
    link.parent.mkdir(parents=True)
    link.symlink_to(secret)
    with pytest.raises(OSError):
        file_sha256(link)
    assert file_sha256(secret) == hashlib.sha256(secret.read_bytes()).hexdigest()


def test_casefolded_paths_are_one_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os.path, "normcase", lambda value: value.casefold())
    first = _attachment("/tmp/Data/Foo.csv")
    second = _attachment("/tmp/data/foo.csv")
    merged, problem = merge_attachments([first], [second])
    assert problem is None
    assert len(merged) == 1
    assert merged[0].path == second.path


def test_false_pit_claim_does_not_suppress_the_limitation() -> None:
    text = annotate_summary("point-in-time validity is verified", unchecked=True)
    assert "point-in-time validity is NOT verified" in text
    assert annotate_summary(PIT_NOTE, unchecked=True) == PIT_NOTE
    assert annotate_summary("pinball 0.1", unchecked=False) == "pinball 0.1"


def test_stale_dataset_refusal_is_an_error(tmp_path: Path) -> None:
    from fx1.interactive.approvals import ApprovalGate
    from fx1.interactive.concierge import Concierge
    from fx1.interactive.sessions import SessionStore

    class InspectHarness:
        def run(
            self, name: str, extra_args: list[str] | None = None, **kwargs: object
        ) -> SimpleNamespace:
            del name, kwargs
            args = list(extra_args or [])
            target = Path(args[args.index("--data") + 1])
            return SimpleNamespace(exit_code=0, stdout=_profile(target), stderr="")

    store = SessionStore(tmp_path / "sessions")
    out: list[str] = []
    concierge = Concierge(
        session_store=store,
        harness=InspectHarness(),
        approvals=ApprovalGate(),
        output_fn=out.append,
        animated=False,
        input_fn=lambda _prompt: "n",
    )
    path = _csv(tmp_path / "data" / "panel.csv")
    concierge.handle_line(f"/data attach {path}")
    path.write_text(path.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
    out.clear()
    concierge.handle_line("/superpower score the forecasts --sync")
    assert concierge.last_status == "error"
    assert "changed since attach" in concierge.last_answer
    session_id = str(concierge._session["id"])  # noqa: SLF001
    outcome = run_exec(
        ExecRequest(
            prompt="/superpower score the forecasts --sync",
            session=session_id,
            timeout_s=30,
        ),
        session_store=store,
        install_signals=False,
        harness=InspectHarness(),
    )
    assert outcome.status == "error"
    assert outcome.exit_code == 1
    assert any("changed since attach" in item for item in outcome.errors)
    plan = SuperpowerPlan(goal="g")
    assert operator_status(SuperpowerResult(goal="g", plan=plan, status="no-evidence")) == "ok"
    assert operator_status(SuperpowerResult(goal="g", plan=plan, status="error")) == "error"


def test_dataset_note_stays_inside_the_user_turn(tmp_path: Path) -> None:
    from fx1.interactive import profiles
    from fx1.interactive.concierge import Concierge
    from fx1.interactive.model_client import ModelClient, RetryPolicy
    from tests.fx1.stub_model_server import StubModelServer

    path = _csv(tmp_path / "data" / "panel.csv")
    out: list[str] = []
    retry = RetryPolicy(
        max_attempts=1,
        base_delay_s=0.0,
        max_delay_s=0.0,
        jitter=0.0,
        honor_retry_after_up_to_s=1.0,
    )

    def factory(**kwargs: object) -> ModelClient:
        options: dict[str, object] = {"retry": retry, "sleep": lambda _delay: None, "timeout_s": 20}
        options.update(kwargs)
        return ModelClient(**options)  # type: ignore[arg-type]

    with StubModelServer({"fx1": _KEY}) as server:
        profiles.set_endpoint("fx1", _KEY, server.url)
        concierge = Concierge(
            harness=ScriptHarness(_profile(path)),
            backend_factory=factory,
            output_fn=out.append,
            animated=False,
            input_fn=lambda _prompt: "n",
        )
        concierge.handle_line(f"/data attach {path}")
        concierge.handle_line("hello from the operator")
        body = server.requests[-1].body
    assert body is not None
    messages = body["messages"]
    users = [message for message in messages if message["role"] == "user"]
    assert [message["role"] for message in messages[:2]] == ["system", "user"]
    assert len(users) == 1
    assert "hello from the operator" in users[0]["content"]
    assert "UNTRUSTED_DATA" in users[0]["content"]
    assert "END_UNTRUSTED_DATA" in users[0]["content"]
    stored = [message for message in concierge.messages() if message.get("role") == "user"]
    assert stored
    assert "UNTRUSTED_DATA" not in str(stored[-1].get("content"))
    plain = build_outbound([{"role": "user", "content": "hello from the operator"}], "fx1")
    noted = build_outbound(
        [{"role": "user", "content": f"{users[0]['content']}"}],
        "fx1",
    )
    assert noted.chars > plain.chars


def test_shrink_counts_the_dataset_note() -> None:
    note = "N" * 800
    seen: list[int] = []

    class Budget(ContextBudget):
        def shrink(self, model: str, sent_chars: int) -> bool:
            seen.append(sent_chars)
            return False

    class Host:
        model = "fx1"
        _context_trim_said = False
        _context_budget = Budget()
        _pending_snapshot = None
        _pass_deadline = False

        def dataset_model_note(self) -> str:
            return note

        def _transcript(self) -> list[dict[str, str]]:
            return [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "second"},
            ]

        def _clock(self) -> float:
            return 0.0

        def _orb(self) -> object:
            return _Null()

        def _say(self, text: str) -> None:
            del text

    class Backend:
        def complete_with_tools(
            self, messages: list[dict[str, object]], **kwargs: object
        ) -> object:
            del messages, kwargs
            raise ModelError(
                ErrorKind.CONTEXT_TOO_LONG,
                status=400,
                retryable=False,
                retry_after_s=None,
                model="fx1",
                host="stub",
                hint="shorter",
                summary="too long",
            )

    with pytest.raises(ModelError):
        _call_backend(Host(), Backend(), _Tracker(), _NullEvent(), [])
    assert seen
    assert seen[0] >= 800


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _NullEvent:
    def is_set(self) -> bool:
        return False


class _Tracker:
    def charge_step(self) -> None:
        return None

    def charge_model_call(self) -> None:
        return None

    def request_max_tokens(self) -> None:
        return None


def test_served_model_id_cannot_break_the_models_table(tmp_path: Path) -> None:
    del tmp_path
    plain = render_models_table(
        "fx1",
        {"fx1": {"served": "fx1", "latency_s": 0.2, "matched": False}},
    )
    hostile = render_models_table(
        "fx1",
        {"fx1": {"served": "fx1\nINJECTED\x1b[31m", "latency_s": 0.2, "matched": False}},
    )
    assert len(hostile.splitlines()) == len(plain.splitlines())
    assert "\x1b" not in hostile
    assert "INJECTED" not in hostile
