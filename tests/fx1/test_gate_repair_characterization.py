"""Pins for the behavior-preserving fx1 gate repairs.

These lock return values, exception text, and on-disk outcomes of the
functions whose control flow was split to satisfy the complexity ratchet.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fx1.flash.refresh import refresh_entry
from fx1.flash.store import FlashStore
from fx1.serve.api import create_app
from fx1.serve.backends import hosted_api_url_problem
from fx1.webresearch.engine import Claim, group_claims


@pytest.fixture(autouse=True)
def _isolate_fx1_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("", "must be an HTTPS URL"),
        ("not-a-url", "must be an HTTPS URL"),
        ("ftp://example.com/v1", "must be an HTTPS URL"),
        ("https://", "must be an HTTPS URL"),
        ("http://[not-an-ip]/v1", "must be a well-formed HTTPS URL"),
        ("https://user:pass@example.com/v1", "must not embed userinfo credentials"),
        ("https://user@example.com/v1", "must not embed userinfo credentials"),
        ("https://example.com/v1?x=1", "must not carry params, query, or fragment"),
        ("https://example.com/v1#f", "must not carry params, query, or fragment"),
        ("https://example.com/v1;p=1", "must not carry params, query, or fragment"),
        ("https://example.com:0/v1", "port must be 1-65535"),
        ("https://example.com:65536/v1", "port must be 1-65535"),
        ("https://example.com:abc/v1", "port must be 1-65535"),
        ("https://:443/x", "must name a host"),
        ("http://example.com/v1", "must use HTTPS unless the host is a literal loopback address"),
        ("http://localhost/v1", "must use HTTPS unless the host is a literal loopback address"),
        ("HTTP://example.com/v1", "must use HTTPS unless the host is a literal loopback address"),
        ("http://0.0.0.0/v1", "must use HTTPS unless the host is a literal loopback address"),
        ("https://example.com/v1", None),
        ("https://example.com:443/v1/chat/completions", None),
        ("https://example.com:1/v1", None),
        ("https://example.com:65535/v1", None),
        ("http://127.0.0.1/v1", None),
        ("http://127.0.0.1:9/v1", None),
        ("http://[::1]/v1", None),
        ("http://[::ffff:127.0.0.1]/v1", None),
        ("https://[::1]/v1", None),
    ],
)
def test_hosted_api_url_problem_messages(url: str, problem: str | None) -> None:
    assert hosted_api_url_problem(url) == problem


def test_refresh_without_sources_does_not_stamp(tmp_path: Path) -> None:
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    entry = store.add(text="unsourced finding")
    report = refresh_entry(store, entry, fetch_fn=lambda url: None)
    assert report.checked == []
    assert report.reachable == []
    assert report.unreachable == []
    assert report.note == "no sources recorded; nothing to refresh"
    saved = store.get(entry.id)
    assert saved is not None
    assert saved.refresh_attempted_at == ""
    assert saved.refreshed_at == ""
    assert saved.refresh_note == ""
    assert saved.text == "unsourced finding"


def test_refresh_skips_blank_urls_and_formats_each_outcome(tmp_path: Path) -> None:
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    entry = store.add(
        text="mixed sources",
        sources=[
            {"url": "", "title": "blank"},
            {"url": "https://ok.example/a", "title": "ok"},
            {"url": "https://bad.example/b", "title": "bad"},
            {"url": "https://err.example/c", "title": "err"},
            {"url": "https://edge.example/d", "title": "edge"},
        ],
    )

    class _Page:
        def __init__(self, status: int | None, error: str | None) -> None:
            self.status = status
            self.error = error

    def fetch(url: str) -> object:
        if url.endswith("/a"):
            return _Page(200, "")
        if url.endswith("/b"):
            return _Page(404, "")
        if url.endswith("/c"):
            raise TimeoutError("timed out")
        if url.endswith("/d"):
            return _Page(399, None)
        raise AssertionError(url)

    report = refresh_entry(store, entry, fetch_fn=fetch)
    assert report.checked == [
        "https://ok.example/a",
        "https://bad.example/b",
        "https://err.example/c",
        "https://edge.example/d",
    ]
    assert report.reachable == ["https://ok.example/a", "https://edge.example/d"]
    assert report.unreachable == [
        "https://bad.example/b (HTTP 404)",
        "https://err.example/c (TimeoutError: timed out)",
    ]
    assert report.note == (
        "2/4 sources reachable; unreachable: "
        "https://bad.example/b (HTTP 404), "
        "https://err.example/c (TimeoutError: timed out)"
    )
    saved = store.get(entry.id)
    assert saved is not None
    assert saved.refresh_note == report.note
    assert saved.refresh_attempted_at
    assert saved.refreshed_at == ""
    assert saved.text == "mixed sources"


def test_refresh_all_reachable_stamps_the_same_clock(tmp_path: Path) -> None:
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    entry = store.add(
        text="both live",
        sources=[{"url": "https://a.example/1"}, {"url": "https://b.example/2"}],
    )

    class _Ok:
        status = 200
        error = ""

    report = refresh_entry(store, entry, fetch_fn=lambda url: _Ok())
    assert report.all_reachable is True
    assert report.note == "2/2 sources reachable; finding refreshed without change"
    saved = store.get(entry.id)
    assert saved is not None
    assert saved.refresh_attempted_at == saved.refreshed_at
    assert saved.refresh_note == report.note


def test_refresh_http_error_string_wins_over_status(tmp_path: Path) -> None:
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    entry = store.add(text="policy block", sources=[{"url": "https://blocked.example/x"}])

    class _Blocked:
        status = 200
        error = "policy"

    class _Bare:
        pass

    report = refresh_entry(store, entry, fetch_fn=lambda url: _Blocked())
    assert report.unreachable == ["https://blocked.example/x (policy)"]
    assert report.note == "0/1 sources reachable; unreachable: https://blocked.example/x (policy)"

    bare = store.add(text="no attributes", sources=[{"url": "https://bare.example/y"}])
    bare_report = refresh_entry(store, bare, fetch_fn=lambda url: _Bare())
    assert bare_report.unreachable == ["https://bare.example/y (HTTP None)"]


def test_store_read_of_missing_file_is_empty(tmp_path: Path) -> None:
    store = FlashStore(tmp_path / "absent" / "entries.jsonl")
    assert store.all() == []
    assert store.read_errors() == []


def test_store_rejects_directory_at_file_path(tmp_path: Path) -> None:
    path = tmp_path / "flash" / "entries.jsonl"
    path.parent.mkdir()
    path.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        FlashStore(path).add(text="secret")


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_store_symlink_and_mode_contract(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("do not alter", encoding="utf-8")
    linked = tmp_path / "flash" / "entries.jsonl"
    linked.parent.mkdir()
    linked.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        FlashStore(linked).add(text="secret")
    assert target.read_text(encoding="utf-8") == "do not alter"

    path = tmp_path / "real" / "entries.jsonl"
    store = FlashStore(path)
    first = store.add(text="first finding")
    second = store.add(text="second finding")
    assert [entry.text for entry in store.all()] == ["first finding", "second finding"]
    assert store.get(first.id) is not None
    assert store.get(second.id) is not None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_group_claims_keeps_cross_source_pairs_in_walk_order() -> None:
    left = Claim(
        text="Not so: inflation was 3.4%",
        source_url="https://a",
        tokens=frozenset({"inflation", "was"}),
    )
    right = Claim(
        text="Inflation was 8.9%",
        source_url="https://b",
        tokens=frozenset({"inflation", "was"}),
    )
    same = Claim(
        text="Not so: inflation was 1.0%",
        source_url="https://a",
        tokens=frozenset({"inflation", "was"}),
    )
    groups = group_claims([left, right, same])
    assert len(groups) == 1
    assert groups[0].contradictions == [
        (left.text, right.text),
        (right.text, same.text),
    ]
    assert groups[0].status == "contested"


def test_receipt_fetch_rejects_a_non_hex_digest(tmp_path: Path) -> None:
    client = TestClient(create_app(receipts_dir=tmp_path / "receipts"))
    response = client.get("/receipts/not-a-sha")
    assert response.status_code == 422
    assert response.json() == {
        "detail": "sha256 must be 64 lowercase hex",
        "code": "invalid_sha256",
    }
