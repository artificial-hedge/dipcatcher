"""The formal/property collection guard rejects an empty test module."""

from __future__ import annotations

from tests.support.collection_guard import (
    finish_collection,
    note_collect_report,
    reset_collection,
    zero_collected_module,
)


def test_empty_formal_module_is_reported() -> None:
    message = zero_collected_module("tests/formal/test_empty.py", "passed", 0)
    assert message == "tests/formal/test_empty.py collected 0 tests"


def test_empty_property_module_is_reported() -> None:
    message = zero_collected_module("tests/property/test_empty.py", "passed", 0)
    assert message == "tests/property/test_empty.py collected 0 tests"


def test_note_and_finish_count_an_empty_module_once() -> None:
    class _Report:
        nodeid = "tests/formal/test_empty.py"
        outcome = "passed"
        result: list[object] = []
        longrepr = None

    class _Session:
        testsfailed = 0

    reset_collection()
    report = _Report()
    note_collect_report(report)
    note_collect_report(report)
    assert report.outcome == "failed"
    assert report.longrepr == "tests/formal/test_empty.py collected 0 tests"
    session = _Session()
    finish_collection(session)
    finish_collection(session)
    assert session.testsfailed == 1
    reset_collection()


def test_populated_and_unwatched_modules_pass() -> None:
    assert zero_collected_module("tests/formal/test_stateful.py", "passed", 2) is None
    assert zero_collected_module("tests/property/test_drawdown.py", "passed", 1) is None
    assert zero_collected_module("tests/unit/test_empty.py", "passed", 0) is None
    assert zero_collected_module("tests/formal/conftest.py", "passed", 0) is None
    assert zero_collected_module("tests/property/_books.py", "passed", 0) is None
    assert zero_collected_module("tests/formal", "passed", 0) is None
    assert zero_collected_module("tests/formal/test_broken.py", "failed", 0) is None
