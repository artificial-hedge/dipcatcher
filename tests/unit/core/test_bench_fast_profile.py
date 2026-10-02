"""Regression tests for the ``_bench_fast.py --profile`` report.

The profile branch must render the internal-time table and the cumulative-time
table independently. Sharing one ``StringIO`` and slicing it twice makes the
second print re-emit the first table's prefix, so the cumulative-time table
never reaches stdout.
"""

from __future__ import annotations

import cProfile
import sys

from scripts import _bench_fast as bf


def _profiled() -> None:
    sum(i * i for i in range(100))


def _profile() -> cProfile.Profile:
    pr = cProfile.Profile()
    pr.enable()
    _profiled()
    pr.disable()
    return pr


def test_profile_table_renders_the_requested_sort() -> None:
    pr = _profile()
    tottime = bf._profile_table(pr, "tottime", 10)
    cumtime = bf._profile_table(pr, "cumtime", 10)

    assert "internal time" in tottime
    assert "cumulative time" not in tottime
    assert "cumulative time" in cumtime


def test_profile_branch_prints_both_tables(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bf, "load_workload", lambda: (object(), object()))
    monkeypatch.setattr(bf, "make_cfg", lambda: object())
    monkeypatch.setattr(bf, "run_fast", lambda bars, weights, cfg: _profiled())
    monkeypatch.setattr(sys, "argv", ["_bench_fast.py", "--profile"])

    assert bf.main() is None
    out = capsys.readouterr().out
    assert "internal time" in out
    assert "cumulative time" in out
