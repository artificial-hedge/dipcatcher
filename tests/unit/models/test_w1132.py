"""Wave-1132 history-4 canon tests."""

from __future__ import annotations

from quant_fund.models.history_of_capitalism import bench_history_of_capitalism
from quant_fund.models.history_of_emotions import bench_history_of_emotions
from quant_fund.models.history_of_religions import bench_history_of_religions
from quant_fund.models.history_of_sexuality import bench_history_of_sexuality
from quant_fund.models.history_of_the_book import bench_history_of_the_book
from quant_fund.models.microhistory import bench_microhistory


def test_history_of_emotions():
    assert bench_history_of_emotions()["synthetic_history_of_emotions"] == 1.0


def test_history_of_sexuality():
    assert bench_history_of_sexuality()["synthetic_history_of_sexuality"] == 1.0


def test_history_of_the_book():
    assert bench_history_of_the_book()["synthetic_history_of_the_book"] == 1.0


def test_history_of_capitalism():
    assert bench_history_of_capitalism()["synthetic_history_of_capitalism"] == 1.0


def test_history_of_religions():
    assert bench_history_of_religions()["synthetic_history_of_religions"] == 1.0


def test_microhistory():
    assert bench_microhistory()["synthetic_microhistory"] == 1.0
