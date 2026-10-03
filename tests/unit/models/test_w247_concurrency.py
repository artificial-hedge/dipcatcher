"""Wave-247 concurrency canon tests."""

import random

from quant_fund.models.atomics_tas import TasLock, bench_atomics_tas
from quant_fund.models.bakery_lock import bench_bakery_lock, run_bakery
from quant_fund.models.channel_select import Chan, bench_channel_select, select
from quant_fund.models.peterson_lock import bench_peterson_lock, run_peterson
from quant_fund.models.rw_lock import bench_rw_lock, run_rw
from quant_fund.models.work_stealing import bench_work_stealing, run_ws


def test_peterson_basic():
    m, p = run_peterson([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert m and p


def test_peterson_bench():
    assert bench_peterson_lock()["synthetic_mutual_exclusion"] == 1.0


def test_bakery_basic():
    m, order = run_bakery(2, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert m and order


def test_bakery_bench():
    assert bench_bakery_lock()["synthetic_mutual_exclusion"] == 1.0


def test_rw_basic():
    ok, n = run_rw([("r", 0), ("r", 1), ("w", 2)])
    assert ok and n == 3


def test_rw_bench():
    assert bench_rw_lock()["synthetic_all_served"] == 1.0


def test_atomics_basic():
    lock = TasLock()
    lock.acquire()
    lock.release()
    assert lock.state == 0


def test_atomics_bench():
    assert bench_atomics_tas()["synthetic_cas_count_exact"] == 1.0


def test_ws_basic():
    rng = random.Random(0)
    done = run_ws(list(range(10)), 2, rng)
    assert sorted(done) == list(range(10))


def test_ws_bench():
    assert bench_work_stealing()["synthetic_work_conserved"] == 1.0


def test_channel_basic():
    c = Chan(cap=1)
    c.buf.append(5)
    i = select([("recv", c, 0)], random.Random(0))
    assert i == 0


def test_channel_bench():
    assert bench_channel_select()["synthetic_messages_conserved"] == 1.0
