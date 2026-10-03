"""Wave-256 memory-models canon tests."""

from quant_fund.models.epoch_reclaim import EBR, bench_epoch_reclaim
from quant_fund.models.flat_combining import _combine, bench_flat_combining
from quant_fund.models.hazard_pointer import HazardSim, bench_hazard_pointer
from quant_fund.models.ms_queue import MSQueue, bench_ms_queue
from quant_fund.models.rcu_lock import RCU, bench_rcu_lock
from quant_fund.models.seqlock import bench_seqlock


def test_hazard_basic():
    sim = HazardSim(2)
    sim.protect(0, 5)
    sim.retire(5)
    sim.retire(6)
    freed = sim.collect()
    assert freed == {6}


def test_hazard_bench():
    assert bench_hazard_pointer()["synthetic_hazard_safety"] == 1.0


def test_seqlock_bench():
    assert bench_seqlock()["synthetic_seqlock_consistent"] == 1.0


def test_msq_fifo():
    q = MSQueue()
    q.enq(1)
    q.enq(2)
    assert q.deq() == 1
    assert q.deq() == 2
    assert q.deq() is None


def test_msq_bench():
    assert bench_ms_queue()["synthetic_msq_linearizable"] == 1.0


def test_ebr_advance():
    ebr = EBR(2)
    ebr.retire(9)
    freed = ebr.try_advance()
    assert freed >= 0


def test_ebr_bench():
    assert bench_epoch_reclaim()["synthetic_ebr_safe"] == 1.0


def test_combining():
    assert _combine([("push", 3), ("push", 4), ("pop", 0)]) == [-1, -1, 4]


def test_combining_bench():
    assert bench_flat_combining()["synthetic_combining_equiv"] == 1.0


def test_rcu_basic():
    rcu = RCU()
    rcu.update()
    assert rcu.synchronize() == 1


def test_rcu_bench():
    assert bench_rcu_lock()["synthetic_rcu_grace"] == 1.0
