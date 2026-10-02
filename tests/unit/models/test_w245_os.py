"""Wave-245 OS-2 canon tests."""

import random

from quant_fund.models.elf_loader import bench_elf_loader, build_elf, parse_elf
from quant_fund.models.malloc_freelist import Heap, bench_malloc_freelist
from quant_fund.models.mlfq_sched import bench_mlfq_sched, run_mlfq
from quant_fund.models.mmap_pager import Pager, bench_mmap_pager
from quant_fund.models.semaphore_monitor import bench_semaphore_monitor, run_bounded_buffer
from quant_fund.models.syscall_layer import Kernel, bench_syscall_layer, syscall


def test_elf_basic():
    blob = build_elf({"a": b"hello", "b": b"world"}, entry=0x401000)
    ent, secs = parse_elf(blob)
    assert ent == 0x401000 and sorted(secs.values()) == [b"hello", b"world"]


def test_elf_bench():
    assert bench_elf_loader()["synthetic_payload_roundtrip"] == 1.0


def test_malloc_basic():
    h = Heap(10)
    a = h.alloc(4)
    assert a == 0
    h.dealloc(a, 4)
    assert h.free == [(0, 10)]


def test_malloc_bench():
    assert bench_malloc_freelist()["synthetic_no_overlap"] == 1.0


def test_pager_basic():
    p = Pager(5, 2)
    p.write(3, 42)
    assert p.read(3) == 42 and p.faults == 1


def test_pager_bench():
    assert bench_mmap_pager()["synthetic_read_write_correct"] == 1.0


def test_mlfq_basic():
    order = run_mlfq([(0, 2), (0, 1)])
    assert sorted(order) == [0, 0]


def test_mlfq_bench():
    assert bench_mlfq_sched()["synthetic_all_jobs_complete"] == 1.0


def test_semaphore_basic():
    rng = random.Random(0)
    consumed, ok = run_bounded_buffer([1, 2, 3], 2, rng)
    assert ok and consumed == [1, 2, 3]


def test_semaphore_bench():
    assert bench_semaphore_monitor()["synthetic_no_loss_no_dup"] == 1.0


def test_syscall_basic():
    k = Kernel()
    fd = syscall(k, 0, "file")
    assert fd >= 0 and syscall(k, 2, fd) == 4
    assert syscall(k, 1, fd) == 0


def test_syscall_bench():
    assert bench_syscall_layer()["synthetic_enosys"] == 1.0
