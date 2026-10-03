"""Wave-245 adapter tests."""

from quant_fund.research.benches_w245 import (
    bench_elf_loader_family,
    bench_malloc_freelist_family,
    bench_mlfq_sched_family,
    bench_mmap_pager_family,
    bench_semaphore_monitor_family,
    bench_syscall_layer_family,
)


def test_bench_elf_loader_family():
    out = bench_elf_loader_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_malloc_freelist_family():
    out = bench_malloc_freelist_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_mlfq_sched_family():
    out = bench_mlfq_sched_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_mmap_pager_family():
    out = bench_mmap_pager_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_semaphore_monitor_family():
    out = bench_semaphore_monitor_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_syscall_layer_family():
    out = bench_syscall_layer_family()
    assert out and all(k.startswith("synthetic_") for k in out)
