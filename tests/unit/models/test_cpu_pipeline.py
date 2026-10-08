"""Tests for cpu_pipeline — 5-stage pipeline hazard model."""

from __future__ import annotations

import quant_fund.models.cpu_pipeline as cp
from quant_fund.models.cpu_pipeline import bench_cpu_pipeline, simulate


def test_simulate_owns_its_register_file() -> None:
    # the old simulate() discarded the pipeline-modeled register file and
    # re-ran sequential exec, making regs_match_seq vacuous. Breaking
    # _seq_exec must not change simulate's output.
    prog = [(1, 0, 5), (2, 1, 3), (3, 2, -1)]
    ref = [0] * 8
    cp._seq_exec(prog, ref)
    orig = cp._seq_exec
    try:
        cp._seq_exec = lambda p, r: r.__setitem__(slice(None), [-999] * 8)
        _, regs_stall = simulate(prog, forwarding=False)
        _, regs_fwd = simulate(prog, forwarding=True)
    finally:
        cp._seq_exec = orig
    assert regs_stall == ref
    assert regs_fwd == ref


def test_raw_hazard_gap1() -> None:
    prog = [(1, 0, 5), (2, 1, 3)]
    _, regs = simulate(prog, forwarding=False)
    assert regs[2] == 8


def test_forwarding_cycles_not_worse() -> None:
    prog = [(1, 0, 5), (2, 1, 3), (3, 2, 1), (4, 3, 2)]
    c_stall, r1 = simulate(prog, forwarding=False)
    c_fwd, r2 = simulate(prog, forwarding=True)
    assert c_fwd <= c_stall
    assert r1 == r2


def test_empty_program() -> None:
    assert simulate([], forwarding=False) == (0, [0] * 8)


def test_bench_cpu_pipeline() -> None:
    out = bench_cpu_pipeline()
    assert out["synthetic_regs_match_seq"] == 1.0
    assert out["synthetic_fwd_never_worse"] == 1.0
    assert out["synthetic_cpi_ge_1"] == 1.0
