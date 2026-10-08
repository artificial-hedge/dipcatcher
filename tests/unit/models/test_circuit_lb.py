"""Exact small-circuit complexity tests."""

from __future__ import annotations

from quant_fund.models.circuit_lb import bench_circuit_lb, circuit_size, eval_gate, input_tables


def test_nand_is_not_of_and_not_not_of_a():
    """Old code returned ``~a & full`` for nand — ignoring b entirely."""
    full = (1 << (1 << 2)) - 1
    a, b = input_tables(2)
    assert eval_gate("nand", a, b, 2) == (~(a & b)) & full
    # counterexample the old code got wrong: nand(b, a) differs from ~b
    assert eval_gate("nand", full, b, 2) == (~b) & full
    assert eval_gate("nand", a, full, 2) == (~a) & full


def test_circuit_size_counts_shared_circuits_not_level_closure():
    """Tables needing two disjoint subcircuits were undercounted by 1:
    the old BFS let the last gate combine parents proven only by
    DIFFERENT (k-1)-gate circuits."""
    assert circuit_size(0x42, 3) == 3
    assert circuit_size(0x18, 3) == 3
    assert circuit_size(0x9A, 3) == 3


def test_circuit_size_basics():
    assert circuit_size(0b0110, 2) == 1  # xor2 in basis
    assert circuit_size(0b1000, 2) == 1  # and2
    assert circuit_size(0b1001, 2) == 2  # eq = NOT xor
    assert circuit_size(0, 2) == 0  # constant
    assert circuit_size(0b10010110, 3) == 2  # parity3 = x^y^z


def test_gate_algebra():
    full2 = 15
    x, y = input_tables(2)
    assert eval_gate("and", x, y, 2) == 0b1000
    assert eval_gate("or", x, y, 2) == 0b1110
    assert eval_gate("xor", x, y, 2) == 0b0110
    assert eval_gate("nand", x, y, 2) == 0b0111
    assert full2 == 15


def test_bench():
    assert bench_circuit_lb()["synthetic_circuit_lb"] == 1.0
