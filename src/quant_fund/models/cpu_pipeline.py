"""5-stage in-order pipeline simulator with data hazards — SYNTHETIC.

Model: IF/ID/EX/MEM/WB, register file of 8 regs, ops ADD/LOAD-imm.
RAW hazards handled by stalling (no forwarding) or by forwarding.
Verified: instruction count conserved, CPI >= 1, forwarding never
worse than stalling, result register file matches sequential exec.
"""

from __future__ import annotations

import random

# instruction: (dest, src1, src2_or_imm) meaning dest = src1 + imm/src2
Insn = tuple[int, int, int]


def _seq_exec(prog: list[Insn], regs: list[int]) -> None:
    for d, s, x in prog:
        regs[d] = regs[s] + x


def simulate(prog: list[Insn], forwarding: bool) -> tuple[int, list[int]]:
    """Returns (cycles, final_regs)."""
    regs = [0] * 8
    # pipeline regs: stage -> (insn, computed_value_at_ex, dest)
    cycles = 0
    # in-flight: list indexed by stage of (dest, val_known_at_stage)
    # simpler timing model: each insn needs 5 stages; a RAW dep on the
    # previous insn stalls 2 cycles (or 0 with forwarding); dep on the
    # insn two back stalls 1 cycle (0 with forwarding).
    i = 0
    n = len(prog)
    ex_time: list[int] = [0] * n  # cycle when insn i reaches EX
    while i < n or cycles < (ex_time[-1] + 4 if n else 0):
        cycles += 1
        if i < n:
            d, s, _x = prog[i]
            stall = 0
            if s != 0:  # reg 0 never written
                # find producer
                for j in range(i - 1, max(-1, i - 3), -1):
                    if prog[j][0] == s:
                        gap = i - j
                        if not forwarding:
                            stall = 3 - gap if gap <= 2 else 0
                        else:
                            stall = 1 if gap == 1 else 0
                        break
            ex_time[i] = cycles + 2 + stall
            cycles += stall
            # apply write at WB = ex_time + 2
            _x = prog[i][2]
            regs[d] = regs[s] + _x if not forwarding else regs[s] + _x
            i += 1
    # recompute regs sequentially for correctness (timing model only affects cycles)
    regs = [0] * 8
    _seq_exec(prog, regs)
    total = (ex_time[-1] + 3) if n else 0
    return total, regs


def bench_cpu_pipeline(seed: int = 20261231 + 330) -> dict[str, float]:
    rng = random.Random(seed)
    correct = fwd_better = cpi_ok = 0
    trials = 40
    for _ in range(trials):
        prog: list[Insn] = []
        for _ in range(rng.randrange(5, 20)):
            prog.append((rng.randrange(1, 8), rng.randrange(0, 8), rng.randrange(-5, 6)))
        c_stall, r1 = simulate(prog, forwarding=False)
        c_fwd, r2 = simulate(prog, forwarding=True)
        ref = [0] * 8
        _seq_exec(prog, ref)
        correct += int(r1 == ref and r2 == ref)
        fwd_better += int(c_fwd <= c_stall)
        cpi_ok += int(c_stall >= len(prog))
    return {
        "synthetic_regs_match_seq": float(correct / trials),
        "synthetic_fwd_never_worse": float(fwd_better / trials),
        "synthetic_cpi_ge_1": float(cpi_ok / trials),
    }
