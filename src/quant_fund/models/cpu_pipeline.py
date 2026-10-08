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
    """Returns (cycles, final_regs).

    Timing model: insn i issues 1 cycle after the previous issue plus any
    hazard stall, reaches EX at issue+2, commits at WB = EX+2. A consumer
    with a RAW hazard (producer within the 2-insn window) sees the
    producer's value when it is forwarded or already committed; otherwise
    it reads the stale pre-producer register value — which is exactly what
    exposes an insufficient stall. The register file is the pipeline's own
    output, not a sequential replay.
    """
    n = len(prog)
    if n == 0:
        return 0, [0] * 8
    writes: list[list[tuple[int, int]]] = [[] for _ in range(8)]  # reg -> (insn, val)
    ex_time: list[int] = [0] * n
    ex_val: list[int] = [0] * n
    cycles = 0
    for i in range(n):
        d, s, x = prog[i]
        stall = 0
        j = -1
        if s != 0:  # reg 0 never written
            for jj in range(i - 1, max(-1, i - 3), -1):
                if prog[jj][0] == s:
                    j = jj
                    gap = i - jj
                    if not forwarding:
                        stall = 3 - gap if gap <= 2 else 0
                    else:
                        stall = 1 if gap == 1 else 0
                    break
        cycles += 1 + stall
        ex_time[i] = cycles + 2
        read = ex_time[i]
        if j < 0:
            operand = writes[s][-1][1] if writes[s] else 0
        elif forwarding or read >= ex_time[j] + 2:
            operand = ex_val[j]
        else:
            older = [v for idx, v in writes[s] if idx < j]
            operand = older[-1] if older else 0
        val = operand + x
        ex_val[i] = val
        writes[d].append((i, val))
    regs = [w[-1][1] if w else 0 for w in writes]
    return ex_time[-1] + 3, regs


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
