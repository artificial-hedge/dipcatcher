"""Unit tests for quant_fund.models.debruijn_assemble."""

from __future__ import annotations

import os
import subprocess
import sys

from quant_fund.models.debruijn_assemble import assemble


def test_assemble_recovers_superstring() -> None:
    out = assemble(["ACG", "CGT", "GTA"], 3)
    assert out is not None
    assert all(km in out for km in ["ACG", "CGT", "GTA"])


def test_deterministic_across_hash_seeds() -> None:
    """The Eulerian start used to come from set-iteration order —
    PYTHONHASHSEED-dependent. A balanced graph with several valid
    circuits must assemble identically under any hash seed."""
    kms = ["AA", "AB", "BA", "AC", "CA"]  # balanced circuit with choices
    outs = set()
    for seed in ("0", "1", "7"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                "from quant_fund.models.debruijn_assemble import assemble;"
                "print(assemble(['AA','AB','BA','AC','CA'],2))",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outs.add(r.stdout.strip())
    assert len(outs) == 1
    assert assemble(kms, 2) is not None
