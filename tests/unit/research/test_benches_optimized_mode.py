"""Optimized-mode (``python -O``) regression probes for the generated
``benches_*`` validator adapters.

The validators exist to keep the honesty contract fail-closed under optimized
execution: when ``python -O`` strips ``assert``, a ``sharpe`` / ``pnl`` /
``nav`` key, a non-``synthetic_*`` key, or a non-finite value emitted by a
bench must still be stopped. These probes execute each distinct
``_finite_blob`` signature in a real ``python -O`` subprocess.

Two authored contracts are covered, per the module's own docstring:

- ``raise`` — tainted blobs raise ``ValueError`` (the dominant family).
- ``filter`` — the ``(mapped)`` variant returns ``{}`` unless every value is
  finite (fail-closed: tainted content never reaches the caller).

An ``assert``-based version of either contract would silently pass tainted
blobs through under ``-O``; each probe fails loudly in that case.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# (module, signature-shape, contract): one representative per distinct
# _finite_blob signature observed across the generated family.
_PROBES = [
    ("quant_fund.research.benches_w100", "named", "raise"),
    ("quant_fund.research.benches_w17", "mapped", "filter"),
    ("quant_fund.research.benches_w1000", "blob", "raise"),
    ("quant_fund.research.benches_w1236", "blob", "raise"),
    ("quant_fund.research.benches_w1257", "blob", "raise"),
    ("quant_fund.research.benches_w1258", "blob", "raise"),
    ("quant_fund.research.benches_w1259", "blob", "raise"),
]

_RAISE_SNIPPET = """
import sys
from {module} import _finite_blob

forbidden = {{"sharpe": 1.0}}
nonfinite = {{"synthetic_x": float("nan")}}
clean = {{"synthetic_x": 0.5}}

def call(blob):
    if {named!r}:
        return _finite_blob("bench_probe", blob)
    return _finite_blob(blob)

stopped = 0
for bad in (forbidden, nonfinite):
    try:
        call(bad)
    except Exception:
        stopped += 1

out = call(clean)
if stopped != 2 or out != clean:
    print("VALIDATION_STRIPPED", file=sys.stderr)
    sys.exit(2)
print("ok")
"""

_FILTER_SNIPPET = """
import sys
from {module} import _finite_blob

tainted = {{"x_metric": float("nan")}}
clean = {{"x_metric": 0.5}}

out_tainted = _finite_blob(tainted)
if out_tainted != {{}}:
    print("VALIDATION_STRIPPED", file=sys.stderr)
    sys.exit(2)
assert _finite_blob(clean) == clean
print("ok")
"""


@pytest.mark.parametrize(("module", "shape", "contract"), _PROBES)
def test_validator_survives_optimized_mode(module: str, shape: str, contract: str) -> None:
    if contract == "filter":
        snippet = _FILTER_SNIPPET.format(module=module)
    else:
        snippet = _RAISE_SNIPPET.format(module=module, named=shape == "named")
    proc = subprocess.run(
        [sys.executable, "-O", "-c", snippet],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{module}._finite_blob failed under `python -O`:\n{proc.stderr}"
    assert "ok" in proc.stdout
