"""One-process timing of a disabled span versus an enabled span.

The numbers are printed and, when the artifact directory exists, saved.
They are not a benchmark claim.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

from quant_fund.observe.overhead import measure


def test_disabled_span_stays_cheap_and_enabled_stays_local() -> None:
    result = measure(iterations=2000)
    assert result["iterations"] == 2000
    assert result["direct_ns_per_call"] > 0
    assert result["disabled_ns_per_call"] > 0
    assert result["enabled_ns_per_call"] > 0
    assert result["otel_endpoint_cleared"] is True
    # Disabled work is the flag check plus a context manager. 100µs is loose
    # on a shared CI runner and still far from an SDK startup.
    assert result["disabled_ns_per_call"] < 100_000
    # Enabled work is in-process only (no socket). A few milliseconds would
    # mean the span path started doing I/O.
    assert result["enabled_ns_per_call"] < 5_000_000
    print(json.dumps(result, sort_keys=True))
    destination = os.environ.get(
        "OBSERVE_OVERHEAD_OUT", "/opt/cursor/artifacts/observe_overhead.json"
    )
    target = Path(destination)
    # The directory can exist on a mount that still refuses a new file.
    # The timing assertions above are the contract; the dump is optional.
    if target.parent.is_dir():
        with contextlib.suppress(OSError):
            target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
