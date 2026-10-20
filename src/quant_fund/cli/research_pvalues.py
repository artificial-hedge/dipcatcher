"""P-value harvest for the online-FDR receipt replay.

``research_cmds`` re-exports ``_harvest_p_values`` under the same name.
The helper stays here so that command module can shrink without changing
what the replay selects.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _harvest_p_values(findings: Iterable[Any]) -> list[float]:
    """Pull the ``stat == "p"`` values out of one harvest_findings result.

    Findings are dicts on the corpus-inference lane; attribute access is
    tolerated so the command survives a dataclass-shaped finding.
    """
    out: list[float] = []
    for finding in findings:
        if isinstance(finding, Mapping):
            stat, value = finding.get("stat"), finding.get("value")
        else:
            stat = getattr(finding, "stat", None)
            value = getattr(finding, "value", None)
        if stat == "p" and isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out
