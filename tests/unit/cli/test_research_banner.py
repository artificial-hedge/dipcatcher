"""Pin the SYNTHETIC notebook line moved out of the research command."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from quant_fund.cli.research_banner import _echo_synthetic_marker
from quant_fund.cli.research_cmds import _echo_synthetic_marker as exported


def test_echo_synthetic_marker_matches_the_notebook_line() -> None:
    assert exported is _echo_synthetic_marker
    printed = io.StringIO()
    with redirect_stdout(printed):
        _echo_synthetic_marker(True)
        _echo_synthetic_marker(False)
    assert printed.getvalue() == "SYNTHETIC\n"
