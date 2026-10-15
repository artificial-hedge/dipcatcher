"""Long options stay grep-able when Typer colors help and errors."""

from __future__ import annotations

import os
import subprocess
import sys

from rich.console import Console
from rich.text import Text


def test_error_sentences_are_not_split_around_flags() -> None:
    from rich.text import Text
    from typer.rich_utils import highlighter

    from quant_fund.cli.option_highlight import install_stable_option_highlighting

    install_stable_option_highlighting()
    sentence = "--receipt-version must be 1 or 2"
    text = Text(f"Invalid value: {sentence}")
    highlighter.highlight(text)
    assert sentence in text.plain
    # No style span may start or end inside the operator-facing phrase.
    start = text.plain.index(sentence)
    end = start + len(sentence)
    for span in text.spans:
        assert span.start <= start or span.start >= end
        assert span.end <= start or span.end >= end


def test_long_options_stay_contiguous_under_color() -> None:
    from typer.rich_utils import highlighter

    from quant_fund.cli.option_highlight import install_stable_option_highlighting

    install_stable_option_highlighting()
    text = Text(" --spec --receipt-version --horizons --blocks -h ")
    highlighter.highlight(text)
    console = Console(force_terminal=True, color_system="standard", width=120)
    with console.capture() as captured:
        console.print(text, end="")
    rendered = captured.get()
    for token in ("--spec", "--receipt-version", "--horizons", "--blocks", "-h"):
        assert token in rendered, repr(rendered)


def test_github_actions_color_keeps_reality_help_flags() -> None:
    """CI sets GITHUB_ACTIONS, which forces a color terminal inside Typer."""
    env = os.environ.copy()
    env.pop("NO_COLOR", None)
    env.pop("FORCE_COLOR", None)
    env["GITHUB_ACTIONS"] = "true"
    env["TERM"] = "xterm-256color"
    script = (
        "from typer.testing import CliRunner\n"
        "from quant_fund.cli.main import app\n"
        "result = CliRunner().invoke(app, ['reality', 'sweep', '--help'])\n"
        "text = result.output\n"
        "assert result.exit_code == 0, text\n"
        "assert '--spec' in text, text\n"
        "assert '--cache-dir' in text, text\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd="/workspace",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
