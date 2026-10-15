"""Keep long options contiguous when help and errors are colored.

Typer's ``OptionHighlighter`` treats ``-`` as a non-word character, so the
short-switch pattern ``-\\w+`` matches the tail of a long option. Rich then
emits a reset between the two dashes: ``--spec`` becomes ``-`` + ``-spec``
and is no longer a contiguous substring. ``grep -- --spec`` and the CLI
contract tests fail whenever color is on, including GitHub Actions, which
forces a terminal.

The corrected patterns refuse a dash that continues another dash. A short
switch (``-h``) and a long option (``--spec``) each stay one span.
"""

from __future__ import annotations

_INSTALLED = False

# A switch is one dash, not the second dash of a long option.
_SHORT_SWITCH = r"(?<![\w-])(?P<switch>-\w+)(?![A-Za-z0-9])"
_LONG_OPTION = r"(?<![\w-])(?P<option>--[\w-]+)(?![A-Za-z0-9])"
_NEGATIVE_SWITCH = r"(?<![\w-])(?P<negative_switch>-\w+)(?![A-Za-z0-9])"
_NEGATIVE_OPTION = r"(?<![\w-])(?P<negative_option>--[\w-]+)(?![A-Za-z0-9])"


def _is_option_listing(plain: str) -> bool:
    """True when the whole string is options, types, or a usage label.

    An error sentence such as ``--receipt-version must be 1 or 2`` must stay
    plain. Highlighting only the flag inserts a reset before the rest of the
    sentence, so the operator-facing phrase is no longer one substring.
    """
    stripped = plain.strip()
    if not stripped or " " not in stripped:
        return True
    for word in stripped.split():
        if word.startswith("-") or word.startswith("<") or word.startswith("Usage"):
            continue
        if word in {"Usage:", "[OPTIONS]", "[ARGS]..."}:
            continue
        return False
    return True


def install_stable_option_highlighting() -> None:
    """Patch Typer's highlighters so option tokens survive ANSI styling."""
    global _INSTALLED
    if _INSTALLED:
        return
    from typer import rich_utils

    rich_utils.OptionHighlighter.highlights = [
        _SHORT_SWITCH,
        _LONG_OPTION,
        r"(?P<types>\<[^\>]+\>)",
        r"(?P<usage>Usage: )",
    ]
    rich_utils.NegativeOptionHighlighter.highlights = [
        _NEGATIVE_SWITCH,
        _NEGATIVE_OPTION,
    ]
    from rich.text import Text

    original = rich_utils.OptionHighlighter.highlight

    def highlight(self: rich_utils.OptionHighlighter, text: Text) -> None:
        if not _is_option_listing(text.plain):
            return
        original(self, text)

    rich_utils.OptionHighlighter.highlight = highlight  # type: ignore[assignment]
    _INSTALLED = True
