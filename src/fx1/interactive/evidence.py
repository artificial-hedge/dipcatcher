"""Evidence-class gate for dipcatcher chat and synthesis summaries.

``fx1.honesty.validate_fx1_output`` remains the lexical honesty gate.
This module does not replace it and does not weaken it. A summary may
keep a verification claim only when that line names a receipt path the
caller already placed in ``verified_receipts``. Nothing here is semantic
proof of the science. The function does not touch the network or the
filesystem; the path sets passed by the caller are the run record.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["EvidenceError", "ground_summary"]

# Case-insensitive verification claims. Longer phrases are first so an
# overlapping span is consumed before a shorter phrase inside it.
_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bindependently\s+verified\b", re.IGNORECASE),
    re.compile(r"\bverification\s+passed\b", re.IGNORECASE),
    re.compile(r"\bverified\s+receipt\b", re.IGNORECASE),
    re.compile(r"\breceipt\s+verified\b", re.IGNORECASE),
    re.compile(r"\bconfirmed\s+by\s+receipt\b", re.IGNORECASE),
    re.compile(r"\bverified\s*=\s*true\b", re.IGNORECASE),
    # Bare words last so a longer phrase is consumed first. Lookbehind keeps
    # "unverified" and "unproven" from matching.
    re.compile(r"(?<![A-Za-z])proven\b", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])verified\b", re.IGNORECASE),
)
# Zero-width and homoglyph disguises are folded before the claim scan.
_CONFUSABLES = str.maketrans(
    {
        "\u0430": "a",
        "\u0410": "a",
        "\u0435": "e",
        "\u0415": "e",
        "\u043e": "o",
        "\u041e": "o",
        "\u0440": "p",
        "\u0420": "p",
        "\u0441": "c",
        "\u0421": "c",
        "\u0443": "y",
        "\u0423": "y",
        "\u0445": "x",
        "\u0425": "x",
        "\u0456": "i",
        "\u0406": "i",
        "\u0455": "s",
        "\u0501": "d",
        "\u0475": "v",
        "\u03bd": "v",
    }
)
_LOOSE_VERIFIED = re.compile(
    r"(?<![A-Za-z])v[\W_]{0,3}e[\W_]{0,3}r[\W_]{0,3}i[\W_]{0,3}"
    r"f[\W_]{0,3}i[\W_]{0,3}e[\W_]{0,3}d(?![A-Za-z])",
    re.IGNORECASE,
)
_LOOSE_PROVEN = re.compile(
    r"(?<![A-Za-z])p[\W_]{0,3}r[\W_]{0,3}o[\W_]{0,3}v[\W_]{0,3}e[\W_]{0,3}n(?![A-Za-z])",
    re.IGNORECASE,
)

_STUB = "unverified and partial result"
_CANONICAL = "The result is unverified and partial."
# Drop a/an/the that was glued to a claim phrase we just replaced.
_DANGLING_ARTICLE = re.compile(
    r"\b(?:a|an|the)\s+(?=unverified and partial result\b)",
    re.IGNORECASE,
)

# Bare presence is enough. Rewriting a forbidden headline into other words
# would hide the claim; the caller withholds the summary instead.
_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sharpe", re.compile(r"\bsharpe\b", re.IGNORECASE)),
    ("sortino", re.compile(r"\bsortino\b", re.IGNORECASE)),
    ("calmar", re.compile(r"\bcalmar\b", re.IGNORECASE)),
    ("p&l", re.compile(r"\bp\s*&\s*l\b", re.IGNORECASE)),
    ("nav", re.compile(r"\bnav\b", re.IGNORECASE)),
    ("pnl", re.compile(r"\bpnl\b", re.IGNORECASE)),
)

_TOKEN = re.compile(r"\S+")
_LEADING_EDGE = "\"'`([<{*"
_TRAILING_EDGE = "\"'`)]}>*.,;:!?"


class EvidenceError(ValueError):
    """Raised to withhold a summary that carries a forbidden headline token.

    This is an evidence-class failure, not a scientific verdict and not a
    substitute for ``fx1.honesty.validate_fx1_output``.
    """


def ground_summary(
    summary: str,
    *,
    verified_receipts: set[str] | None = None,
    artifact_paths: set[str] | None = None,
    secrets: list[str] | None = None,
) -> str:
    """Return *summary* grounded in the caller's run record.

    An empty summary is returned unchanged. Secret substrings of length 6
    or more are replaced with ``[redacted]``. Verification phrasing is
    rewritten to say the result is unverified and partial unless that line
    names a path in ``verified_receipts``. Bare ``verified`` / ``proven``,
    and zero-width, spaced, or homoglyph disguises of those words, count.
    Cited tokens that end in ``.json`` or contain ``receipts/`` and are
    absent from both path sets
    produce a final ``evidence:`` line. Forbidden headline tokens raise
    :class:`EvidenceError` instead of being paraphrased.
    """
    if summary == "":
        return ""
    _reject_forbidden(summary)
    verified = set(verified_receipts or ())
    known = verified | set(artifact_paths or ())
    redacted = _redact(summary, secrets)
    cited = _cited_paths(redacted)
    rewritten = _rewrite_unverified(redacted, verified)
    return _append_unknown(rewritten, cited, known)


def _reject_forbidden(summary: str) -> None:
    for name, pattern in _FORBIDDEN:
        if pattern.search(summary):
            raise EvidenceError(
                "summary contains forbidden headline token "
                f"{name!r}; withholding rather than paraphrasing it"
            )


def _redact(text: str, secrets: list[str] | None) -> str:
    if not secrets:
        return text
    needles = sorted(
        {item for item in secrets if len(item) >= 6},
        key=lambda item: (-len(item), item),
    )
    if not needles:
        return text
    pattern = re.compile("|".join(re.escape(item) for item in needles))
    return pattern.sub("[redacted]", text)


def _clean_token(raw: str) -> str:
    token = raw
    while token and token[0] in _LEADING_EDGE:
        token = token[1:]
    while token and token[-1] in _TRAILING_EDGE:
        token = token[:-1]
    return token


def _is_cited_path(token: str) -> bool:
    return token.endswith(".json") or "receipts/" in token


def _cited_paths(text: str) -> list[str]:
    found: list[str] = []
    for match in _TOKEN.finditer(text):
        token = _clean_token(match.group(0))
        if token and _is_cited_path(token) and token not in found:
            found.append(token)
    return found


def _claims_verification(line: str) -> bool:
    return any(pattern.search(line) for pattern in _CLAIM_PATTERNS)


def _split_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith(("\n", "\r")):
        return line[:-1], line[-1]
    return line, ""


def _replace_claims(line: str) -> str:
    body, ending = _split_ending(line)
    # One pass consumes every phrase; the stub does not contain them.
    # A second pass catches a span that only became visible after an overlap.
    for _ in range(2):
        if not _claims_verification(body):
            break
        for pattern in _CLAIM_PATTERNS:
            body = pattern.sub(_STUB, body)
    body = _DANGLING_ARTICLE.sub("", body)
    if "the result is unverified and partial" not in body.lower():
        if body and body[-1].isalnum():
            body = f"{body}."
        pad = "" if not body or body[-1].isspace() else " "
        body = f"{body}{pad}{_CANONICAL}"
    return body + ending


def _strip_format(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _fold_disguise(text: str) -> str:
    """Map zero-width, compatibility, and homoglyph disguises onto plain words."""
    folded = unicodedata.normalize("NFKC", _strip_format(text)).translate(_CONFUSABLES)
    folded = _LOOSE_VERIFIED.sub("verified", folded)
    return _LOOSE_PROVEN.sub("proven", folded)


def _disguised(text: str) -> bool:
    if _claims_verification(text):
        return False
    return _claims_verification(_fold_disguise(text))


def _grounded_claim(line: str, verified: set[str]) -> bool:
    """True when every cited path is verified. A disguise is never grounded."""
    if not _claims_verification(line):
        return False
    paths = _cited_paths(line)
    return bool(paths) and all(path in verified for path in paths)


def _rewrite_line(line: str, verified: set[str]) -> str:
    disguised = _disguised(line)
    if not _claims_verification(line) and not disguised:
        return line
    # Every cited receipt on the line must be in the verified set. An
    # artifact path alone does not make a verification claim true.
    if _grounded_claim(line, verified):
        return line
    target = line if _claims_verification(line) else _fold_disguise(line)
    return _replace_claims(target)


def _rewrite_unverified(text: str, verified: set[str]) -> str:
    if text == "":
        return ""
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    return "".join(_rewrite_line(line, verified) for line in lines)


def _append_unknown(text: str, cited: list[str], known: set[str]) -> str:
    unknown = [path for path in cited if path not in known]
    if not unknown:
        return text
    notes = "\n".join(f"evidence: cited path not in the run record: {path}" for path in unknown)
    return f"{text.rstrip('\r\n')}\n{notes}"
