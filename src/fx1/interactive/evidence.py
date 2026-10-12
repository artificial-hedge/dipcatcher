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

__all__ = ["EvidenceError", "ground_summary"]

# Case-insensitive verification claims. Longer phrases are first so an
# overlapping span is consumed before a shorter phrase inside it.
_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bindependently\s+verified\b", re.IGNORECASE),
    re.compile(r"\bverification\s+passed\b", re.IGNORECASE),
    re.compile(r"\bverified\s+receipt\b", re.IGNORECASE),
    re.compile(r"\breceipt\s+verified\b", re.IGNORECASE),
    re.compile(r"\bverified\s*=\s*true\b", re.IGNORECASE),
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
    names a path in ``verified_receipts``. Cited tokens that end in
    ``.json`` or contain ``receipts/`` and are absent from both path sets
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


def _rewrite_line(line: str, verified: set[str]) -> str:
    if not _claims_verification(line):
        return line
    paths = _cited_paths(line)
    # Every cited receipt on the line must be in the verified set. An
    # artifact path alone does not make a verification claim true.
    if paths and all(path in verified for path in paths):
        return line
    return _replace_claims(line)


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
