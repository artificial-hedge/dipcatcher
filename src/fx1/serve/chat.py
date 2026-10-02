"""Serving utilities: receipts-cited chat wrapper around fx-1 backends.

Every fx-1 answer that references research evidence should carry the receipt
hash it rests on. ``cited_complete`` runs the model, validates honesty, and
appends the provenance footer — the serving path enforces in production what
training teaches.
"""

from __future__ import annotations

from typing import Any

from fx1.honesty import validate_fx1_output
from fx1.serve.backends import (
    InferenceBackend,
    SamplingParams,
    ToolCompletion,
    truncate_at_stops,
)


def cited_complete(
    backend: InferenceBackend,
    messages: list[dict[str, Any]],
    *,
    receipt_hashes: list[str] | None = None,
    sampling: SamplingParams | None = None,
) -> str:
    """Complete with honesty validation and provenance footer.

    ``sampling.stop`` truncates the model's text before the gate — the
    shipped bytes are what gets validated — and the provider receives the
    stop list too (backends that honor it save the tokens; the harness
    truncation is the belt that makes the contract hold regardless)."""
    response = backend.complete(messages, sampling=sampling)
    if sampling is not None:
        response = truncate_at_stops(response, sampling.stop)
    validate_fx1_output(response)  # fail-closed on contract violations
    if receipt_hashes:
        footer = (
            "\n\nEvidence: "
            + ", ".join(f"`{h[:16]}…`" for h in receipt_hashes)
            + " — verify with `dipcatcher verify-research`."
        )
        response += footer
    return response


def cited_complete_tools(
    backend: InferenceBackend,
    messages: list[dict[str, Any]],
    *,
    receipt_hashes: list[str] | None = None,
    sampling: SamplingParams | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    parallel_tool_calls: bool | None = None,
) -> ToolCompletion:
    """The tool-calling half of the gated completion contract.

    Same rules as :func:`cited_complete` — ``sampling.stop`` truncates the
    assistant text before the gate, the honesty gate validates whatever
    text shipped, and declared receipts append the provenance footer.
    The gate reads ``content`` only: ``tool_calls[].function.arguments``
    are machine-bound JSON (a calculator needs ``{"sharpe": ...}`` keys a
    headline must never carry), so gating the arguments would punish a
    correct call — the *text* is the claim surface and it stays gated.

    A backend without ``complete_with_tools`` fails closed
    (``NotImplementedError`` → 501), never a silent drop of the tool
    context."""
    fn = getattr(backend, "complete_with_tools", None)
    if fn is None:
        raise NotImplementedError(f"backend {type(backend).__name__} has no tool-calling channel")
    result: ToolCompletion = fn(
        messages,
        sampling=sampling,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    content = result.content
    if content is not None and sampling is not None:
        content = truncate_at_stops(content, sampling.stop)
    if content:
        validate_fx1_output(content)  # fail-closed on contract violations
    if receipt_hashes:
        footer = (
            "\n\nEvidence: "
            + ", ".join(f"`{h[:16]}…`" for h in receipt_hashes)
            + " — verify with `dipcatcher verify-research`."
        )
        content = (content or "") + footer
    if content == result.content:
        return result
    return ToolCompletion(
        content=content,
        tool_calls=result.tool_calls,
        finish_reason=result.finish_reason,
    )
