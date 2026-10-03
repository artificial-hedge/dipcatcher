"""harness_audit — adversarial probes on the fx-1 tool boundary.

One real hole fixed: ``extra_args`` carried ``--config`` verbatim past the
configs/ allowlist — ``run("doctor", extra_args=["--config",
"/etc/passwd"])`` reached the runner with an unconstrained path. The kwarg
is now the only config path; ``--config``/``--config=`` inside extra_args
is refused fail-closed.

Pinned edges: unknown commands raise KeyError naming the allowed set;
``api`` and ``lab`` are absent from the registry by design; configs/ paths
resolve through the allowlist (absolute outside, relative .., and
symlink escapes all raise); timeout bounds are pydantic-enforced; the
runner receives the exact argv and exit codes propagate to ``ok``.
Sealed ``harness_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["harness_audit", "harness_audit_bench"]


def _raises(fn: Any) -> str:
    try:
        fn()
    except (ValueError, KeyError) as exc:
        return f"raise:{type(exc).__name__}"
    return "no-raise"


def harness_audit() -> dict[str, Any]:
    from fx1.harness import Harness, HarnessRole

    calls: list[list[str]] = []

    def recorder(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        calls.append(list(argv))
        return 0, "ok", ""

    h = Harness(runner=recorder)
    registry_names = {c.name for c in h.list_commands()}

    out: dict[str, Any] = {
        "n_commands": len(registry_names),
        "api_absent": "api" not in registry_names,
        "lab_absent": "lab" not in registry_names,
        "unknown_raises": _raises(lambda: h.run("definitely-not-a-command")),
        "all_roles_populated": all(h.list_commands(role) for role in HarnessRole),
    }

    # config allowlist: real configs/ under a scratch cwd
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        (tdir / "configs").mkdir()
        (tdir / "configs" / "ok.yaml").write_text("a: 1")
        (tdir / "outside.yaml").write_text("a: 1")
        link = tdir / "configs" / "evil.yaml"
        try:
            link.symlink_to(tdir / "outside.yaml")
            linked = True
        except OSError:
            linked = False
        os.chdir(tdir)
        try:
            h.run("doctor", config=tdir / "configs" / "ok.yaml")
            out["inside_config_argv"] = calls[-1]
            out["outside_config_raises"] = _raises(
                lambda: h.run("doctor", config=tdir / "outside.yaml")
            )
            out["relative_escape_raises"] = _raises(
                lambda: h.run("doctor", config=Path("../outside.yaml"))
            )
            if linked:
                out["symlink_escape_raises"] = _raises(lambda: h.run("doctor", config=link))
            # the fixed hole: --config smuggled through extra_args
            out["extra_args_config_raises"] = _raises(
                lambda: h.run("doctor", extra_args=["--config", "/etc/passwd"])
            )
            out["extra_args_config_eq_raises"] = _raises(
                lambda: h.run("doctor", extra_args=["--config=/etc/passwd"])
            )
            benign = h.run("doctor", extra_args=["--seed", "7"])
            out["benign_extra_args_reach_runner"] = calls[-1]
            out["exit_code_propagates"] = benign.ok
        finally:
            os.chdir(cwd)

    # timeout bounds are model-validated
    from fx1.harness import HarnessCommand

    out["timeout_bound_raises"] = _raises(
        lambda: HarnessCommand(
            name="x", role=HarnessRole.EVALUATION, argv=["x"], timeout_s=0, description=""
        )
    )
    return out


def harness_audit_bench() -> dict[str, Any]:
    r = harness_audit()
    ok = (
        r["api_absent"] is True
        and r["lab_absent"] is True
        and r["unknown_raises"] == "raise:KeyError"
        and r["all_roles_populated"] is True
        and "--config" in r["inside_config_argv"]
        and r["outside_config_raises"] == "raise:ValueError"
        and r["relative_escape_raises"] == "raise:ValueError"
        and r.get("symlink_escape_raises") == "raise:ValueError"
        and r["extra_args_config_raises"] == "raise:ValueError"
        and r["extra_args_config_eq_raises"] == "raise:ValueError"
        and r["benign_extra_args_reach_runner"][-2:] == ["--seed", "7"]
        and r["exit_code_propagates"] is True
        and r["timeout_bound_raises"].startswith("raise:")
    )
    payload: dict[str, Any] = {
        "kind": "harness_audit",
        "schema": "harness_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Harness boundary holds: unknown commands and api/lab are "
            "unreachable, configs/ containment survives absolute/relative/"
            "symlink escapes, and the extra_args --config bypass is closed."
            if ok
            else f"HARNESS DEFECT: {r}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
