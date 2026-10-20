"""Corpus-wide replay sweep — every committed replay carrier, one verdict.

``dipcatcher replay`` proves a single carrier; this module sweeps the
whole carrier directory (``data/manifests/replay/*.json``) and seals a
``replay_coverage.v1`` receipt: per-carrier verdicts, the corpus's
reproducibility counts, and an aggregate ``coverage`` fraction — the
single number answering "what fraction of the declared lane corpus
reproduces byte-for-byte under this checkout".

Skip classes are honest, never folded into failures:

- ``inputs_not_verified`` → ``skipped`` (the lane could not run here —
  e.g. tape absent on CI). Honest, never counted as a failure. A skipped
  row promotes nothing.
- ``fail`` means the lane *ran* and its produced bytes diverged, or it
  exited non-zero, or its manifest is dangerous (committed-overwrite
  guard, cwd escape, carrier seal invalid) — the class the gate exists
  for — or the replay matched but the live corpus no longer holds the
  receipt bytes the carrier says it reproduces (below).

**Reproduced-receipt binding.** A carrier's ``reproduces[]`` entries name
committed receipts (``receipt``, repo-root relative) and pin their bytes
(``committed_sha256``). Re-executing argv proves the *artifact* digests;
it says nothing about whether the live receipt at ``reproduces[].receipt``
is still the evidence that was reproduced. So each entry is bound to the
live tree **before** argv runs: the path must stay inside ``root`` (no
absolute paths, no ``..``), the file must exist, its sha256 must equal
``committed_sha256``, and the very same bytes must pass the production
verifier (``verify_receipt_bytes``). A row is ``pass`` only when the
replay passed AND every entry binds; a matched replay over a drifted,
missing, invalid, or out-of-root receipt is ``fail`` with the binding
errors in ``note`` (``reproduced_receipt_drift:<path>``,
``reproduced_receipt_missing:<path>``, ``reproduced_receipt_invalid:<path>``,
``reproduced_receipt_path_invalid:<path>``, ``reproduced_receipt_pin_invalid:<path>``).
Skipped and failed rows record the same binding errors so drift stays
visible. A carrier with an empty ``reproduces`` list (e.g. a lane whose
digest-named outputs are its artifacts) may still pass — it promotes no
receipt, and its row says so with ``n_reproduces: 0``.

Bodies written before binding existed carry no ``n_binding_errors``; the
contract checks binding fields only when present (such legacy ``pass``
rows attest execution, never binding).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeGuard

from quant_fund.research.replay_proof import run_replay

REPLAY_COVERAGE_SCHEMA = "replay_coverage.v1"

__all__ = [
    "REPLAY_COVERAGE_SCHEMA",
    "replay_coverage_contract_errors",
    "run_replay_sweep",
]

_SKIP_CLASSES = {"inputs_not_verified"}


def _path_escapes(rel: str) -> bool:
    """A ``reproduces[].receipt`` must be a plain repo-relative path."""
    posix = PurePosixPath(rel.replace("\\", "/"))
    windows = PureWindowsPath(rel)
    return posix.is_absolute() or bool(windows.drive) or bool(windows.root) or ".." in posix.parts


def _bind_reproduced(
    entry: object, index: int, root_path: Path
) -> tuple[dict[str, Any], list[str]]:
    """Bind one ``reproduces[]`` entry to the live receipt bytes under ``root``.

    Returns ``(record, errors)``. ``record["bound"]`` is true iff the path
    is in-root, the file exists, its digest equals the pinned
    ``committed_sha256``, and those same bytes verify. The digest and the
    verifier read one byte string, so the two checks cannot disagree about
    which file they saw.
    """
    from quant_fund.research.receipt_v2 import verify_receipt_bytes

    if not isinstance(entry, Mapping):
        label = f"reproduces[{index}]"
        return (
            {"receipt": label, "bound": False},
            [f"reproduced_receipt_path_invalid:{label}"],
        )
    rel = entry.get("receipt")
    pinned = entry.get("committed_sha256")
    label = rel if isinstance(rel, str) and rel.strip() else f"reproduces[{index}]"
    record: dict[str, Any] = {
        "receipt": label,
        "committed_sha256": pinned if isinstance(pinned, str) else None,
        "live_sha256": None,
        "receipt_valid": None,
        "bound": False,
    }
    if "claims_equal" in entry:
        record["claims_equal"] = entry.get("claims_equal")
    if not isinstance(rel, str) or not rel.strip() or _path_escapes(rel):
        return record, [f"reproduced_receipt_path_invalid:{label}"]
    resolved = (root_path / rel).resolve()
    if not resolved.is_relative_to(root_path):  # symlink escape
        return record, [f"reproduced_receipt_path_invalid:{label}"]
    errors: list[str] = []
    if not _is_sha256(pinned):
        errors.append(f"reproduced_receipt_pin_invalid:{label}")
    try:
        data = resolved.read_bytes() if resolved.is_file() else None
    except OSError:
        data = None
    if data is None:
        errors.append(f"reproduced_receipt_missing:{label}")
        return record, errors
    live = hashlib.sha256(data).hexdigest()
    record["live_sha256"] = live
    if _is_sha256(pinned) and live != pinned:
        errors.append(f"reproduced_receipt_drift:{label}")
    verification = verify_receipt_bytes(data, resolved)
    record["receipt_valid"] = bool(verification["valid"])
    if not verification["valid"]:
        record["receipt_errors"] = [str(error) for error in verification["errors"]]
        errors.append(f"reproduced_receipt_invalid:{label}")
    record["bound"] = not errors
    return record, errors


def _bind_carrier(reproduces: object, root_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Bind every ``reproduces[]`` entry; a non-list declaration is an error."""
    if reproduces is None:
        return [], []
    if not isinstance(reproduces, list):
        return [], ["reproduces_malformed"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, entry in enumerate(reproduces):
        record, entry_errors = _bind_reproduced(entry, index, root_path)
        records.append(record)
        errors.extend(entry_errors)
    return records, errors


def _executed_row(rel: str, proof: Mapping[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    """The sweep row for a carrier whose replay ran (or was skipped) — binding gates ``pass``.

    A skipped row stays ``skipped`` (it promotes nothing; its binding errors
    stay visible). A passed execution over any binding error is ``fail``
    with the errors in ``note``.
    """
    binding_errors: list[str] = binding["binding_errors"]
    skipped = proof.get("execution_skipped")
    execution_verdict = str(proof["verdict"])
    if skipped is not None:
        verdict = "skipped"
    elif execution_verdict == "pass" and binding_errors:
        verdict = "fail"
    else:
        verdict = execution_verdict
    row: dict[str, Any] = {
        "carrier": rel,
        "verdict": verdict,
        "execution_verdict": execution_verdict,
        "all_match": proof["all_match"],
        "exit_code": proof["exit_code"],
        **binding,
    }
    if skipped is not None:
        row["skipped_reason"] = skipped
    elif verdict == "fail" and execution_verdict == "pass":
        row["note"] = ";".join(binding_errors)
    return row


def _headline_claim_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    """Executed claim total, executed equals, and skipped claims.

    A skipped carrier never ran. Folding its ``claims_equal`` into the
    headline would report equality for a lane that produced no bytes.
    """
    executed_total = 0
    executed_equal = 0
    skipped_claims = 0
    for row in rows:
        reproduces = row.get("reproduces")
        if not isinstance(reproduces, list):
            continue
        n_claims = len(reproduces)
        if row.get("verdict") == "skipped":
            skipped_claims += n_claims
        else:
            executed_total += n_claims
            # Only executed carriers promote claims_equal into the headline.
            executed_equal += sum(
                1
                for item in reproduces
                if isinstance(item, Mapping) and item.get("claims_equal") is True
            )
    return executed_total, executed_equal, skipped_claims


def run_replay_sweep(
    carriers_dir: Path | str,
    *,
    root: Path | str,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Replay every ``replay_manifest.v1`` carrier under ``carriers_dir``.

    Returns the ``replay_coverage.v1`` body (unsealed — callers seal via
    ``seal_receipt``/``wrap_receipt_v2``). Raises ``ValueError`` when the
    directory is absent or carries no well-formed carriers — an empty
    sweep attests nothing.

    Each carrier must itself verify (its ``receipt_sha256`` self-seal under
    the production verifier) before its argv runs, and each of its
    ``reproduces[]`` entries is bound to the live receipt bytes before
    execution (see the module docstring); a row is ``pass`` only when the
    replay passed and every entry binds.
    """
    from quant_fund.research.receipt_v2 import verify_receipt_bytes

    carriers_path = Path(carriers_dir)
    root_path = Path(root).resolve()
    if not carriers_path.is_dir():
        raise ValueError(f"carriers directory does not exist: {carriers_path}")

    rows: list[dict[str, Any]] = []
    for carrier in sorted(carriers_path.glob("*.json")):
        try:
            carrier_bytes = carrier.read_bytes()
            carrier_body = json.loads(carrier_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"carrier unreadable: {carrier.name}: {exc}") from exc
        if not isinstance(carrier_body, dict) or carrier_body.get("schema") != "replay_manifest.v1":
            continue

        try:
            rel = str(carrier.resolve().relative_to(root_path))
        except ValueError:
            rel = carrier.name
        # Bind before argv runs: the binding attests the corpus as found,
        # never bytes the replay itself may have rewritten.
        bindings, binding_errors = _bind_carrier(carrier_body.get("reproduces"), root_path)
        binding: dict[str, Any] = {
            "n_reproduces": len(bindings),
            "reproduces": bindings,
            "binding_errors": binding_errors,
        }
        # A carrier whose own seal does not verify is a corrupt declaration:
        # its argv must not run and its pins promote nothing.
        carrier_check = verify_receipt_bytes(carrier_bytes, carrier)
        if not carrier_check["valid"]:
            rows.append(
                {
                    "carrier": rel,
                    "verdict": "fail",
                    "note": "carrier_invalid:" + ",".join(str(e) for e in carrier_check["errors"]),
                    **binding,
                }
            )
            continue
        try:
            proof = run_replay(carrier, root=root_path, timeout_s=timeout_s)
        except ValueError as exc:
            rows.append(
                {
                    "carrier": rel,
                    "verdict": "fail",
                    "note": f"manifest_malformed:{exc}",
                    **binding,
                }
            )
            continue

        rows.append(_executed_row(rel, proof, binding))

    if not rows:
        raise ValueError(f"no replay_manifest.v1 carriers under {carriers_path}")

    n_pass = sum(1 for row in rows if row["verdict"] == "pass")
    n_fail = sum(1 for row in rows if row["verdict"] == "fail")
    n_skipped = sum(1 for row in rows if row["verdict"] == "skipped")
    n_claims_total, n_claims_byte_equal, n_claims_skipped = _headline_claim_counts(rows)
    n_executed = n_pass + n_fail
    coverage = (n_pass / n_executed) if n_executed else 0.0
    verdict = "pass" if n_fail == 0 and n_pass > 0 else "fail"

    return {
        "schema": REPLAY_COVERAGE_SCHEMA,
        "kind": "replay_coverage",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "carriers_dir": str(carriers_path),
        "rows": rows,
        "n_carriers": n_executed,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skipped": n_skipped,
        "n_declared": len(rows),
        "coverage": coverage,
        "n_claims_total": n_claims_total,
        "n_claims_byte_equal": n_claims_byte_equal,
        "n_claims_skipped": n_claims_skipped,
        "claims_byte_equal_frac": (n_claims_byte_equal / n_claims_total)
        if n_claims_total
        else None,
        "n_binding_errors": sum(len(row["binding_errors"]) for row in rows),
        "n_reproduces_bound": sum(
            1 for row in rows for record in row["reproduces"] if record.get("bound") is True
        ),
        "timeout_s": float(timeout_s),
        "verdict": verdict,
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_error_list(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


_ROW_BINDING_KEYS = ("binding_errors", "n_reproduces", "reproduces")


def _row_records(row: Mapping[str, Any], prefix: str) -> tuple[list[Any], list[str]]:
    """A row's ``reproduces`` records, checked against its ``n_reproduces``."""
    errors: list[str] = []
    n_reproduces = row.get("n_reproduces")
    if not _is_count(n_reproduces):
        errors.append(f"{prefix}.n_reproduces")
    records = row.get("reproduces")
    if not isinstance(records, list):
        errors.append(f"{prefix}.reproduces")
        return [], errors
    if _is_count(n_reproduces) and len(records) != n_reproduces:
        errors.append(f"{prefix}.n_reproduces")
    return records, errors


def _record_errors(records: list[Any], prefix: str) -> tuple[list[str], int, int]:
    """Per-record checks; returns ``(errors, n_bound, n_unbound)``.

    A bound record must carry equal, well-formed pinned/live digests and
    ``receipt_valid``; a malformed record counts as unbound.
    """
    errors: list[str] = []
    n_bound = 0
    n_unbound = 0
    for position, record in enumerate(records):
        where = f"{prefix}.reproduces[{position}]"
        if not isinstance(record, Mapping) or not isinstance(record.get("bound"), bool):
            errors.append(f"{where}.bound")
            n_unbound += 1
            continue
        if not record["bound"]:
            n_unbound += 1
            continue
        pinned = record.get("committed_sha256")
        live = record.get("live_sha256")
        if not (_is_sha256(pinned) and _is_sha256(live) and pinned == live):
            errors.append(f"{where}.digest")
        if record.get("receipt_valid") is not True:
            errors.append(f"{where}.receipt_valid")
        n_bound += 1
    return errors, n_bound, n_unbound


def _row_binding_errors(
    row: Mapping[str, Any], index: int, verdict: str, binding_declared: bool
) -> tuple[list[str], int, int]:
    """Re-derive one row's reproduced-receipt binding claims.

    Returns ``(errors, n_binding_errors, n_bound)``; ``([], 0, 0)`` for a
    legacy row in a legacy body (no binding anywhere). An unbound record
    must be explained by at least one binding error, and a ``pass`` row
    must carry zero binding errors and only bound records.
    """
    if not (binding_declared or any(key in row for key in _ROW_BINDING_KEYS)):
        return [], 0, 0
    prefix = f"rows[{index}]"
    errors: list[str] = []
    claimed = row.get("binding_errors")
    binding_errors: list[str] = []
    if _is_error_list(claimed):
        binding_errors = claimed
    else:
        errors.append(f"{prefix}.binding_errors")
    records, record_shape_errors = _row_records(row, prefix)
    errors.extend(record_shape_errors)
    record_errors, n_bound, n_unbound = _record_errors(records, prefix)
    errors.extend(record_errors)
    if n_unbound and not binding_errors:
        errors.append(f"{prefix}.binding_errors")
    if verdict == "pass" and (binding_errors or n_unbound):
        errors.append(f"{prefix}.pass_with_binding_errors")
    return errors, len(binding_errors), n_bound


def _pass_row_errors(row: Mapping[str, Any], index: int, verdict: object) -> list[str]:
    """A ``pass`` row must report a clean execution: all_match, exit 0."""
    if verdict != "pass":
        return []
    errors: list[str] = []
    if row.get("all_match") is not True:
        errors.append(f"rows[{index}].all_match")
    exit_code = row.get("exit_code")
    if isinstance(exit_code, bool) or exit_code != 0:
        errors.append(f"rows[{index}].exit_code")
    if "execution_verdict" in row and row["execution_verdict"] != "pass":
        errors.append(f"rows[{index}].execution_verdict")
    return errors


def _binding_total_errors(
    payload: Mapping[str, Any], binding_declared: bool, n_binding_errors: int, n_bound: int
) -> list[str]:
    """Aggregate binding counts must re-derive from the rows when declared."""
    if not binding_declared:
        return []
    errors: list[str] = []
    for field, expected_total in (
        ("n_binding_errors", n_binding_errors),
        ("n_reproduces_bound", n_bound),
    ):
        value = payload.get(field)
        if not _is_count(value) or value != expected_total:
            errors.append(field)
    return errors


def replay_coverage_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Internal consistency for a ``replay_coverage.v1`` body — fail closed.

    Re-derives every count from the carrier rows plus the headline
    ``verdict == ("pass" iff n_fail == 0 and n_pass > 0)`` equivalence, so
    a body lying about its own arithmetic fails under a fresh seal. A
    ``pass`` row must report ``all_match`` and ``exit_code == 0``.

    Reproduced-receipt binding (``n_binding_errors`` plus per-row
    ``binding_errors``/``n_reproduces``/``reproduces``) is optional for
    bodies sealed before binding existed, but checked whenever present: a
    body declaring ``n_binding_errors`` must carry binding on every row,
    ``pass`` rows must have zero binding errors, and the aggregate counts
    must re-derive from the rows.
    """
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    errors: list[str] = []
    if payload.get("schema") != REPLAY_COVERAGE_SCHEMA:
        errors.append("schema")
    if payload.get("research_only") is not True:
        errors.append("research_only")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if payload.get("data_label") != "CORPUS":
        errors.append("data_label")

    binding_declared = "n_binding_errors" in payload or "n_reproduces_bound" in payload
    total_binding_errors = 0
    total_bound = 0
    rows = payload.get("rows")
    row_verdicts: list[str] = []
    if not isinstance(rows, list) or not rows:
        errors.append("rows")
    else:
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append(f"rows[{index}]")
                continue
            verdict = row.get("verdict")
            if verdict not in ("pass", "fail", "skipped"):
                errors.append(f"rows[{index}].verdict")
                verdict = "fail"
            row_verdicts.append(str(verdict))
            carrier = row.get("carrier")
            if not isinstance(carrier, str) or not carrier.strip():
                errors.append(f"rows[{index}].carrier")
            if verdict == "skipped" and row.get("skipped_reason") not in _SKIP_CLASSES:
                errors.append(f"rows[{index}].skipped_reason")
            errors.extend(_pass_row_errors(row, index, verdict))
            row_errors, n_row_errors, n_row_bound = _row_binding_errors(
                row, index, str(verdict), binding_declared
            )
            errors.extend(row_errors)
            total_binding_errors += n_row_errors
            total_bound += n_row_bound

    for field in ("n_carriers", "n_pass", "n_fail", "n_skipped"):
        value = payload.get(field)
        if not _is_count(value):
            errors.append(field)
    n_declared = payload.get("n_declared")
    if n_declared is not None and not _is_count(n_declared):
        errors.append("n_declared")
    errors.extend(
        _binding_total_errors(payload, binding_declared, total_binding_errors, total_bound)
    )
    if row_verdicts and not errors:
        n_rows = len(row_verdicts)
        n_pass = row_verdicts.count("pass")
        n_fail = row_verdicts.count("fail")
        n_skipped = row_verdicts.count("skipped")
        n_executed = n_pass + n_fail
        headline_is_executed = n_declared is not None
        expected_carriers = n_executed if headline_is_executed else n_rows
        if payload.get("n_carriers") != expected_carriers:
            errors.append("n_carriers")
        if headline_is_executed and n_declared != n_rows:
            errors.append("n_declared")
        if payload.get("n_pass") != n_pass:
            errors.append("n_pass")
        if payload.get("n_fail") != n_fail:
            errors.append("n_fail")
        if payload.get("n_skipped") != n_skipped:
            errors.append("n_skipped")
        coverage = payload.get("coverage")
        if headline_is_executed:
            expected_coverage = (n_pass / n_executed) if n_executed else 0.0
        else:
            expected_coverage = n_pass / n_rows
        if (
            not isinstance(coverage, (int, float))
            or isinstance(coverage, bool)
            or not math.isclose(float(coverage), expected_coverage, rel_tol=0, abs_tol=1e-12)
        ):
            errors.append("coverage")
        expected = "pass" if n_fail == 0 and n_pass > 0 else "fail"
        if payload.get("verdict") != expected:
            errors.append("verdict")

    for field in ("n_claims_total", "n_claims_byte_equal"):
        value = payload.get(field)
        if not _is_count(value):
            errors.append(field)
    if "n_claims_skipped" in payload:
        if not _is_count(payload.get("n_claims_skipped")):
            errors.append("n_claims_skipped")
        elif row_verdicts and not errors:
            total, equal, skipped_claims = _headline_claim_counts(rows)  # type: ignore[arg-type]
            if payload.get("n_claims_total") != total:
                errors.append("n_claims_total")
            if payload.get("n_claims_byte_equal") != equal:
                errors.append("n_claims_byte_equal")
            if payload.get("n_claims_skipped") != skipped_claims:
                errors.append("n_claims_skipped")
            frac_expected = (equal / total) if total else None
            frac = payload.get("claims_byte_equal_frac")
            if frac_expected is None:
                if frac is not None:
                    errors.append("claims_byte_equal_frac")
            elif (
                not isinstance(frac, (int, float))
                or isinstance(frac, bool)
                or not math.isclose(float(frac), frac_expected, rel_tol=0, abs_tol=1e-12)
            ):
                errors.append("claims_byte_equal_frac")
    frac = payload.get("claims_byte_equal_frac")
    if frac is not None and (
        not isinstance(frac, (int, float)) or isinstance(frac, bool) or not math.isfinite(frac)
    ):
        errors.append("claims_byte_equal_frac")

    return errors
