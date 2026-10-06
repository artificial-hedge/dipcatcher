"""dataqual_audit — data-quality ``audit_*`` operation contract battery.

The twelve ``skills.audit_*`` operations are the harness's data-integrity
sentinels: they *describe* violations in supplied tables — never repair,
never impute, never reorder. Every one of these must fail closed on
ambiguous input (strict models, bounded diagnostics, explicit unassessed
statuses), because a silent pass here launders bad data into downstream
sealed claims. These probes pin the contracts:

- *audit_bar_integrity* — per-cell failure codes (missing / nonfinite
  string vs numeric / nonnumeric / nonpositive / negative volume),
  inverted-range suppression of envelope checks, bounded diagnostics.
- *audit_duplicate_keys* — composite-key grouping with JSON scalar
  semantics (``1`` = ``1.0``, ``True`` ≠ ``1``, ``"1"`` ≠ ``1``),
  reject/equal/distinct null policies, bounded examples.
- *audit_missingness* — absent/null/empty-string counting, optional
  empty-string-is-missing, supplied-order run tracking, union-of-names
  column scope, unassessed states for empty inputs.
- *audit_monotonic_sequences* — input-order adjacency (never sorted),
  reversals and whole-group duplicate clocks (timezone-equivalent
  instants are the same clock), direction + strict knobs, singleton
  groups unassessed.
- *audit_cross_field_contracts* — type-aware equality (bool ≠ number),
  ordering restricted to numbers, missing-before-null precedence,
  null fail/skip/compare policies, unique rule ids.
- *audit_missingness_association* — 2×2 contingencies, Jaccard, phi
  null on constant indicators.
- *audit_panel_gaps* — per-security elapsed-time grids anchored at each
  security's first observation: missing spans, duplicates, out-of-order
  rows, off-grid rows.
- *audit_point_in_time* — availability by the decision clock always
  required; completed-event and ingestion-lineage checks opt-in.
- *audit_referential_integrity* — invalid/duplicate parents, child
  matched/ambiguous/unmatched verdicts, null policies, compatible-vs-
  distinct numeric keys.
- *audit_revision_conflicts* — same (security, event, revision) groups:
  field-level conflicts vs exact-duplicate-only, variants-first
  diagnostic row selection.
- *audit_schema_drift* — added/removed fields, non-null type /
  optionality / nullability changes, bool distinct from numbers, empty
  tables unassessed.
- *audit_source_coverage* — expected-pair coverage with latest-event
  selection, availability/future exclusions, staleness by chosen clock,
  unexpected-pair reporting.

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect it found. In-process and deterministic; no served app, no
network, no workspace files.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pydantic

from fx1.operations import (
    audit_bar_integrity,
    audit_cross_field_contracts,
    audit_duplicate_keys,
    audit_missingness,
    audit_missingness_association,
    audit_monotonic_sequences,
    audit_panel_gaps,
    audit_point_in_time,
    audit_referential_integrity,
    audit_revision_conflicts,
    audit_schema_drift,
    audit_source_coverage,
)
from fx1.operations.base import Operation, OperationContext

__all__ = ["dataqual_audit", "dataqual_audit_bench"]

_T0 = "2024-01-01T00:00:00+00:00"
_T1 = "2024-01-01T00:01:00+00:00"
_T2 = "2024-01-01T00:02:00+00:00"
_T3 = "2024-01-01T00:03:00+00:00"
_DECISION = "2024-01-02T00:00:00+00:00"


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


_DT0 = _dt(_T0)
_DT1 = _dt(_T1)
_DT2 = _dt(_T2)
_DT3 = _dt(_T3)
_DECISION_DT = _dt(_DECISION)
_STALE_EVENT = _dt("2023-01-01T00:00:00+00:00")
_STALE_AVAIL = _dt("2023-01-01T00:01:00+00:00")
_LATE = _dt("2024-01-03T00:00:00+00:00")
_FUTURE = _dt("2024-02-01T00:00:00+00:00")


def _ctx() -> OperationContext:
    # These audits never read the workspace; a real (validated) root keeps
    # the call shape honest anyway.
    return OperationContext(workspace_root=Path(tempfile.gettempdir()))


def _refuses(model: Any, **kwargs: Any) -> bool:
    try:
        model(**kwargs)
    except (pydantic.ValidationError, ValueError):
        return True
    return False


def _probe_descriptors() -> dict[str, bool]:
    out: dict[str, bool] = {}
    ops: tuple[Operation[Any, Any], ...] = (
        audit_bar_integrity.OPERATION,
        audit_duplicate_keys.OPERATION,
        audit_missingness.OPERATION,
        audit_monotonic_sequences.OPERATION,
        audit_cross_field_contracts.OPERATION,
        audit_missingness_association.OPERATION,
        audit_panel_gaps.OPERATION,
        audit_point_in_time.OPERATION,
        audit_referential_integrity.OPERATION,
        audit_revision_conflicts.OPERATION,
        audit_schema_drift.OPERATION,
        audit_source_coverage.OPERATION,
    )
    out["dq_all_skills_kind"] = all(o.kind == "skill" for o in ops)
    out["dq_ids_skills_ns"] = all(o.id.startswith("skills.audit_") for o in ops)
    out["dq_schemas_module_local"] = all(
        o.input_model.__module__ == o.handler.__module__ == o.output_model.__module__ for o in ops
    )
    return out


def _probe_bar_integrity() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_bar_integrity
    ctx = _ctx()

    def bar(**kw: Any) -> Any:
        base: dict[str, Any] = {
            "open": 10.0,
            "high": 12.0,
            "low": 9.0,
            "close": 11.0,
            "volume": 5.0,
        }
        base.update(kw)
        return mod.Bar(**base)

    clean = mod.execute(mod.Input(bars=[bar()]), ctx)
    out["bi_clean_passes"] = (
        clean.passed is True and clean.valid_rows == 1 and clean.violation_count == 0
    )
    r = mod.execute(
        mod.Input(
            bars=[
                bar(open=None),
                bar(high="nan"),
                bar(low="not-a-number"),
                bar(open=-1.0),
                bar(volume=-3.0),
                bar(low=13.0),  # low > high → inverted
                bar(close=15.0),  # close above high → envelope
            ]
        ),
        ctx,
    )
    codes = {f.code for f in r.diagnostics}
    out["bi_missing_code"] = "missing_value" in codes
    out["bi_nonfinite_string_code"] = "nonfinite_value" in codes
    out["bi_nonnumeric_code"] = "nonnumeric_value" in codes
    out["bi_nonpositive_code"] = "nonpositive_price" in codes
    out["bi_negative_volume_code"] = "negative_volume" in codes
    inv = [f for f in r.diagnostics if f.code == "inverted_range"]
    out["bi_inverted_on_high_field"] = len(inv) == 1 and inv[0].field == "high"
    env = [f for f in r.diagnostics if f.code == "outside_envelope"]
    out["bi_envelope_on_close"] = len(env) == 1 and env[0].field == "close"
    # inverted range suppresses envelope checks for that row (row 5 only)
    out["bi_inverted_suppresses_envelope"] = (
        all(f.row_index != 5 for f in env) and r.invalid_rows == 7
    )
    capped = mod.execute(
        mod.Input(
            bars=[bar(open=None), bar(high=None), bar(low=None), bar(volume=None)],
            max_diagnostics=2,
        ),
        ctx,
    )
    out["bi_diagnostics_bounded"] = (
        len(capped.diagnostics) == 2
        and capped.violation_count == 4
        and capped.omitted_diagnostics == 2
    )
    out["bi_counts_sorted"] = list(r.violation_counts) == sorted(r.violation_counts)
    out["bi_reject_empty"] = _refuses(mod.Input, bars=[])
    out["bi_reject_bad_cell_type"] = _refuses(
        mod.Bar, open={"nested": 1}, high=1.0, low=1.0, close=1.0, volume=1.0
    )
    out["bi_extra_forbid"] = _refuses(mod.Input, bars=[bar()], bogus=1)
    return out


def _probe_duplicate_keys() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_duplicate_keys
    ctx = _ctx()
    unique = mod.execute(mod.Input(rows=[{"k": "a"}, {"k": "b"}], keys=["k"]), ctx)
    out["dk_unique_passes"] = unique.passed is True and unique.duplicate_key_groups == 0
    dup = mod.execute(mod.Input(rows=[{"k": "a"}, {"k": "a"}, {"k": "a"}], keys=["k"]), ctx)
    out["dk_duplicate_counted"] = (
        dup.duplicate_key_groups == 1
        and dup.duplicate_rows == 2
        and dup.rows_in_duplicate_groups == 3
        and dup.distinct_keys == 1
        and dup.passed is False
    )
    out["dk_example_key_values"] = dup.duplicate_examples[0].key_values == {"k": "a"}
    miss = mod.execute(mod.Input(rows=[{"k": "a"}, {"other": 1}], keys=["k"]), ctx)
    out["dk_missing_field_invalid"] = miss.invalid_key_rows == 1 and miss.invalid_key_examples[
        0
    ].missing_fields == ["k"]
    rej = mod.execute(
        mod.Input(rows=[{"k": None}, {"k": None}], keys=["k"], null_policy="reject"),
        ctx,
    )
    out["dk_null_reject_invalid"] = rej.invalid_key_rows == 2 and rej.duplicate_key_groups == 0
    eq = mod.execute(
        mod.Input(rows=[{"k": None}, {"k": None}], keys=["k"], null_policy="equal"),
        ctx,
    )
    out["dk_null_equal_duplicates"] = eq.duplicate_key_groups == 1
    dist = mod.execute(
        mod.Input(rows=[{"k": None}, {"k": None}], keys=["k"], null_policy="distinct"),
        ctx,
    )
    out["dk_null_distinct_unique"] = dist.duplicate_key_groups == 0 and dist.comparable_rows == 2
    # JSON scalar semantics: 1 == 1.0, but True != 1 and "1" != 1
    num = mod.execute(mod.Input(rows=[{"k": 1}, {"k": 1.0}], keys=["k"]), ctx)
    out["dk_int_float_same_key"] = num.duplicate_key_groups == 1
    boolv = mod.execute(mod.Input(rows=[{"k": True}, {"k": 1}], keys=["k"]), ctx)
    out["dk_bool_ne_number"] = boolv.duplicate_key_groups == 0
    strv = mod.execute(mod.Input(rows=[{"k": "1"}, {"k": 1}], keys=["k"]), ctx)
    out["dk_string_ne_number"] = strv.duplicate_key_groups == 0
    out["dk_reject_dup_key_names"] = _refuses(mod.Input, rows=[{"k": 1}], keys=["k", "k"])
    out["dk_reject_wide_row"] = _refuses(
        mod.Input, rows=[{f"f{i}": 1 for i in range(65)}], keys=["f0"]
    )
    many: list[dict[str, Any]] = [{"k": i % 2} for i in range(8)]
    capped = mod.execute(
        mod.Input(
            rows=many,
            keys=["k"],
            max_examples=1,
        ),
        ctx,
    )
    out["dk_examples_bounded"] = (
        len(capped.duplicate_examples) == 1 and capped.omitted_duplicate_groups == 1
    )
    out["dk_extra_forbid"] = _refuses(mod.Input, rows=[{"k": 1}], keys=["k"], bogus=1)
    return out


def _probe_missingness() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_missingness
    ctx = _ctx()
    rows: list[dict[str, Any]] = [
        {"a": 1, "b": None},
        {"a": "", "b": 2},
        {"b": 3},
        {"a": None},
    ]
    r = mod.execute(mod.Input(rows=rows), ctx)
    cols = {c.column: c for c in r.columns}
    out["m_union_columns_sorted"] = [c.column for c in r.columns] == ["a", "b"]
    out["m_absent_counted"] = cols["a"].absent_count == 1 and cols["b"].absent_count == 1
    out["m_null_counted"] = cols["a"].null_count == 1 and cols["b"].null_count == 1
    out["m_empty_string_value_by_default"] = (
        cols["a"].empty_string_count == 1
        and cols["a"].effective_missing_count == 2  # absent+null; "" is a value
    )
    flagged = mod.execute(mod.Input(rows=rows, empty_string_is_missing=True), ctx)
    fcols = {c.column: c for c in flagged.columns}
    out["m_empty_string_optional"] = fcols["a"].effective_missing_count == 3
    # runs in supplied order: column b missing at rows 0? b: None(0), 2(1), 3(2), absent(3)
    out["m_trailing_run"] = cols["b"].trailing_missing_run == 1
    out["m_leading_run"] = cols["b"].leading_missing_run == 1
    out["m_longest_run"] = cols["b"].longest_missing_run == 1
    long_run = mod.execute(mod.Input(rows=[{"x": None}, {"x": None}, {"x": 1}, {"x": None}]), ctx)
    out["m_longest_run_multi"] = (
        long_run.columns[0].longest_missing_run == 2
        and long_run.columns[0].leading_missing_run == 2
        and long_run.columns[0].trailing_missing_run == 1
    )
    out["m_complete_rate"] = r.complete_row_count == 1 and (r.incomplete_row_count == 3)
    empty = mod.execute(mod.Input(rows=[]), ctx)
    out["m_no_observations_unassessed"] = (
        empty.assessment == "no_observations" and empty.no_missing_values is None
    )
    out["m_declared_columns"] = (
        mod.execute(mod.Input(rows=rows, columns=["b"]), ctx).column_count == 1
    )
    out["m_reject_dup_columns"] = _refuses(mod.Input, rows=rows, columns=["a", "a"])
    out["m_extra_forbid"] = _refuses(mod.Input, rows=rows, bogus=1)
    return out


def _probe_monotonic() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_monotonic_sequences
    ctx = _ctx()

    def obs(t: str, v: float, g: str = "g") -> Any:
        return mod.Observation(group=g, event_time=_dt(t), value=v)

    ok = mod.execute(
        mod.Input(observations=[obs(_T0, 1.0), obs(_T1, 2.0), obs(_T2, 3.0)]),
        ctx,
    )
    out["ms_increasing_passes"] = ok.assessment == "passed" and ok.passed is True
    rev = mod.execute(
        mod.Input(observations=[obs(_T0, 1.0), obs(_T2, 2.0), obs(_T1, 3.0)]),
        ctx,
    )
    out["ms_reversal_flagged"] = rev.event_time_reversals == 1 and any(
        d.code == "event_time_reversal" for d in rev.diagnostics
    )
    out["ms_never_reorders"] = rev.groups[0].row_count == 3
    # nonadjacent duplicate clock still counted
    dup = mod.execute(
        mod.Input(observations=[obs(_T0, 1.0), obs(_T1, 2.0), obs(_T0, 3.0)]),
        ctx,
    )
    out["ms_nonadjacent_dup_counted"] = dup.duplicate_event_rows == 1
    # timezone-equivalent instants are the same clock
    tzdup = mod.execute(
        mod.Input(
            observations=[
                obs("2024-01-01T00:00:00+00:00", 1.0),
                obs("2024-01-01T02:00:00+02:00", 2.0),  # same instant
            ]
        ),
        ctx,
    )
    out["ms_tz_equivalent_duplicate"] = tzdup.duplicate_event_rows == 1
    allowed = mod.execute(
        mod.Input(
            observations=[obs(_T0, 1.0), obs(_T0, 2.0)],
            allow_duplicate_event_times=True,
        ),
        ctx,
    )
    out["ms_dup_allowed_not_violation"] = (
        allowed.duplicate_event_rows == 1
        and allowed.violation_count == 0
        and all(not d.is_violation for d in allowed.diagnostics)
    )
    dec = mod.execute(
        mod.Input(
            observations=[obs(_T0, 3.0), obs(_T1, 2.0)],
            direction="decreasing",
        ),
        ctx,
    )
    out["ms_decreasing_direction"] = dec.assessment == "passed"
    wrong = mod.execute(mod.Input(observations=[obs(_T0, 1.0), obs(_T1, 0.5)]), ctx)
    out["ms_numeric_violation"] = (
        wrong.numeric_direction_violations == 1
        and wrong.diagnostics[0].code == "numeric_direction_violation"
    )
    strict = mod.execute(
        mod.Input(observations=[obs(_T0, 1.0), obs(_T1, 1.0)], strict=True),
        ctx,
    )
    out["ms_strict_flags_plateau"] = strict.numeric_direction_violations == 1
    single = mod.execute(mod.Input(observations=[obs(_T0, 1.0)]), ctx)
    out["ms_singleton_unassessed"] = (
        single.assessment == "no_comparable_pairs"
        and single.passed is None
        and single.groups[0].passed is None
    )
    out["ms_reject_naive_time"] = _refuses(
        mod.Observation, group="g", event_time="2024-01-01 00:00", value=1.0
    ) and _refuses(
        mod.Observation,
        group="g",
        event_time=_dt("2024-01-01T00:00:00"),  # naive — must be refused
        value=1.0,
    )
    out["ms_reject_epoch_text"] = _refuses(
        mod.Observation, group="g", event_time="1704067200", value=1.0
    )
    out["ms_extra_forbid"] = _refuses(mod.Input, observations=[obs(_T0, 1.0)], bogus=1)
    return out


def _probe_cross_field() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_cross_field_contracts
    ctx = _ctx()

    def rule(
        rid: str,
        left: str,
        op: Literal["eq", "ne", "lt", "le", "gt", "ge"],
        right: Any,
        **kw: Any,
    ) -> Any:
        operand = (
            mod.FieldOperand(kind="field", field=right)
            if isinstance(right, str) and right.startswith("fld_")
            else mod.LiteralOperand(kind="literal", value=right)
        )
        return mod.Rule(id=rid, left_field=left, operator=op, right=operand, **kw)

    r = mod.execute(
        mod.Input(
            rows=[{"a": 5, "fld_b": 3}],
            rules=[rule("r1", "a", "gt", "fld_b")],
        ),
        ctx,
    )
    out["cf_field_gt_passes"] = r.assessment == "passed" and r.checked_comparisons == 1
    fails = mod.execute(mod.Input(rows=[{"a": 1}], rules=[rule("r1", "a", "gt", 5)]), ctx)
    out["cf_failed_code"] = (
        fails.assessment == "failed" and fails.diagnostics[0].code == "comparison_failed"
    )
    bool_eq = mod.execute(mod.Input(rows=[{"a": True}], rules=[rule("r1", "a", "eq", 1)]), ctx)
    out["cf_bool_ne_number"] = (
        bool_eq.failed_comparisons == 1 and bool_eq.diagnostics[0].code == "comparison_failed"
    )
    int_float = mod.execute(mod.Input(rows=[{"a": 1}], rules=[rule("r1", "a", "eq", 1.0)]), ctx)
    out["cf_compatible_int_float_eq"] = int_float.checked_comparisons == 1 and (
        int_float.failed_comparisons == 0
    )
    distinct = mod.execute(
        mod.Input(
            rows=[{"a": 1}],
            rules=[rule("r1", "a", "eq", 1.0)],
            numeric_policy="distinct",
        ),
        ctx,
    )
    out["cf_distinct_int_float_ne"] = distinct.failed_comparisons == 1
    str_order = mod.execute(mod.Input(rows=[{"a": "x"}], rules=[rule("r1", "a", "lt", "zzz")]), ctx)
    out["cf_string_order_incompatible"] = str_order.diagnostics[0].code == "incompatible_types"
    miss_fail = mod.execute(mod.Input(rows=[{"a": 1}], rules=[rule("r1", "gone", "eq", 1)]), ctx)
    out["cf_missing_fails"] = miss_fail.diagnostics[0].code == "missing_field"
    miss_skip = mod.execute(
        mod.Input(
            rows=[{"a": 1}],
            rules=[rule("r1", "gone", "eq", 1, missing_policy="skip")],
        ),
        ctx,
    )
    out["cf_missing_skip_unassessed"] = (
        miss_skip.skipped_comparisons == 1
        and miss_skip.checked_comparisons == 0
        and miss_skip.assessment == "no_assessed_comparisons"
    )
    null_fail = mod.execute(mod.Input(rows=[{"a": None}], rules=[rule("r1", "a", "eq", 1)]), ctx)
    out["cf_null_fails"] = null_fail.diagnostics[0].code == "null_operand"
    null_cmp = mod.execute(
        mod.Input(
            rows=[{"a": None}],
            rules=[rule("r1", "a", "eq", None, null_policy="compare")],
        ),
        ctx,
    )
    out["cf_null_compare_eq"] = null_cmp.checked_comparisons == 1 and (
        null_cmp.failed_comparisons == 0
    )
    null_ne = mod.execute(
        mod.Input(
            rows=[{"a": None}],
            rules=[rule("r1", "a", "ne", 1, null_policy="compare")],
        ),
        ctx,
    )
    out["cf_null_ne_value_passes"] = null_ne.failed_comparisons == 0
    out["cf_reject_dup_rule_ids"] = _refuses(
        mod.Input,
        rows=[{"a": 1}],
        rules=[rule("r1", "a", "eq", 1), rule("r1", "a", "eq", 2)],
    )
    out["cf_extra_forbid"] = _refuses(
        mod.Input, rows=[{"a": 1}], rules=[rule("r1", "a", "eq", 1)], bogus=1
    )
    return out


def _probe_association() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_missingness_association
    ctx = _ctx()
    rows: list[dict[str, Any]] = [
        {"a": None, "b": None},
        {"a": None, "b": 1},
        {"a": 1, "b": None},
        {"a": 1, "b": 1},
    ]
    r = mod.execute(mod.Input(rows=rows, columns=["a", "b"]), ctx)
    pair = r.associations[0]
    out["ma_contingency"] = (
        pair.both_missing == 1
        and pair.left_missing_only == 1
        and pair.right_missing_only == 1
        and pair.neither_missing == 1
    )
    out["ma_jaccard"] = abs((pair.jaccard or 0.0) - 1.0 / 3.0) < 1e-12
    out["ma_phi_zero_balanced"] = pair.phi is not None and abs(pair.phi) < 1e-9
    const = mod.execute(
        mod.Input(rows=rows, columns=["a", "c"]),
        ctx,  # c absent everywhere
    )
    out["ma_constant_phi_null"] = const.associations[0].phi is None
    out["ma_joint_rate"] = abs((pair.joint_missing_rate or 0.0) - 0.25) < 1e-12
    out["ma_reject_one_column"] = _refuses(mod.Input, rows=rows, columns=["a"])
    out["ma_reject_dup_columns"] = _refuses(mod.Input, rows=rows, columns=["a", "a"])
    out["ma_extra_forbid"] = _refuses(mod.Input, rows=rows, columns=["a", "b"], bogus=1)
    return out


def _probe_panel_gaps() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_panel_gaps
    ctx = _ctx()

    def obs(sec: str, t: str) -> Any:
        return mod.Observation(security_id=sec, event_time=_dt(t))

    dense = mod.execute(
        mod.Input(
            observations=[
                obs("s1", "2024-01-01T00:00:00+00:00"),
                obs("s1", "2024-01-01T00:01:00+00:00"),
                obs("s1", "2024-01-01T00:02:00+00:00"),
            ],
            interval_seconds=60,
        ),
        ctx,
    )
    out["pg_dense_passes"] = dense.passed is True and dense.missing_grid_points == 0
    gappy = mod.execute(
        mod.Input(
            observations=[
                obs("s1", "2024-01-01T00:00:00+00:00"),
                obs("s1", "2024-01-01T00:03:00+00:00"),
            ],
            interval_seconds=60,
        ),
        ctx,
    )
    span = [d for d in gappy.diagnostics if d.code == "missing_grid_span"]
    out["pg_missing_span"] = (
        gappy.missing_grid_points == 2
        and len(span) == 1
        and span[0].missing_points == 2
        and span[0].first_missing_time is not None
        and span[0].last_missing_time is not None
    )
    dup = mod.execute(
        mod.Input(
            observations=[obs("s1", _T0), obs("s1", _T0)],
            interval_seconds=60,
        ),
        ctx,
    )
    out["pg_duplicate_time"] = dup.duplicate_rows == 1 and any(
        d.code == "duplicate_time" for d in dup.diagnostics
    )
    reorder = mod.execute(
        mod.Input(
            observations=[obs("s1", _T1), obs("s1", _T0)],
            interval_seconds=60,
        ),
        ctx,
    )
    out["pg_out_of_order"] = reorder.out_of_order_rows == 1 and any(
        d.code == "out_of_order" for d in reorder.diagnostics
    )
    off = mod.execute(
        mod.Input(
            observations=[obs("s1", _T0), obs("s1", "2024-01-01T00:01:30+00:00")],
            interval_seconds=60,
        ),
        ctx,
    )
    out["pg_off_grid"] = off.off_grid_rows == 1 and any(
        d.code == "off_grid" for d in off.diagnostics
    )
    per_sec = mod.execute(
        mod.Input(
            observations=[
                obs("a", _T0),
                obs("b", _T2),  # b anchors at its own first obs — no gap
                obs("b", _T3),
            ],
            interval_seconds=60,
        ),
        ctx,
    )
    out["pg_per_security_anchor"] = per_sec.security_count == 2 and per_sec.missing_grid_points == 0
    out["pg_reject_interval_zero"] = _refuses(
        mod.Input, observations=[obs("s1", _T0)], interval_seconds=0
    )
    out["pg_extra_forbid"] = _refuses(
        mod.Input, observations=[obs("s1", _T0)], interval_seconds=60, bogus=1
    )
    return out


def _probe_point_in_time() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_point_in_time
    ctx = _ctx()

    def obs(et: str, at: str, it: str | None = None) -> Any:
        return mod.Observation(
            event_time=_dt(et),
            available_time=_dt(at),
            ingested_time=_dt(it) if it is not None else None,
        )

    ok = mod.execute(
        mod.Input(observations=[obs(_T0, _T1, _T2)], decision_time=_DECISION_DT),
        ctx,
    )
    out["pt_clean_passes"] = ok.passed is True
    late_avail = mod.execute(
        mod.Input(
            observations=[obs(_T0, "2024-01-03T00:00:00+00:00")],
            decision_time=_DECISION_DT,
        ),
        ctx,
    )
    out["pt_available_after_decision"] = (
        late_avail.diagnostics[0].code == "available_after_decision" and late_avail.passed is False
    )
    # future announced events allowed by default
    future_ok = mod.execute(
        mod.Input(
            observations=[obs("2024-02-01T00:00:00+00:00", _T0)],
            decision_time=_DECISION_DT,
        ),
        ctx,
    )
    out["pt_future_event_allowed"] = future_ok.passed is True
    future_strict = mod.execute(
        mod.Input(
            observations=[obs("2024-02-01T00:00:00+00:00", _T0)],
            decision_time=_DECISION_DT,
            require_completed_events=True,
        ),
        ctx,
    )
    out["pt_completed_event_required"] = any(
        d.code == "event_after_decision" for d in future_strict.diagnostics
    )
    pre_avail = mod.execute(
        mod.Input(
            observations=[obs(_T1, _T0)],  # available BEFORE the event
            decision_time=_DECISION_DT,
            require_completed_events=True,
        ),
        ctx,
    )
    out["pt_available_before_event"] = any(
        d.code == "available_before_event" for d in pre_avail.diagnostics
    )
    no_ingest = mod.execute(
        mod.Input(
            observations=[obs(_T0, _T1)],
            decision_time=_DECISION_DT,
            require_ingestion=True,
        ),
        ctx,
    )
    out["pt_missing_ingestion"] = any(d.code == "missing_ingestion" for d in no_ingest.diagnostics)
    early_ingest = mod.execute(
        mod.Input(
            observations=[obs(_T0, _T2, _T1)],  # ingested before available
            decision_time=_DECISION_DT,
        ),
        ctx,
    )
    out["pt_ingested_before_availability"] = any(
        d.code == "ingested_before_availability" for d in early_ingest.diagnostics
    )
    late_ingest = mod.execute(
        mod.Input(
            observations=[obs(_T0, _T1, "2024-01-03T00:00:00+00:00")],
            decision_time=_DECISION_DT,
            require_ingestion_by_decision=True,
        ),
        ctx,
    )
    out["pt_ingested_after_decision"] = any(
        d.code == "ingested_after_decision" for d in late_ingest.diagnostics
    )
    out["pt_flags_echoed"] = (
        ok.completed_events_required is False
        and ok.ingestion_required is False
        and no_ingest.ingestion_required is True
    )
    out["pt_extra_forbid"] = _refuses(
        mod.Input, observations=[obs(_T0, _T1)], decision_time=_DECISION_DT, bogus=1
    )
    return out


def _probe_referential() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_referential_integrity
    ctx = _ctx()

    def _rows(*rows: dict[str, Any]) -> list[dict[str, Any]]:
        return list(rows)

    matched = mod.execute(
        mod.Input(
            parent_rows=_rows({"id": 1}),
            child_rows=_rows({"pid": 1}),
            parent_keys=["id"],
            child_keys=["pid"],
        ),
        ctx,
    )
    out["ri_matched"] = matched.matched_child_rows == 1 and matched.passed is True
    unmatched = mod.execute(
        mod.Input(
            parent_rows=_rows({"id": 1}),
            child_rows=_rows({"pid": 9}),
            parent_keys=["id"],
            child_keys=["pid"],
        ),
        ctx,
    )
    out["ri_unmatched"] = (
        unmatched.unmatched_child_rows == 1
        and unmatched.diagnostics[0].code == "unmatched_child_reference"
        and unmatched.passed is False
    )
    ambiguous = mod.execute(
        mod.Input(
            parent_rows=_rows({"id": 1}, {"id": 1}),
            child_rows=_rows({"pid": 1}),
            parent_keys=["id"],
            child_keys=["pid"],
        ),
        ctx,
    )
    out["ri_ambiguous"] = (
        ambiguous.ambiguous_child_rows == 1
        and ambiguous.duplicate_parent_key_groups == 1
        and any(d.code == "duplicate_parent_key" for d in ambiguous.diagnostics)
    )
    bad_parent = mod.execute(
        mod.Input(
            parent_rows=_rows({"other": 1}),
            child_rows=_rows({"pid": 1}),
            parent_keys=["id"],
            child_keys=["pid"],
        ),
        ctx,
    )
    out["ri_invalid_parent"] = (
        bad_parent.invalid_parent_rows == 1
        and bad_parent.diagnostics[0].code == "invalid_parent_key"
    )
    bad_child = mod.execute(
        mod.Input(
            parent_rows=_rows({"id": 1}),
            child_rows=_rows({"other": 1}),
            parent_keys=["id"],
            child_keys=["pid"],
        ),
        ctx,
    )
    out["ri_invalid_child"] = bad_child.invalid_child_rows == 1
    null_child = mod.execute(
        mod.Input(
            parent_rows=_rows({"id": 1}),
            child_rows=_rows({"pid": None}),
            parent_keys=["id"],
            child_keys=["pid"],
            child_null_policy="ignore",
        ),
        ctx,
    )
    out["ri_null_child_ignored"] = (
        null_child.ignored_null_child_rows == 1
        and null_child.comparable_child_rows == 0
        and null_child.invalid_child_rows == 0
    )
    # numeric policy: int parent key vs float child key
    compat = mod.execute(
        mod.Input(
            parent_rows=_rows({"id": 1}),
            child_rows=_rows({"pid": 1.0}),
            parent_keys=["id"],
            child_keys=["pid"],
            numeric_policy="compatible",
        ),
        ctx,
    )
    out["ri_compatible_numeric_match"] = compat.matched_child_rows == 1
    dist = mod.execute(
        mod.Input(
            parent_rows=_rows({"id": 1}),
            child_rows=_rows({"pid": 1.0}),
            parent_keys=["id"],
            child_keys=["pid"],
            numeric_policy="distinct",
        ),
        ctx,
    )
    out["ri_distinct_numeric_unmatched"] = dist.unmatched_child_rows == 1
    out["ri_reject_len_mismatch"] = _refuses(
        mod.Input,
        parent_rows=_rows({"id": 1}),
        child_rows=_rows({"a": 1, "b": 2}),
        parent_keys=["id"],
        child_keys=["a", "b"],
    )
    out["ri_extra_forbid"] = _refuses(
        mod.Input,
        parent_rows=_rows({"id": 1}),
        child_rows=_rows({"pid": 1}),
        parent_keys=["id"],
        child_keys=["pid"],
        bogus=1,
    )
    return out


def _probe_revision_conflicts() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_revision_conflicts
    ctx = _ctx()

    def rec(src: str, avail: str, values: dict[str, Any], rev: str = "r1") -> Any:
        return mod.Record(
            security_id="s1",
            event_time=_DT0,
            revision_id=rev,
            available_time=_dt(avail),
            source=src,
            values=values,
        )

    same = mod.execute(
        mod.Input(records=[rec("a", _T1, {"x": 1}), rec("a", _T1, {"x": 1})]),
        ctx,
    )
    out["rc_exact_dup_only"] = (
        same.conflicting_revision_groups == 0
        and same.exact_duplicate_only_groups == 1
        and same.findings[0].status == "exact_duplicates_only"
        and same.conflict_free is True
    )
    diff = mod.execute(
        mod.Input(records=[rec("a", _T1, {"x": 1}), rec("a", _T1, {"x": 2})]),
        ctx,
    )
    conflict = diff.findings[0]
    out["rc_value_conflict"] = (
        conflict.status == "conflicting_revision"
        and conflict.conflict_fields == ["values"]
        and diff.conflicting_revision_groups == 1
        and diff.conflict_free is False
    )
    multisrc = mod.execute(
        mod.Input(records=[rec("a", _T1, {"x": 1}), rec("b", _T1, {"x": 1})]),
        ctx,
    )
    out["rc_source_conflict"] = "source" in multisrc.findings[0].conflict_fields
    multiavail = mod.execute(
        mod.Input(records=[rec("a", _T1, {"x": 1}), rec("a", _T2, {"x": 1})]),
        ctx,
    )
    out["rc_availability_conflict"] = "available_time" in multiavail.findings[0].conflict_fields
    separate = mod.execute(
        mod.Input(records=[rec("a", _T1, {"x": 1}, rev="r1"), rec("b", _T1, {"x": 2}, rev="r2")]),
        ctx,
    )
    out["rc_distinct_revisions_separate"] = (
        separate.conflicting_revision_groups == 0
        and separate.revision_group_count == 2
        and separate.unique_and_consistent is True
    )
    single = mod.execute(mod.Input(records=[rec("a", _T1, {"x": 1})]), ctx)
    out["rc_singleton_no_finding"] = (
        single.conflicting_revision_groups == 0 and len(single.findings) == 0
    )
    out["rc_value_comparison_declared"] = diff.value_comparison == "canonical_json_type_sensitive"
    out["rc_extra_forbid"] = _refuses(mod.Input, records=[rec("a", _T1, {"x": 1})], bogus=1)
    return out


def _probe_schema_drift() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_schema_drift
    ctx = _ctx()

    def _rows(*rows: dict[str, Any]) -> list[dict[str, Any]]:
        return list(rows)

    same = mod.execute(
        mod.Input(
            reference_rows=_rows({"a": 1, "b": "x"}),
            candidate_rows=_rows({"a": 2, "b": "y"}),
        ),
        ctx,
    )
    out["sd_equal_schemas"] = same.schema_equal is True and same.assessment == "compared"
    added = mod.execute(
        mod.Input(
            reference_rows=_rows({"a": 1}),
            candidate_rows=_rows({"a": 1, "b": 2}),
        ),
        ctx,
    )
    out["sd_added_field"] = (
        added.changed_field_count == 1
        and added.changes[0].name == "b"
        and "added_field" in added.changes[0].changes
    )
    removed = mod.execute(
        mod.Input(
            reference_rows=_rows({"a": 1, "b": 2}),
            candidate_rows=_rows({"a": 1}),
        ),
        ctx,
    )
    out["sd_removed_field"] = "removed_field" in removed.changes[0].changes
    typed = mod.execute(
        mod.Input(
            reference_rows=_rows({"a": 1}),
            candidate_rows=_rows({"a": "x"}),
        ),
        ctx,
    )
    out["sd_type_change"] = "non_null_types_changed" in typed.changes[0].changes
    optional = mod.execute(
        mod.Input(
            reference_rows=_rows({"a": 1}, {"a": 2}),
            candidate_rows=_rows({"a": 1}, {}),
        ),
        ctx,
    )
    out["sd_optionality_change"] = "optionality_changed" in optional.changes[0].changes
    nullable = mod.execute(
        mod.Input(
            reference_rows=_rows({"a": 1}),
            candidate_rows=_rows({"a": None}),
        ),
        ctx,
    )
    out["sd_nullability_change"] = (
        "nullability_changed" in nullable.changes[0].changes
        or "non_null_types_changed" in nullable.changes[0].changes
    )
    bool_num = mod.execute(
        mod.Input(
            reference_rows=_rows({"a": True}),
            candidate_rows=_rows({"a": 1}),
        ),
        ctx,
    )
    out["sd_bool_ne_number"] = bool_num.schema_equal is False
    empty = mod.execute(mod.Input(reference_rows=[], candidate_rows=_rows({"a": 1})), ctx)
    out["sd_empty_unassessed"] = (
        empty.assessment == "insufficient_observations" and empty.schema_equal is None
    )
    out["sd_extra_forbid"] = _refuses(
        mod.Input, reference_rows=_rows({"a": 1}), candidate_rows=_rows({"a": 1}), bogus=1
    )
    return out


def _probe_source_coverage() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = audit_source_coverage
    ctx = _ctx()

    def obs(src: str, sec: str, et: datetime, at: datetime) -> Any:
        return mod.Observation(source=src, security_id=sec, event_time=et, available_time=at)

    def pair(src: str, sec: str) -> Any:
        return mod.Pair(source=src, security_id=sec)

    covered = mod.execute(
        mod.Input(
            expected_pairs=[pair("s", "a")],
            observations=[obs("s", "a", _DT0, _DT1)],
            decision_time=_DECISION_DT,
        ),
        ctx,
    )
    out["sc_covered"] = (
        covered.covered_expected_pairs == 1
        and covered.assessment == "covered"
        and covered.complete_expected_coverage is True
    )
    missing = mod.execute(
        mod.Input(
            expected_pairs=[pair("s", "a")],
            observations=[],
            decision_time=_DECISION_DT,
        ),
        ctx,
    )
    out["sc_missing"] = missing.missing_expected_pairs == 1 and missing.assessment == "incomplete"
    stale = mod.execute(
        mod.Input(
            expected_pairs=[pair("s", "a")],
            observations=[obs("s", "a", _STALE_EVENT, _STALE_AVAIL)],
            decision_time=_DECISION_DT,
            max_age_seconds=1000,
        ),
        ctx,
    )
    out["sc_stale"] = stale.stale_expected_pairs == 1 and stale.assessment == "incomplete"
    unavailable = mod.execute(
        mod.Input(
            expected_pairs=[pair("s", "a")],
            observations=[obs("s", "a", _DT0, _LATE)],
            decision_time=_DECISION_DT,
        ),
        ctx,
    )
    out["sc_unavailable_excluded"] = (
        unavailable.unavailable_records == 1 and unavailable.missing_expected_pairs == 1
    )
    future = mod.execute(
        mod.Input(
            expected_pairs=[pair("s", "a")],
            observations=[obs("s", "a", _FUTURE, _DT1)],
            decision_time=_DECISION_DT,
        ),
        ctx,
    )
    out["sc_future_event_excluded"] = (
        future.future_event_records_excluded == 1 and future.missing_expected_pairs == 1
    )
    unexpected = mod.execute(
        mod.Input(
            expected_pairs=[pair("s", "a")],
            observations=[obs("s", "a", _DT0, _DT1), obs("s", "b", _DT0, _DT1)],
            decision_time=_DECISION_DT,
        ),
        ctx,
    )
    out["sc_unexpected_reported"] = unexpected.unexpected_eligible_pair_count == 1
    # latest event wins the selection per pair
    latest = mod.execute(
        mod.Input(
            expected_pairs=[pair("s", "a")],
            observations=[obs("s", "a", _DT0, _DT1), obs("s", "a", _DT2, _DT3)],
            decision_time=_DECISION_DT,
        ),
        ctx,
    )
    sel = latest.expected_pairs[0]
    out["sc_latest_selected"] = sel.selected_row_index == 1 and sel.eligible_records == 2
    out["sc_selection_policy_declared"] = (
        latest.selection_policy == "latest_event_then_availability_then_earliest_input_row"
    )
    out["sc_extra_forbid"] = _refuses(
        mod.Input,
        expected_pairs=[],
        observations=[],
        decision_time=_DECISION_DT,
        bogus=1,
    )
    return out


def dataqual_audit() -> dict[str, bool]:
    """Every data-quality audit contract as literal booleans."""
    out: dict[str, bool] = {}
    out.update(_probe_descriptors())
    out.update(_probe_bar_integrity())
    out.update(_probe_duplicate_keys())
    out.update(_probe_missingness())
    out.update(_probe_monotonic())
    out.update(_probe_cross_field())
    out.update(_probe_association())
    out.update(_probe_panel_gaps())
    out.update(_probe_point_in_time())
    out.update(_probe_referential())
    out.update(_probe_revision_conflicts())
    out.update(_probe_schema_drift())
    out.update(_probe_source_coverage())
    return out


def dataqual_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = dataqual_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "dataqual_audit",
        "schema": "dataqual_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process execute() calls; no served app, no workspace I/O",
            "not_verified": [
                "registry dispatch wiring",
                "file-backed readers/inspectors (separate battery)",
                "operations/base workspace containment (separate battery)",
            ],
        },
        "interpretation": (
            "Data-quality sentinels hold: bar integrity emits the full "
            "failure-code taxonomy with inverted-range suppression and "
            "bounded diagnostics; duplicate keys use JSON scalar semantics "
            "(1=1.0, bool and string distinct) under explicit null "
            "policies; missingness counts absent/null/optional-empty with "
            "supplied-order runs and unassessed empty states; monotonic "
            "sequences never reorder input and count timezone-equivalent "
            "duplicates group-wide; cross-field rules keep bool distinct "
            "from numbers, restrict ordering to numbers, and honor "
            "missing/null fail-skip policies; associations report 2×2 "
            "contingencies with null phi on constants; panel gaps anchor "
            "each security's own grid and flag spans/dupes/reordering/"
            "off-grid rows; point-in-time always requires availability "
            "by decision with opt-in completed-event and ingestion "
            "checks; referential integrity classifies invalid/duplicate "
            "parents and matched/ambiguous/unmatched children under the "
            "declared numeric policy; revision conflicts distinguish "
            "field-level conflicts from exact duplicates; schema drift "
            "reports added/removed/type/optionality/nullability changes "
            "with bool-number distinctness; source coverage selects "
            "latest-event-per-pair, excludes unavailable/future records, "
            "and reports stale/missing/unexpected pairs."
            if ok
            else f"DATAQUAL AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(dataqual_audit_bench(), indent=2, sort_keys=True))
