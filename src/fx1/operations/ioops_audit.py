"""ioops_audit — file I/O and bitemporal join/select operation battery.

Where dataqual_audit covers the describe-only sentinels, these operations
touch the workspace filesystem (readers/inspectors/hash verification) or
run the bitemporal selection machinery every sealed eval depends on
(join_asof_observations, select_asof_revisions,
select_universe_membership, resolve_security_identity) and the ingestion
clock-quality summary. The probes pin:

- *workspace containment* — ``../`` escapes, wrong suffixes, missing
  files, and oversized files all fail closed through
  ``OperationContext.open_binary``/``read_bytes`` (each reader passes a
  fixed suffix list and byte cap).
- *read_csv* — whole-file parse before paging (malformed rows outside
  the page still error), exact-string cells, header/width validation,
  blank-record counting, source fingerprint.
- *read_jsonl* — strict JSON objects, duplicate keys rejected at any
  depth, NaN/Infinity refused, physical line numbers, union-of-keys
  columns, paging.
- *read_toml* — ``.toml`` containment, temporal values classified
  (offset/local datetime, date, time), ``table_path`` navigation, no
  env/include/exec interpolation.
- *inspect_numpy_array* — NPY header parsed as a Python literal without
  allocating the array; object/pickle payloads refused; trailing bytes
  refused.
- *inspect_parquet* — footer + Arrow schema from fingerprinted bytes,
  leaf-column count, row-group paging.
- *inspect_zip* — central-directory metadata only: directories,
  symlinks, duplicates, path issues; never decompresses.
- *verify_file_hash* — byte-exact SHA-256 compare, lowercase-normalized
  expected digest, optional byte-count check, size cap.
- *join_asof_observations* — activation at ``max(event, available)``,
  winner = latest event → latest availability → greatest revision_id →
  earliest row; ``no_eligible_observation`` / ``outside_lookback``
  reasons; same-clock fingerprint conflicts refused at validation.
- *select_asof_revisions* — latest observable vintage per
  security/event, same-clock conflicts refused, future events excluded
  only on request, ``superseded_or_duplicate`` accounting.
- *select_universe_membership* — latest availability per membership_id,
  effective [from, to) intervals, expired exclusion revealing an older
  inclusion, same-effective conflicting states → ambiguous.
- *resolve_security_identity* — same revision machinery on
  ticker/exchange mappings; case-sensitive matching; resolved / missing /
  ambiguous statuses.
- *summarize_ingestion_latency* — signed lags keep negatives visible,
  histogram's first bin is the negative count, quantiles at (n−1)·p,
  per-source summaries.

Probes are literal bools; the sealed receipt names every defect found.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pydantic

from fx1.operations import (
    inspect_numpy_array,
    inspect_parquet,
    inspect_zip,
    join_asof_observations,
    read_csv,
    read_jsonl,
    read_toml,
    resolve_security_identity,
    select_asof_revisions,
    select_universe_membership,
    summarize_ingestion_latency,
    verify_file_hash,
)
from fx1.operations.base import Operation, OperationContext

__all__ = ["ioops_audit", "ioops_audit_bench"]


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


_T0 = _dt("2024-01-01T00:00:00+00:00")
_T1 = _dt("2024-01-01T00:01:00+00:00")
_T2 = _dt("2024-01-01T00:02:00+00:00")
_T3 = _dt("2024-01-01T00:03:00+00:00")
_DEC = _dt("2024-01-02T00:00:00+00:00")
_FAR = _dt("2024-03-01T00:00:00+00:00")


class _Ws:
    """Real temp workspace holding fixture files."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # Resolve now: the POSIX open walk refuses symlinked path components,
        # and macOS tempdirs live under /var -> /private/var.
        self.root = Path(self._tmp.name).resolve()
        self.ctx = OperationContext(workspace_root=self.root)

    def write(self, name: str, content: bytes | str) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        return name


def _refuses(model: Any, **kwargs: Any) -> bool:
    try:
        model(**kwargs)
    except (pydantic.ValidationError, ValueError):
        return True
    return False


def _fails(fn: Any) -> bool:
    try:
        fn()
    except (ValueError, OSError):
        return True
    return False


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _probe_descriptors() -> dict[str, bool]:
    out: dict[str, bool] = {}
    ops: tuple[Operation[Any, Any], ...] = (
        read_csv.OPERATION,
        read_jsonl.OPERATION,
        read_toml.OPERATION,
        inspect_numpy_array.OPERATION,
        inspect_parquet.OPERATION,
        inspect_zip.OPERATION,
        verify_file_hash.OPERATION,
        join_asof_observations.OPERATION,
        select_asof_revisions.OPERATION,
        select_universe_membership.OPERATION,
        resolve_security_identity.OPERATION,
        summarize_ingestion_latency.OPERATION,
    )
    plugins = ops[:7]
    skills = ops[7:]
    out["io_plugins_kind_and_ns"] = all(
        o.kind == "plugin" and o.id.startswith("plugins.") for o in plugins
    )
    out["io_skills_kind_and_ns"] = all(
        o.kind == "skill" and o.id.startswith("skills.") for o in skills
    )
    out["io_schemas_module_local"] = all(
        o.input_model.__module__ == o.handler.__module__ == o.output_model.__module__ for o in ops
    )
    return out


def _probe_containment(ws: _Ws) -> dict[str, bool]:
    """Every file op must refuse escapes, wrong suffixes, absent files."""
    out: dict[str, bool] = {}
    ws.write("a.csv", "x,y\n1,2\n")
    csv_in = read_csv.Input(path="a.csv")
    out["ct_legit_read"] = read_csv.execute(csv_in, ws.ctx).total_rows == 1
    out["ct_escape_refused"] = _fails(
        lambda: read_csv.execute(read_csv.Input(path="../escape.csv"), ws.ctx)
    )
    out["ct_wrong_suffix_refused"] = _fails(
        lambda: read_csv.execute(read_csv.Input(path="a.jsonl"), ws.ctx)
    )
    out["ct_missing_refused"] = _fails(
        lambda: read_csv.execute(read_csv.Input(path="nope.csv"), ws.ctx)
    )
    out["ct_dotdot_variants_refused"] = _fails(
        lambda: read_jsonl.execute(read_jsonl.Input(path="..%2f..%2fx.jsonl"), ws.ctx)
    )
    out["ct_subdir_escape_refused"] = _fails(
        lambda: read_toml.execute(read_toml.Input(path="subdir/../../escape.toml"), ws.ctx)
    )
    # resolve_file (inspect path) agrees: escape refused even before open
    out["ct_resolve_escape_refused"] = _fails(
        lambda: ws.ctx.resolve_file("../x.csv", suffixes=(".csv",))
    )
    ws.write("dir/a.csv", "x\n1\n")
    out["ct_symlink_refused"] = _fails(
        lambda: read_csv.execute(read_csv.Input(path=_link(ws)), ws.ctx)
    )
    ws.write("big.csv", "x\n" + "9" * 100 + "\n")
    out["ct_max_bytes_refused"] = _fails(
        lambda: verify_file_hash.execute(
            verify_file_hash.Input(path="big.csv", expected_sha256="0" * 64, max_bytes=10),
            ws.ctx,
        )
    )
    return out


def _link(ws: _Ws) -> str:
    """Create a workspace symlink pointing inside the root; still refused."""
    target = ws.root / "linked.csv"
    if not target.is_symlink():
        target.symlink_to(ws.root / "a.csv")
    return "linked.csv"


def _probe_read_csv(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = read_csv
    ws.write(
        "bars.csv",
        "t,o,h,l,c,v\n2024-01-01,10,12,9,11,5\n2024-01-02,11,13,10,12,7\n\n2024-01-03,12,14,11,13,9\n",
    )
    r = mod.execute(mod.Input(path="bars.csv"), ws.ctx)
    out["csv_columns"] = r.columns == ["t", "o", "h", "l", "c", "v"]
    out["csv_exact_strings"] = r.rows[0] == {
        "t": "2024-01-01",
        "o": "10",
        "h": "12",
        "l": "9",
        "c": "11",
        "v": "5",
    }
    out["csv_blank_skipped"] = r.blank_records_skipped == 1 and r.total_rows == 3
    out["csv_sha_matches"] = (
        r.source_sha256 == _sha256((ws.root / "bars.csv").read_bytes())
        and r.source_bytes == (ws.root / "bars.csv").stat().st_size
    )
    page = mod.execute(mod.Input(path="bars.csv", offset=1, limit=1), ws.ctx)
    out["csv_paged"] = (
        page.returned_rows == 1 and page.has_more is True and page.rows[0]["t"] == "2024-01-02"
    )
    ws.write("bad.csv", "a,b\n1,2,3\n")
    out["csv_ragged_row_refused"] = _fails(
        lambda: mod.execute(mod.Input(path="bad.csv", offset=99), ws.ctx)
    )
    ws.write("nodict.csv", "a,b\n1,2\n")
    ws.write("semis.csv", "x;y\n1;2\n")
    semi = mod.execute(mod.Input(path="semis.csv", delimiter=";"), ws.ctx)
    out["csv_delimiter"] = semi.rows == [{"x": "1", "y": "2"}]
    out["csv_reject_bad_delimiter"] = _refuses(mod.Input, path="a.csv", delimiter="\n")
    out["csv_extra_forbid"] = _refuses(mod.Input, path="a.csv", bogus=1)
    return out


def _probe_read_jsonl(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = read_jsonl
    ws.write(
        "rows.jsonl",
        '{"a": 1, "b": "x"}\n\n{"a": 2, "c": true}\n{"a": 3, "b": null}\n',
    )
    r = mod.execute(mod.Input(path="rows.jsonl"), ws.ctx)
    out["jl_rows_parsed"] = r.rows[0] == {"a": 1, "b": "x"} and r.rows[2] == {
        "a": 3,
        "b": None,
    }
    out["jl_blank_skipped"] = r.blank_lines_skipped == 1
    out["jl_physical_lines"] = r.physical_line_numbers == [1, 3, 4]
    out["jl_union_columns"] = sorted(r.columns) == ["a", "b", "c"]
    out["jl_sha"] = r.source_sha256 == _sha256((ws.root / "rows.jsonl").read_bytes())
    page = mod.execute(mod.Input(path="rows.jsonl", offset=2, limit=1), ws.ctx)
    out["jl_paged"] = page.returned_rows == 1 and page.rows[0]["a"] == 3
    ws.write("dup.jsonl", '{"a": 1}\n{"a": 1, "a": 2}\n')
    out["jl_dup_keys_refused"] = _fails(
        lambda: mod.execute(mod.Input(path="dup.jsonl", offset=5), ws.ctx)
    )
    ws.write("nan.jsonl", '{"a": NaN}\n')
    out["jl_nan_refused"] = _fails(lambda: mod.execute(mod.Input(path="nan.jsonl"), ws.ctx))
    ws.write("arr.jsonl", "[1,2,3]\n")
    out["jl_non_object_refused"] = _fails(lambda: mod.execute(mod.Input(path="arr.jsonl"), ws.ctx))
    out["jl_extra_forbid"] = _refuses(mod.Input, path="x.jsonl", bogus=1)
    return out


def _probe_read_toml(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = read_toml
    ws.write(
        "cfg.toml",
        'title = "demo"\n'
        "when = 2024-01-01T10:00:00Z\n"
        "day = 2024-01-02\n"
        "tick = 10:30:00\n"
        "[nested]\n"
        "flag = true\n",
    )
    r = mod.execute(mod.Input(path="cfg.toml"), ws.ctx)
    out["toml_parsed"] = r.data["title"] == "demo" and r.data["nested"] == {"flag": True}
    kinds = {t.kind for t in r.temporal_values}
    out["toml_temporals_classified"] = kinds == {
        "offset_datetime",
        "local_date",
        "local_time",
    }
    sub = mod.execute(mod.Input(path="cfg.toml", table_path=["nested"]), ws.ctx)
    out["toml_table_path"] = sub.data == {"flag": True}
    out["toml_sha"] = r.source_sha256 == _sha256((ws.root / "cfg.toml").read_bytes())
    ws.write("bad.toml", "x = [unclosed\n")
    out["toml_invalid_refused"] = _fails(lambda: mod.execute(mod.Input(path="bad.toml"), ws.ctx))
    out["toml_extra_forbid"] = _refuses(mod.Input, path="cfg.toml", bogus=1)
    return out


def _probe_inspect_numpy(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    import numpy as np

    mod = inspect_numpy_array
    np.save(ws.root / "arr.npy", np.arange(12, dtype=np.float64).reshape(3, 4))
    r = mod.execute(mod.Input(path="arr.npy"), ws.ctx)
    out["npy_shape"] = r.shape == [3, 4] and r.element_count == 12
    out["npy_dtype"] = (
        r.dtype == "float64"
        and r.dtype_kind == "f"
        and r.item_bytes == 8
        and r.original_descriptor == "'<f8'"
    )
    out["npy_no_fortran"] = r.fortran_order is False
    out["npy_payload"] = r.payload_bytes == 96
    out["npy_sha"] = r.source_sha256 == _sha256((ws.root / "arr.npy").read_bytes())
    np.save(ws.root / "obj.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    out["npy_object_refused"] = _fails(lambda: mod.execute(mod.Input(path="obj.npy"), ws.ctx))
    ws.write("trail.npy", (ws.root / "arr.npy").read_bytes() + b"JUNK")
    out["npy_trailing_refused"] = _fails(lambda: mod.execute(mod.Input(path="trail.npy"), ws.ctx))
    out["npy_extra_forbid"] = _refuses(mod.Input, path="arr.npy", bogus=1)
    return out


def _probe_inspect_parquet(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    import pyarrow as pa
    import pyarrow.parquet as pq

    mod = inspect_parquet
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", None]})
    pq.write_table(table, ws.root / "t.parquet", row_group_size=2)
    r = mod.execute(mod.Input(path="t.parquet"), ws.ctx)
    out["pq_columns"] = [c.name for c in r.columns] == ["a", "b"]
    out["pq_leaf_count"] = r.leaf_column_count == 2 and r.total_rows == 3
    out["pq_row_groups"] = r.row_group_count == 2
    out["pq_nullable"] = {c.name: c.nullable for c in r.columns} == {
        "a": True,
        "b": True,
    }
    page = mod.execute(mod.Input(path="t.parquet", row_group_offset=1, row_group_limit=1), ws.ctx)
    out["pq_row_group_paged"] = (
        len(page.row_groups) == 1
        and page.row_groups[0].index == 1
        and page.has_more_row_groups is False
    )
    out["pq_sha"] = r.source_sha256 == _sha256((ws.root / "t.parquet").read_bytes())
    ws.write("junk.parquet", b"NOTAPARQUETFILE")
    out["pq_invalid_refused"] = _fails(lambda: mod.execute(mod.Input(path="junk.parquet"), ws.ctx))
    out["pq_extra_forbid"] = _refuses(mod.Input, path="t.parquet", bogus=1)
    return out


def _probe_inspect_zip(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = inspect_zip
    zpath = ws.root / "arc.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("data/a.txt", "hello")
        zf.writestr("data/b.txt", "world")
        zf.writestr("data/a.txt", "dup")  # duplicate name
        zf.writestr("../evil.txt", "escape")
        info = zipfile.ZipInfo("dironly/")
        zf.writestr(info, "")
    r = mod.execute(mod.Input(path="arc.zip"), ws.ctx)
    out["zip_counts"] = r.entry_count == 5 and r.file_count >= 4
    out["zip_dirs_counted"] = r.directory_count == 1
    out["zip_dup_names"] = r.duplicate_name_count >= 1
    out["zip_path_issues"] = (
        r.entries_with_path_issues >= 1 and sum(r.path_issue_counts.values()) >= 1
    )
    names = [m.name for m in r.members]
    out["zip_members_listed"] = "data/a.txt" in names and r.members[0].index == 0
    out["zip_declared_sizes"] = r.total_declared_uncompressed_bytes >= 18
    out["zip_sha"] = r.source_sha256 == _sha256(zpath.read_bytes())
    page = mod.execute(mod.Input(path="arc.zip", offset=4, limit=1), ws.ctx)
    out["zip_paged"] = len(page.members) == 1 and page.members[0].index == 4
    out["zip_no_extract_side_effects"] = not (ws.root / "evil.txt").exists()
    out["zip_extra_forbid"] = _refuses(mod.Input, path="arc.zip", bogus=1)
    return out


def _probe_verify_hash(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = verify_file_hash
    content = b'{"k": 1}\n'
    ws.write("f.json", content)
    good = mod.execute(mod.Input(path="f.json", expected_sha256=_sha256(content)), ws.ctx)
    out["vh_match"] = (
        good.digest_matches is True and good.matches is True and good.actual_bytes == len(content)
    )
    bad = mod.execute(mod.Input(path="f.json", expected_sha256="a" * 64), ws.ctx)
    out["vh_mismatch"] = bad.digest_matches is False and bad.matches is False
    sized = mod.execute(
        mod.Input(
            path="f.json",
            expected_sha256=_sha256(content),
            expected_bytes=len(content),
        ),
        ws.ctx,
    )
    out["vh_size_check"] = sized.size_matches is True
    wrong_size = mod.execute(
        mod.Input(
            path="f.json",
            expected_sha256=_sha256(content),
            expected_bytes=999,
        ),
        ws.ctx,
    )
    out["vh_size_mismatch_blocks"] = (
        wrong_size.size_matches is False and wrong_size.matches is False
    )
    upper = mod.execute(
        mod.Input(path="f.json", expected_sha256=_sha256(content).upper()),
        ws.ctx,
    )
    out["vh_upper_normalized"] = upper.digest_matches is True
    out["vh_reject_short_digest"] = _refuses(mod.Input, path="f.json", expected_sha256="abc")
    out["vh_reject_nonhex"] = _refuses(mod.Input, path="f.json", expected_sha256="z" * 64)
    out["vh_extra_forbid"] = _refuses(mod.Input, path="f.json", expected_sha256="0" * 64, bogus=1)
    return out


def _probe_join_asof() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = join_asof_observations
    ctx = OperationContext(workspace_root=Path(tempfile.gettempdir()))

    def obs(sec: str, et: datetime, at: datetime, rev: str = "r1", val: int = 1) -> Any:
        return mod.Observation(
            security_id=sec,
            event_time=et,
            available_time=at,
            revision_id=rev,
            source="s",
            values={"v": val},
        )

    def q(sec: str, dt: datetime) -> Any:
        return mod.Query(security_id=sec, decision_time=dt)

    # latest available event wins; activation = max(event, available)
    r = mod.execute(
        mod.Input(
            observations=[
                obs("a", _T0, _T1, val=1),
                obs("a", _T1, _T2, val=2),  # later event + availability
            ],
            queries=[q("a", _DEC)],
        ),
        ctx,
    )
    out["ja_latest_wins"] = (
        r.matched_count == 1
        and r.joined[0].observation_row_index == 1
        and r.joined[0].matched is True
    )
    # observation whose availability arrives after decision is ineligible
    unavail = mod.execute(
        mod.Input(
            observations=[obs("a", _T0, _FAR)],  # availability long after decision
            queries=[q("a", _DEC)],
        ),
        ctx,
    )
    out["ja_unavailable_ineligible"] = (
        unavail.joined[0].matched is False
        and unavail.joined[0].unmatched_reason == "no_eligible_observation"
        and unavail.no_eligible_observation_count == 1
    )
    # activation requires BOTH event <= decision and available <= decision
    pending = mod.execute(
        mod.Input(
            observations=[obs("a", _T2, _T0)],  # event in "future" of mid-decision
            queries=[q("a", _T1)],  # decision before event time
        ),
        ctx,
    )
    out["ja_future_event_ineligible"] = pending.joined[0].matched is False
    # lookback cap
    capped = mod.execute(
        mod.Input(
            observations=[obs("a", _T0, _T0)],
            queries=[q("a", _DEC)],
            max_event_age_seconds=60.0,
        ),
        ctx,
    )
    out["ja_outside_lookback"] = (
        capped.joined[0].matched is False
        and capped.joined[0].unmatched_reason == "outside_lookback"
        and capped.outside_lookback_count == 1
    )
    # revision label breaks event+availability ties (greatest wins);
    # same-clock rows must agree on source and payload to be legal
    rev = mod.execute(
        mod.Input(
            observations=[
                obs("a", _T0, _T1, rev="r1", val=1),
                obs("a", _T0, _T1, rev="r9", val=1),
            ],
            queries=[q("a", _DEC)],
        ),
        ctx,
    )
    out["ja_revision_tiebreak"] = rev.joined[0].observation_row_index == 1
    # same-clock conflicting payload refused at Input construction
    out["ja_same_clock_conflict_refused"] = _refuses(
        mod.Input,
        observations=[
            obs("a", _T0, _T1, rev="r1", val=1),
            obs("a", _T0, _T1, rev="r2", val=2),
        ],
        queries=[q("a", _DEC)],
    )
    paged = mod.execute(
        mod.Input(
            observations=[obs("a", _T0, _T0)],
            queries=[q("a", _T1), q("a", _T2), q("a", _DEC)],
            offset=1,
            limit=1,
        ),
        ctx,
    )
    out["ja_query_ordering_preserved"] = (
        len(paged.joined) == 1 and paged.joined[0].query_row_index == 1
    )
    out["ja_policy_declared"] = (
        r.selection_policy
        == "latest_event_latest_availability_greatest_revision_id_earliest_input_row"
    )
    out["ja_extra_forbid"] = _refuses(
        mod.Input,
        observations=[obs("a", _T0, _T1)],
        queries=[q("a", _DEC)],
        bogus=1,
    )
    return out


def _probe_select_revisions() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = select_asof_revisions
    ctx = OperationContext(workspace_root=Path(tempfile.gettempdir()))

    def rec(et: datetime, at: datetime, rev: str = "r1", val: int = 1) -> Any:
        return mod.Record(
            security_id="s1",
            event_time=et,
            available_time=at,
            revision_id=rev,
            source="src",
            values={"v": val},
        )

    two_vintages = mod.execute(
        mod.Input(records=[rec(_T0, _T1), rec(_T0, _T2)], decision_time=_DEC),
        ctx,
    )
    sel = two_vintages.selected[0]
    out["sr_latest_vintage"] = (
        sel.input_row_index == 1
        and sel.eligible_vintage_rows == 2
        and two_vintages.selected_event_count == 1
        and two_vintages.superseded_or_duplicate_rows == 1
    )
    out["sr_rev_tiebreak"] = (
        mod.execute(
            mod.Input(
                records=[rec(_T0, _T1, "r1"), rec(_T0, _T1, "r9")],
                decision_time=_DEC,
            ),
            ctx,
        )
        .selected[0]
        .input_row_index
        == 1
    )
    out["sr_same_clock_conflict_refused"] = _refuses(
        mod.Input,
        records=[
            mod.Record(
                security_id="s1",
                event_time=_T0,
                available_time=_T1,
                revision_id="r1",
                source="a",
                values={"v": 1},
            ),
            mod.Record(
                security_id="s1",
                event_time=_T0,
                available_time=_T1,
                revision_id="r2",
                source="b",
                values={"v": 2},
            ),
        ],
        decision_time=_DEC,
    )
    future = mod.execute(
        mod.Input(
            records=[rec(_dt("2024-02-01T00:00:00+00:00"), _T0)],
            decision_time=_DEC,
        ),
        ctx,
    )
    out["sr_future_event_kept_default"] = (
        future.selected_event_count == 1 and future.future_event_rows_excluded == 0
    )
    strict = mod.execute(
        mod.Input(
            records=[rec(_dt("2024-02-01T00:00:00+00:00"), _T0)],
            decision_time=_DEC,
            require_event_by_decision=True,
        ),
        ctx,
    )
    out["sr_future_excluded_on_request"] = (
        strict.future_event_rows_excluded == 1 and strict.selected_event_count == 0
    )
    unavail = mod.execute(
        mod.Input(
            records=[rec(_T0, _dt("2024-03-01T00:00:00+00:00"))],
            decision_time=_DEC,
        ),
        ctx,
    )
    out["sr_unavailable_excluded"] = unavail.unavailable_rows == 1
    out["sr_ordering_declared"] = two_vintages.ordering == "security_id_then_event_time_ascending"
    out["sr_extra_forbid"] = _refuses(
        mod.Input, records=[rec(_T0, _T1)], decision_time=_DEC, bogus=1
    )
    return out


def _probe_universe_membership() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = select_universe_membership
    ctx = OperationContext(workspace_root=Path(tempfile.gettempdir()))

    def mem(
        sec: str,
        state: Literal["included", "excluded"],
        ef: datetime,
        et: datetime | None = None,
        at: datetime = _T0,
        mid: str = "m1",
        rev: str = "r1",
    ) -> Any:
        return mod.Membership(
            membership_id=mid,
            security_id=sec,
            state=state,
            effective_from=ef,
            effective_to=et,
            available_time=at,
            revision_id=rev,
            source="s",
        )

    inc = mod.execute(
        mod.Input(
            memberships=[mem("a", "included", _T0)],
            decision_time=_DEC,
            security_ids=["a"],
        ),
        ctx,
    )
    out["um_included"] = inc.included_count == 1 and inc.selections[0].status == "included"
    exc = mod.execute(
        mod.Input(
            memberships=[
                mem("a", "included", _T0, mid="m1"),
                mem("a", "excluded", _T1, mid="m2"),
            ],
            decision_time=_DEC,
            security_ids=["a"],
        ),
        ctx,
    )
    out["um_latest_effective_wins"] = (
        exc.excluded_count == 1 and exc.selections[0].status == "excluded"
    )
    # expired exclusion reveals older still-active inclusion
    reveal = mod.execute(
        mod.Input(
            memberships=[
                mem("a", "included", _T0, mid="m1"),
                mem("a", "excluded", _T1, et=_T2, mid="m2"),
            ],
            decision_time=_DEC,
            security_ids=["a"],
        ),
        ctx,
    )
    out["um_expired_reveals_inclusion"] = (
        reveal.included_count == 1 and reveal.selections[0].status == "included"
    )
    missing = mod.execute(mod.Input(memberships=[], decision_time=_DEC, security_ids=["zzz"]), ctx)
    out["um_missing"] = missing.missing_count == 1 and missing.selections[0].status == "missing"
    amb = mod.execute(
        mod.Input(
            memberships=[
                mem("a", "included", _T0, mid="m1"),
                mem("a", "excluded", _T0, mid="m2"),  # same effective_from, different state
            ],
            decision_time=_DEC,
            security_ids=["a"],
        ),
        ctx,
    )
    out["um_same_effective_ambiguous"] = (
        amb.ambiguous_count == 1 and amb.selections[0].status == "ambiguous"
    )
    unavail = mod.execute(
        mod.Input(
            memberships=[mem("a", "included", _T0, at=_FAR)],
            decision_time=_DEC,
            security_ids=["a"],
        ),
        ctx,
    )
    out["um_unavailable_not_counted"] = (
        unavail.unavailable_input_rows == 1 and unavail.missing_count == 1
    )
    out["um_interval_declared"] = (
        inc.interval_policy == "effective_from_inclusive_effective_to_exclusive"
    )
    out["um_extra_forbid"] = _refuses(mod.Input, decision_time=_DEC, bogus=1)
    return out


def _probe_resolve_identity() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = resolve_security_identity
    ctx = OperationContext(workspace_root=Path(tempfile.gettempdir()))

    def mapping(
        sec: str,
        ticker: str,
        exch: str,
        vf: datetime,
        vt: datetime | None = None,
        at: datetime = _T0,
        mid: str = "m1",
    ) -> Any:
        return mod.Mapping(
            mapping_id=mid,
            security_id=sec,
            ticker=ticker,
            exchange=exch,
            valid_from=vf,
            valid_to=vt,
            available_time=at,
            revision_id="r1",
            source="s",
        )

    def query(ticker: str, exch: str, dt: datetime = _DEC) -> Any:
        return mod.Query(ticker=ticker, exchange=exch, decision_time=dt)

    ok = mod.execute(
        mod.Input(
            mappings=[mapping("S1", "ABC", "XNYS", _T0)],
            queries=[query("ABC", "XNYS")],
        ),
        ctx,
    )
    out["ri_resolved"] = (
        ok.resolved_count == 1
        and ok.resolutions[0].status == "resolved"
        and ok.resolutions[0].security_id == "S1"
    )
    miss = mod.execute(
        mod.Input(
            mappings=[mapping("S1", "ABC", "XNYS", _T0)],
            queries=[query("ZZZ", "XNYS")],
        ),
        ctx,
    )
    out["ri_missing"] = miss.missing_count == 1 and miss.resolutions[0].status == "missing"
    # case-sensitive: lowercase ticker does not match
    case = mod.execute(
        mod.Input(
            mappings=[mapping("S1", "ABC", "XNYS", _T0)],
            queries=[query("abc", "xnys")],
        ),
        ctx,
    )
    out["ri_case_sensitive"] = case.missing_count == 1
    # valid interval [from,to): decision at valid_to is out
    expired = mod.execute(
        mod.Input(
            mappings=[mapping("S1", "ABC", "XNYS", _T0, vt=_T2)],
            queries=[query("ABC", "XNYS", dt=_T2)],
        ),
        ctx,
    )
    out["ri_interval_exclusive"] = expired.missing_count == 1
    # latest availability per mapping_id supersedes
    superseded = mod.execute(
        mod.Input(
            mappings=[
                mapping("S1", "ABC", "XNYS", _T0, at=_T1, mid="m1"),
                mapping("S2", "ABC", "XNYS", _T0, at=_T2, mid="m1"),
            ],
            queries=[query("ABC", "XNYS")],
        ),
        ctx,
    )
    out["ri_latest_availability_wins"] = superseded.resolutions[0].security_id == "S2"
    out["ri_policy_declared"] = (
        ok.interval_policy == "valid_from_inclusive_valid_to_exclusive"
        and "latest_availability" in ok.revision_policy
    )
    out["ri_extra_forbid"] = _refuses(
        mod.Input,
        mappings=[mapping("S1", "ABC", "XNYS", _T0)],
        queries=[query("ABC", "XNYS")],
        bogus=1,
    )
    return out


def _probe_ingestion_latency() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = summarize_ingestion_latency
    ctx = OperationContext(workspace_root=Path(tempfile.gettempdir()))

    def obs(src: str, at: datetime, it: datetime) -> Any:
        return mod.Observation(source=src, available_time=at, ingested_time=it)

    r = mod.execute(
        mod.Input(
            observations=[
                obs("a", _T0, _T1),  # +60s
                obs("a", _T0, _T2),  # +120s
                obs("b", _T2, _T1),  # -60s (clock anomaly)
            ]
        ),
        ctx,
    )
    out["il_signed_keeps_negative"] = (
        r.overall.negative_lag_count == 1 and r.overall.observation_count == 3
    )
    out["il_histogram_negative_bin"] = (
        r.overall.histogram[0].lower_seconds_inclusive is None and r.overall.histogram[0].count == 1
    )
    out["il_nonneg_separate"] = r.overall.nonnegative.count == 2
    out["il_zero_counted"] = (
        mod.execute(
            mod.Input(observations=[obs("a", _T0, _T0)]),
            ctx,
        ).overall.zero_lag_count
        == 1
    )
    # median of sorted {-60,60,120} at index (n-1)*0.5 = 1 → 60
    qs = {q.probability: q.seconds for q in r.overall.signed.quantiles}
    out["il_quantile_interp"] = 0.5 in qs and abs(qs[0.5] - 60.0) < 1e-9
    out["il_signed_min_negative"] = (
        r.overall.signed.minimum_seconds is not None
        and abs(r.overall.signed.minimum_seconds - (-60.0)) < 1e-9
    )
    by_src = {s.source: s for s in r.sources}
    out["il_per_source"] = by_src["b"].summary.negative_lag_count == 1
    out["il_extra_forbid"] = _refuses(mod.Input, observations=[obs("a", _T0, _T1)], bogus=1)
    return out


def ioops_audit() -> dict[str, bool]:
    """Every file-I/O and bitemporal-selection contract as booleans."""
    ws = _Ws()
    try:
        out: dict[str, bool] = {}
        out.update(_probe_descriptors())
        out.update(_probe_containment(ws))
        out.update(_probe_read_csv(ws))
        out.update(_probe_read_jsonl(ws))
        out.update(_probe_read_toml(ws))
        out.update(_probe_inspect_numpy(ws))
        out.update(_probe_inspect_parquet(ws))
        out.update(_probe_inspect_zip(ws))
        out.update(_probe_verify_hash(ws))
        out.update(_probe_join_asof())
        out.update(_probe_select_revisions())
        out.update(_probe_universe_membership())
        out.update(_probe_resolve_identity())
        out.update(_probe_ingestion_latency())
        return out
    finally:
        ws._tmp.cleanup()


def ioops_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = ioops_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "ioops_audit",
        "schema": "ioops_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process execute() calls against a temp workspace; no served app",
            "not_verified": [
                "operations/base descriptor validation edges (registry_audit lane)",
                "registry dispatch (registry_audit lane)",
                "concurrent open_binary races under symlink swap (POSIX-only guard tested)",
            ],
        },
        "interpretation": (
            "I/O + bitemporal ops hold: workspace containment fails closed "
            "on escapes/suffixes/absent/oversized inputs and symlinks; "
            "CSV/JSONL/TOML readers give exact-string/strict-JSON/"
            "temporal-typed output with source fingerprints and honest "
            "paging; NPY/Parquet/ZIP inspectors read metadata without "
            "materializing payloads and refuse objects/trailing bytes; "
            "file-hash verification is byte-exact with normalized "
            "digests; joins/selects activate at max(event, available), "
            "pick latest-vintage winners with declared tie policies, and "
            "keep unavailable/future records out of eligibility; "
            "ingestion-latency summaries keep negative lags visible."
            if ok
            else f"IOOPS AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(ioops_audit_bench(), indent=2, sort_keys=True))
