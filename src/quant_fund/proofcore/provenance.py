"""PROOFCORE provenance DB (duckdb) — W5.

Append-oriented provenance ledger for proof bundles (W2) and reality-filter
trial rows (W4). Legitimizes the existing duckdb hard dependency (A2 F14).

Layering (DESIGN.md §1.3, layer 3): imports contracts + duckdb + stdlib ONLY.
The standalone W2 verifier exists, but no bound result schema or validation
path connects its verdict to this database. Verification ingestion remains
fail-closed; bundles may be logged as unverified evidence.
"""

from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path
from typing import Any

import duckdb

from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    ProofBundleV1,
    ProvenanceError,
    TrialLedgerRow,
    canonical_json_bytes,
)

DEFAULT_DB_PATH: Path = Path("data/metadata/proofcore.duckdb")
PROOFCORE_SCHEMA_VERSION = 1

_BUNDLE_COLUMNS: tuple[str, ...] = (
    "bundle_id",
    "created_utc",
    "run_kind",
    "git_revision",
    "config_sha256",
    "seed",
    "merkle_root",
    "prev_bundle_hash",
    "signature_scheme",
    "bundle_json",
    "verified_ok",
    "verification_json",
)

_TRIAL_COLUMNS: tuple[str, ...] = (
    "trial_id",
    "bundle_hash",
    "family",
    "strategy",
    "cluster_id",
    "n_obs",
    "periods_per_year",
    "sharpe_periodic",
    "skew",
    "kurtosis_raw",
    "returns_sha256",
    "created_utc",
)

# Storage column set for trial_ledger: the caller-facing row plus the
# storage-bound chain link. ``insert_trial`` binds ``prev_trial_hash`` to the
# current chain head — callers never set it, so the chain cannot be
# accidentally or deliberately seeded mid-history.
_TRIAL_STORAGE_COLUMNS: tuple[str, ...] = _TRIAL_COLUMNS + ("prev_trial_hash",)

_DDL = """
CREATE TABLE IF NOT EXISTS proof_bundles (
    bundle_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    run_kind TEXT NOT NULL,
    git_revision TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    seed BIGINT NOT NULL,
    merkle_root TEXT NOT NULL,
    prev_bundle_hash TEXT NOT NULL UNIQUE,
    signature_scheme TEXT NOT NULL,
    bundle_json TEXT NOT NULL,
    verified_ok BOOLEAN,
    verification_json TEXT
);
CREATE TABLE IF NOT EXISTS trial_ledger (
    trial_id TEXT PRIMARY KEY,
    bundle_hash TEXT NOT NULL REFERENCES proof_bundles (bundle_id),
    family TEXT NOT NULL,
    strategy TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    n_obs BIGINT NOT NULL,
    periods_per_year DOUBLE NOT NULL,
    sharpe_periodic DOUBLE NOT NULL,
    skew DOUBLE NOT NULL,
    kurtosis_raw DOUBLE NOT NULL,
    returns_sha256 TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    prev_trial_hash TEXT NOT NULL UNIQUE
);
"""

_SCHEMA_VERSION_DDL = """
CREATE TABLE proofcore_schema_version (
    schema_version BIGINT PRIMARY KEY
);
"""

# Bare names only. Quotes, comments, dots, and statement breaks never interpolate.
_SQL_IDENTIFIER = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")


def _sql_identifier(name: str) -> str:
    """Return one bare SQL name, or refuse anything that can change the statement.

    Values are never passed here. Callers bind those with ``?`` placeholders.
    """
    if _SQL_IDENTIFIER.fullmatch(name) is None:
        raise ProvenanceError(
            f"refusing SQL identifier {name!r}: only a single unquoted name may be interpolated"
        )
    return name


def _sql_identifier_list(names: tuple[str, ...]) -> str:
    if not names:
        raise ProvenanceError("refusing an empty SQL identifier list")
    return ", ".join(_sql_identifier(name) for name in names)


def _bound_placeholders(count: int) -> str:
    if count < 1:
        raise ProvenanceError("refusing SQL with no bound parameters")
    return ", ".join("?" for _ in range(count))


def _insert_statement(table: str, columns: tuple[str, ...]) -> str:
    return " ".join(
        (
            "INSERT INTO",
            _sql_identifier(table),
            "(" + _sql_identifier_list(columns) + ")",
            "VALUES",
            "(" + _bound_placeholders(len(columns)) + ")",
        )
    )


def _select_statement(table: str, columns: tuple[str, ...]) -> str:
    return " ".join(
        (
            "SELECT",
            _sql_identifier_list(columns),
            "FROM",
            _sql_identifier(table),
        )
    )


def _where_bound(column: str) -> str:
    return " WHERE " + _sql_identifier(column) + " = ?"


def _order_by(columns: tuple[str, ...]) -> str:
    return " ORDER BY " + _sql_identifier_list(columns)


_BUNDLE_SQL_TYPES = (
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "BIGINT",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "BOOLEAN",
    "VARCHAR",
)
_TRIAL_SQL_TYPES = (
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "BIGINT",
    "DOUBLE",
    "DOUBLE",
    "DOUBLE",
    "DOUBLE",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
)


def _table_info(connection: duckdb.DuckDBPyConnection, table: str) -> list[Any]:
    present = connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    if present is None or int(present[0]) == 0:
        return []
    return connection.execute(f"PRAGMA table_info({_sql_identifier(table)})").fetchall()


def _validate_table(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: tuple[str, ...],
    sql_types: tuple[str, ...],
    nullable: frozenset[str],
    primary_key: str,
) -> None:
    rows = _table_info(connection, table)
    actual = {str(row[1]): row for row in rows}
    if len(rows) != len(columns) or set(actual) != set(columns):
        raise ProvenanceError(f"incompatible provenance schema: {table} columns")
    for name, sql_type in zip(columns, sql_types, strict=True):
        row = actual[name]
        if (
            str(row[2]).upper() != sql_type
            or bool(row[3]) != (name not in nullable)
            or bool(row[5]) != (name == primary_key)
        ):
            raise ProvenanceError(f"incompatible provenance schema: {table}.{name}")


def _validate_constraints(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    required: set[tuple[str, tuple[str, ...], str | None, tuple[str, ...]]],
) -> None:
    rows = connection.execute(
        "SELECT constraint_type, constraint_column_names, referenced_table, "
        "referenced_column_names FROM duckdb_constraints() WHERE table_name = ?",
        [table],
    ).fetchall()
    actual = {
        (str(kind), tuple(columns or []), referenced, tuple(referenced_columns or []))
        for kind, columns, referenced, referenced_columns in rows
    }
    if not required <= actual:
        raise ProvenanceError(f"incompatible provenance schema: {table} constraints")


def _schema_version_status(connection: duckdb.DuckDBPyConnection) -> bool:
    if not _table_info(connection, "proofcore_schema_version"):
        return False
    _validate_table(
        connection,
        "proofcore_schema_version",
        ("schema_version",),
        ("BIGINT",),
        frozenset(),
        "schema_version",
    )
    _validate_constraints(
        connection,
        "proofcore_schema_version",
        {("PRIMARY KEY", ("schema_version",), None, ())},
    )
    rows = connection.execute("SELECT schema_version FROM proofcore_schema_version").fetchall()
    if len(rows) != 1 or type(rows[0][0]) is not int:
        raise ProvenanceError("incompatible provenance schema: version metadata")
    if rows[0][0] != PROOFCORE_SCHEMA_VERSION:
        raise ProvenanceError(f"unsupported provenance schema version: {rows[0][0]}")
    return True


_SCHEMA_BUNDLE_COLUMNS = _BUNDLE_COLUMNS
_SCHEMA_TRIAL_COLUMNS = _TRIAL_STORAGE_COLUMNS


def _validate_existing_schema(connection: duckdb.DuckDBPyConnection) -> bool:
    bundle_info = _table_info(connection, "proof_bundles")
    trial_info = _table_info(connection, "trial_ledger")
    if not bundle_info or not trial_info:
        raise ProvenanceError("incompatible provenance schema: ledger tables are missing")
    if "prev_trial_hash" not in {str(row[1]) for row in trial_info}:
        raise ProvenanceError(
            "trial_ledger predates the prev_trial_hash chain schema — "
            "regenerate the provenance DB rather than migrating a "
            "ledger that claims append-only evidence"
        )
    _validate_table(
        connection,
        "proof_bundles",
        _SCHEMA_BUNDLE_COLUMNS,
        _BUNDLE_SQL_TYPES,
        frozenset({"verified_ok", "verification_json"}),
        "bundle_id",
    )
    _validate_table(
        connection,
        "trial_ledger",
        _SCHEMA_TRIAL_COLUMNS,
        _TRIAL_SQL_TYPES,
        frozenset(),
        "trial_id",
    )
    _validate_constraints(
        connection,
        "proof_bundles",
        {
            ("PRIMARY KEY", ("bundle_id",), None, ()),
            ("UNIQUE", ("prev_bundle_hash",), None, ()),
        },
    )
    _validate_constraints(
        connection,
        "trial_ledger",
        {
            ("PRIMARY KEY", ("trial_id",), None, ()),
            ("UNIQUE", ("prev_trial_hash",), None, ()),
            ("FOREIGN KEY", ("bundle_hash",), "proof_bundles", ("bundle_id",)),
        },
    )
    return _schema_version_status(connection)


def _write_schema(connection: duckdb.DuckDBPyConnection, *, initialize: bool) -> None:
    connection.execute("BEGIN TRANSACTION")
    try:
        if initialize:
            connection.execute(_DDL)
        connection.execute(_SCHEMA_VERSION_DDL)
        connection.execute(
            "INSERT INTO proofcore_schema_version VALUES (?)", [PROOFCORE_SCHEMA_VERSION]
        )
        connection.execute("COMMIT")
    except Exception:
        with suppress(Exception):
            connection.execute("ROLLBACK")
        raise


class ProvenanceDB:
    """Versioned DuckDB provenance ledger with explicit read-only opens.

    Exact current-layout databases without version metadata are legacy schema
    v1. Read-only opens preserve them; writable opens validate then stamp v1.
    Unknown versions and narrower historical layouts are never auto-migrated.

    Trials chain like bundles: each insert binds ``prev_trial_hash`` to the
    current ``_trial_head()``, so a DELETE of a losing trial — the deflation
    attack this ledger exists to defeat — leaves a dangling link that
    ``verify_trial_chain`` and ``_trial_head`` both surface.
    """

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        existed = self.path.exists()
        if read_only and not self.path.is_file():
            raise ProvenanceError(f"cannot open provenance DB at {self.path}: file does not exist")
        if not read_only and self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection: duckdb.DuckDBPyConnection | None = None
        try:
            connection = duckdb.connect(str(self.path), read_only=read_only)
            if not existed:
                _write_schema(connection, initialize=True)
                _validate_existing_schema(connection)
            else:
                versioned = _validate_existing_schema(connection)
                if not read_only and not versioned:
                    _write_schema(connection, initialize=False)
            self._con = connection
        except ProvenanceError:
            if connection is not None:
                with suppress(Exception):
                    connection.close()
            raise
        except Exception as exc:
            if connection is not None:
                with suppress(Exception):
                    connection.close()
            raise ProvenanceError(f"cannot open provenance DB at {self.path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_bundle(self, bundle: ProofBundleV1, verification: object | None) -> None:
        """Append one bundle, preserving exact content and chain order.

        An identical reinsert is a no-op. Verification results cannot be
        accepted until a bound verifier result schema and ingestion path exist.
        """
        if verification is not None:
            raise ProvenanceError(
                "verification ingestion unavailable until a bound result path exists"
            )
        bundle_json = canonical_json_bytes(bundle.model_dump(mode="json")).decode("utf-8")
        try:
            self._con.execute("BEGIN TRANSACTION")
            existing = self._con.execute(
                "SELECT bundle_json FROM proof_bundles WHERE bundle_id = ?",
                [bundle.bundle_id],
            ).fetchone()
            if existing is not None:
                if existing[0] != bundle_json:
                    raise ProvenanceError(
                        f"bundle {bundle.bundle_id} already stored with different contents; "
                        "the bundle ledger is append-only"
                    )
                self._con.execute("COMMIT")
                return
            expected_prev = self.chain_head()
            if bundle.prev_bundle_hash != expected_prev:
                raise ProvenanceError(
                    f"bundle {bundle.bundle_id} predecessor {bundle.prev_bundle_hash} "
                    f"does not match chain head {expected_prev}"
                )
            self._con.execute(
                _insert_statement("proof_bundles", _BUNDLE_COLUMNS),
                [
                    bundle.bundle_id,
                    bundle.created_utc,
                    bundle.run_kind,
                    bundle.code.git_revision,
                    bundle.config_sha256,
                    bundle.seed,
                    bundle.data_manifest.merkle_root,
                    bundle.prev_bundle_hash,
                    bundle.signature.scheme,
                    bundle_json,
                    None,
                    None,
                ],
            )
            self._con.execute("COMMIT")
        except ProvenanceError:
            self._con.execute("ROLLBACK")
            raise
        except Exception as exc:
            self._con.execute("ROLLBACK")
            raise ProvenanceError(f"insert_bundle({bundle.bundle_id}) failed: {exc}") from exc

    def insert_trial(self, row: TrialLedgerRow) -> None:
        """Append one trial-ledger row.

        Idempotent for byte-identical re-inserts; refuses a ``trial_id`` whose
        stored row differs (raise ``ProvenanceError``) — tamper-evidence at
        the DB layer (DESIGN.md §9.1).
        """
        existing = self._con.execute(
            _select_statement("trial_ledger", _TRIAL_COLUMNS) + _where_bound("trial_id"),
            [row.trial_id],
        ).fetchone()
        values = self._trial_values(row)
        if existing is not None:
            stored = [self._normalize_cell(v) for v in existing]
            wanted = [self._normalize_cell(v) for v in values]
            if stored != wanted:
                raise ProvenanceError(
                    f"trial {row.trial_id} already stored with different contents; "
                    "the trial ledger is append-only"
                )
            return
        try:
            self._con.execute(
                _insert_statement("trial_ledger", _TRIAL_STORAGE_COLUMNS),
                [*values, self._trial_head()],
            )
        except ProvenanceError:
            raise
        except Exception as exc:
            raise ProvenanceError(f"insert_trial({row.trial_id}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def trials(self, *, family: str | None = None) -> list[TrialLedgerRow]:
        """All trial rows (optionally one family), ordered by (created_utc, trial_id)."""
        sql = _select_statement("trial_ledger", _TRIAL_COLUMNS)
        params: list[Any] = []
        if family is not None:
            sql += _where_bound("family")
            params.append(family)
        sql += _order_by(("created_utc", "trial_id"))
        rows = self._con.execute(sql, params).fetchall()
        return [
            TrialLedgerRow(
                trial_id=r[0],
                bundle_hash=r[1],
                family=r[2],
                strategy=r[3],
                cluster_id=r[4],
                n_obs=r[5],
                periods_per_year=r[6],
                sharpe_periodic=r[7],
                skew=r[8],
                kurtosis_raw=r[9],
                returns_sha256=r[10],
                created_utc=r[11],
            )
            for r in rows
        ]

    def bundles(self) -> list[dict[str, Any]]:
        """All proof-bundle rows as plain dicts (audit/export path), in hash-chain
        order — the genesis-linked bundle first, each row followed by the bundle
        that points to it. Wall-clock order is not chain order: skewed
        ``created_utc`` values must not reorder an audit trail.

        Any row the genesis walk cannot reach (dangling prev pointer, hand-edited
        table) raises ``ProvenanceError`` — a broken chain must be loud, not
        silently reordered."""
        sql = _select_statement("proof_bundles", _BUNDLE_COLUMNS)
        rows = [
            dict(zip(_BUNDLE_COLUMNS, r, strict=True)) for r in self._con.execute(sql).fetchall()
        ]
        # prev_bundle_hash is UNIQUE and bundle_id is the PK, so the table is a
        # linked list: at most one row claims each predecessor.
        by_prev = {row["prev_bundle_hash"]: row for row in rows}
        ordered: list[dict[str, Any]] = []
        cursor = by_prev.pop(GENESIS_HASH, None)
        while cursor is not None:
            ordered.append(cursor)
            cursor = by_prev.pop(cursor["bundle_id"], None)
        if by_prev:
            raise ProvenanceError(
                "proof_bundles does not form a single chain from genesis: "
                f"{len(by_prev)} row(s) unreachable — possible tamper"
            )
        return ordered

    def chain_head(self) -> str:
        """Current head of the bundle chain: the stored bundle no other bundle
        points to via ``prev_bundle_hash``. ``GENESIS_HASH`` on an empty DB.

        Inserts and the UNIQUE(prev_bundle_hash) constraint keep a
        well-formed ledger a single chain with exactly one unreferenced
        head. A different head count is only possible by direct DB
        manipulation — a fork (extra heads) or a cycle/dangling link (no
        head) — and the ledger must fail loudly rather than arbitrate.
        """
        heads: list[tuple[str, ...]] = self._con.execute(
            "SELECT bundle_id FROM proof_bundles "
            "WHERE bundle_id NOT IN (SELECT prev_bundle_hash FROM proof_bundles) "
            "ORDER BY created_utc DESC, bundle_id DESC LIMIT 2"
        ).fetchall()
        if len(heads) == 1:
            return heads[0][0]
        n_bundles_row = self._con.execute("SELECT count(*) FROM proof_bundles").fetchone()
        n_bundles = 0 if n_bundles_row is None else int(n_bundles_row[0])
        if n_bundles == 0:
            return GENESIS_HASH
        shape = "forked" if heads else "has no head (cycle or dangling link)"
        raise ProvenanceError(
            f"bundle chain {shape} — {n_bundles} stored bundles, "
            f"{len(heads)} unreferenced heads; the provenance ledger "
            "has been tampered with"
        )

    def _trial_head(self) -> str:
        """Head of the trial chain, mirroring ``chain_head`` semantics.

        Exactly one unreferenced trial means a well-formed ledger; zero rows
        means ``GENESIS_HASH``. Any other state is only reachable through
        direct DB manipulation and must fail loudly.
        """
        heads = self._con.execute(
            "SELECT trial_id FROM trial_ledger "
            "WHERE trial_id NOT IN (SELECT prev_trial_hash FROM trial_ledger) "
            "ORDER BY created_utc DESC, trial_id DESC LIMIT 2"
        ).fetchall()
        if len(heads) == 1:
            return str(heads[0][0])
        n_rows = self._con.execute("SELECT count(*) FROM trial_ledger").fetchone()
        if not heads and n_rows is not None and int(n_rows[0]) == 0:
            return GENESIS_HASH
        shape = "forked" if heads else "has no head (cycle or dangling link)"
        raise ProvenanceError(f"trial chain {shape} — the provenance ledger has been tampered with")

    def verify_trial_chain(self) -> list[str]:
        """Walk the trial chain from genesis to head; return the ordered ids.

        Raises ``ProvenanceError`` on any malformed link: a row whose
        ``prev_trial_hash`` dangles, extra rows unreachable from genesis, a
        fork, or a cycle. Complements the per-row idempotence check — this is
        the audit that proves the log is complete, not just consistent.
        """
        rows = self._con.execute("SELECT trial_id, prev_trial_hash FROM trial_ledger").fetchall()
        if not rows:
            return []
        by_prev: dict[str, str] = {}
        for trial_id, prev in rows:
            if prev in by_prev:
                raise ProvenanceError(
                    f"trial chain forked at {prev}: both {by_prev[prev]} and {trial_id} "
                    "claim it as predecessor"
                )
            by_prev[prev] = trial_id
        ordered: list[str] = []
        cursor = by_prev.pop(GENESIS_HASH, None)
        if cursor is None:
            raise ProvenanceError(
                "trial chain has no genesis link — the first row was rewritten or deleted"
            )
        while cursor is not None:
            ordered.append(cursor)
            cursor = by_prev.pop(cursor, None)
        if by_prev:
            raise ProvenanceError(
                f"trial chain has {len(by_prev)} row(s) unreachable from genesis — "
                "the ledger has been tampered with"
            )
        return ordered

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> ProvenanceDB:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _trial_values(row: TrialLedgerRow) -> list[Any]:
        return [
            row.trial_id,
            row.bundle_hash,
            row.family,
            row.strategy,
            row.cluster_id,
            row.n_obs,
            row.periods_per_year,
            row.sharpe_periodic,
            row.skew,
            row.kurtosis_raw,
            row.returns_sha256,
            row.created_utc,
        ]

    @staticmethod
    def _normalize_cell(value: Any) -> Any:
        # DuckDB returns Python ints for BIGINT and floats for DOUBLE; do not
        # coerce ints to float, which loses precision above 2**53.
        return value
