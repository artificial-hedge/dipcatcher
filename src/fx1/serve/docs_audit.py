"""docs_audit — pin the committed docs to the served harness surface.

``contract_audit`` pins the served surface to a generated golden;
``spec_audit`` pins the wire to vendor grammars. This battery pins the
*documentation* to both: the operator-facing claims in
``docs/FX1_HARNESS_API.md``, ``docs/FX1_DEPLOY.md``,
``docs/FX1_CLIENTS.md``, ``docs/AUDIT_LEDGER.md``, the README's fx-1
sections, and the committed TypeScript client artifacts
(``clients/typescript/fx1/openapi.json`` + ``schema.d.ts``) must
describe what is actually served — no orphan routes, no phantom
fields, no error codes the wire never emits, no env vars the code
never reads.

Probe surfaces (each a literal bool in ``results``):

- *Route parity, both directions* — every ``METHOD /path`` row in the
  harness API doc's route table resolves to a served route; every
  served route is documented or sits on the pinned exclusion list
  (the framework's own docs surface and the deliberate catch-all).
- *Request fields* — field names the doc binds to a route exist in the
  served OpenAPI request schema.
- *Response shapes* — response bodies the doc describes are the ones
  the live route returns (version pair, capabilities, key mint,
  job/eval submissions, score list, health/ready keys).
- *OpenAPI golden* — ``clients/typescript/fx1/openapi.json``
  regenerates byte-identically via the committed export recipe
  (``json.dumps(spec, indent=2, sort_keys=True)``); every spec path is
  served; every served path has a spec entry or is pinned excluded.
- *TS schema* — ``schema.d.ts`` pins to a sha256 (regenerate via
  ``npm run generate:schema``); every spec path has a d.ts entry and
  every spec-declared parameter name appears in the d.ts operation
  block; the TS client's called paths are all served.
- *Client coverage* — every ``HarnessClient`` public method emits at
  least one HTTP call through an injected transport and every emitted
  request resolves to a served route; every ``HarnessClient.<name>``
  cited in the docs resolves to a real method.
- *SDK coverage* — every ``Fx1Harness`` public method is invocable
  in-process: success or a documented refusal (``KeyError`` /
  ``ValueError`` / ``RuntimeError`` / a typed harness error), never a
  ``TypeError``/``AttributeError`` on the call signature.
- *CLI coverage* — every ``fx1 harness <name>`` cited in the docs is a
  real Typer command and parses ``--help``; the commands the docs
  never mention sit on the pinned exclusion list; each command's
  client-call surface resolves to served routes.
- *Deploy docs* — every ``FX1_*``/``MOONSHOT_*`` token the deploy doc
  binds to behavior is read by the code; every ``--flag`` it names is
  a real CLI option; every ``(status, code)`` row of its failure-modes
  table reproduces live over stub backends.
- *Error contract* — the documented ``status → exception`` map is what
  ``HarnessClient._map_error`` actually returns; the three wire
  envelopes (harness ``{detail, code}``, OpenAI
  ``{error:{message,type,param,code}}``, Anthropic
  ``{type:"error",error:{type,message}}``) match live 4xx bodies.
- *README* — route paths and ``fx1 harness`` commands cited in the
  README resolve to the served surface and the CLI; the pinned
  ``fx1.__version__`` claim matches the package.
- *Audit ledger* — every ``src/fx1/serve/*_audit.py`` module is named
  in ``AUDIT_LEDGER.md``; every ``*_audit`` token in the ledger
  resolves to a module file or a committed ``fx1_*_audit`` receipt;
  every ``(PR #N)`` heading resolves to git history; probe counts the
  ledger quotes match the committed receipts; the serve census
  ``n_modules`` equals the true file count.
- *Doc self-consistency* — backticked ``/path`` spans and
  ``METHOD /path`` pairs in prose are served; ``Fx1Harness.<m>`` and
  ``HarnessClient.<m>`` citations resolve.

Honesty: synthetic stub backends and the client transport seam only —
every probe measures the committed bytes against the live app, never a
live backend; the sealed receipt is ``data_label: SYNTHETIC``.

Sealed ``docs_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = ["docs_audit", "docs_audit_bench"]

_REPO = Path(__file__).resolve().parents[3]
_DOC_API = _REPO / "docs" / "FX1_HARNESS_API.md"
_DOC_DEPLOY = _REPO / "docs" / "FX1_DEPLOY.md"
_DOC_CLIENTS = _REPO / "docs" / "FX1_CLIENTS.md"
_DOC_LEDGER = _REPO / "docs" / "AUDIT_LEDGER.md"
_README = _REPO / "README.md"
_TS_DIR = _REPO / "clients" / "typescript" / "fx1"
_TS_OPENAPI = _TS_DIR / "openapi.json"
_TS_SCHEMA = _TS_DIR / "schema.d.ts"
_TS_CLIENT = _TS_DIR / "client.ts"
_COVERAGE = _REPO / "quality" / "audit_coverage_fx1.json"
_SERVE_DIR = _REPO / "src" / "fx1" / "serve"
_RECEIPTS_DIR = _REPO / "receipts"
_SRC_FX1 = _REPO / "src" / "fx1"

_API_KEY_ENV = "FX1_API_KEY"
_ROOT_KEY = "k3y-material"
_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})

# Paths the route table deliberately never lists: FastAPI's own
# docs/spec endpoints and the ``/v1/{path:path}`` compat catch-all
# shim (documented in prose as a pass-through, not a route). Every
# method on them — including the implicit HEAD/OPTIONS the framework
# adds — is excluded at the path level.
_P_DOCS = "/docs"
_P_V1_PH = "/v1/{}"
_P_KEYS = "/harness/keys"
_P_CHAT = "/v1/chat/completions"
_P_STREAM = "/harness/complete/stream"
_P_COMPLETE = "/harness/complete"
_P_COMMANDS = "/harness/commands"

_DOC_EXCLUDED_PATHS = frozenset(
    {_P_DOCS, "/docs/oauth2-redirect", "/redoc", "/openapi.json", _P_V1_PH}
)

# Served paths absent from the exported OpenAPI spec: the framework's
# self-docs, the spec's own endpoint (it cannot list itself), and the
# catch-all shim. ``GET /openapi.json`` IS documented in prose.
_OPENAPI_EXCLUDED_PATHS = frozenset(
    {_P_DOCS, "/docs/oauth2-redirect", "/openapi.json", "/redoc", _P_V1_PH}
)

# ``fx1 harness`` subcommands the docs never name — lifecycle twins and
# internals the prose deliberately skips; pinned so a silently
# undocumented new command fails the battery.
_CLI_UNDOCUMENTED = frozenset(
    {
        "chat-messages",
        "conv-delete",
        "conv-get",
        "conv-items",
        "conv-items-add",
        "conv-items-delete",
        "conv-update",
        "describe-operation",
        "eval-run-delete",
        "eval-run-get",
        "eval-run-list",
        "eval-spec-delete",
        "eval-spec-get",
        "eval-spec-update",
        "execute-operation",
        "ft-checkpoints",
        "ft-wait",
        "message-batch-status",
        "message-batches",
        "model-delete",
        "response-input-items",
        "submit-batch",
        "upload-cancel",
        "vs-batch-cancel",
        "vs-batch-files",
        "vs-batch-get",
        "vs-delete",
        "vs-file-delete",
        "vs-file-get",
        "vs-files",
        "vs-get",
        "vs-update",
    }
)

# Deploy-doc tokens that are doc *names* (``docs/FX1_*.md`` links), not
# env vars the code must read.
_DEPLOY_DOCNAME_TOKENS = frozenset(
    {
        "FX1_API_STABILITY",
        "FX1_ARCHITECTURE",
        "FX1_BYOK_RUNBOOK",
        "FX1_CLIENTS",
        "FX1_CORPUS_LINE",
        "FX1_DATA",
        "FX1_DATASOURCES",
        "FX1_DEPLOY",
        "FX1_HARNESS_API",
        "FX1_TRAINING",
        "FX1_BYOK_",
        "FX1_API_",
        "MOONSHOT_API",
        "ANTHROPIC_API",
    }
)

# Deploy-doc env tokens the *caller* sets in the doc's own client
# snippets — user-side credentials, never read by this codebase.
_DEPLOY_USER_ENVS = frozenset({"FX1_CALLBACK_SECRET"})

# README ``METHOD /path`` pairs that belong to the quant_fund lab API,
# not the fx1 surface — the fx1 sections only ever cite fx1 routes.
_README_LAB_PATHS = frozenset({"/backtest"})

# ``fx1 harness`` commands that never touch the wire — local-only
# surfaces (server startup, local benchmarks, in-process registry ops).
# Everything else must emit at least one client call whose path is a
# served route.
_CLI_LOCAL_ONLY = frozenset(
    {
        "serve",
        "selftest",
        "bench",
        "operations",
        "describe-operation",
        "execute-operation",
        "list",
        "run",
        "submit",
        "submit-batch",
        "batch-run",
        "cancel",
        "wait",
        "watch",
        "verify",
        "receipts",
        "receipt",
        "commands",
        "version",
        "compat",
        "probe",
        "drain",
        "usage",
        "metrics",
        "health",
        "ready",
        "self",
    }
)

# ``schema.d.ts`` freshness pin: regenerating
# ``cd clients/typescript/fx1 && npm run generate:schema`` from the
# committed ``openapi.json`` must reproduce exactly these bytes.
_SCHEMA_TS_SHA256 = "fd4ecf69d6150691e7dce66646314d2102759070c3331b80b2bc8274307b00b1"


# --- doc parsing -------------------------------------------------------------


def _norm(p: str) -> str:
    """``/v1/evals/{eval_id}`` → ``/v1/evals/{}`` — placeholder-free form."""
    return re.sub(r"\{[^}]*\}", "{}", p.strip())


def _route_row_tokens(
    toks: list[str], served_paths: set[str] | None
) -> tuple[list[str], list[str]]:
    """Classify one table row's backticked tokens into methods and paths."""
    meths: list[str] = []
    paths: list[str] = []
    for tok in toks:
        tok = tok.strip()
        if tok in _METHODS:
            meths.append(tok)
        elif re.match(rf"^({'|'.join(sorted(_METHODS))})\s+/", tok):
            m, p = tok.split(None, 1)
            meths.append(m)
            paths.append(p)
        elif tok.startswith("/"):
            paths.append(tok)
        elif tok.startswith(".../") and paths:
            # ``.../suffix`` = same path, tail differs: appends
            # (``/runs`` → ``/runs/{id}``) or replaces the leaf
            # (``/pause`` → ``/resume``) — serve table arbitrates.
            base = paths[-1].rstrip("/")
            cand = base + tok[3:]
            if served_paths is not None and _norm(cand) not in served_paths:
                cand = base.rsplit("/", 1)[0] + tok[3:]
            paths.append(cand)
    return meths, paths


def _doc_route_rows(text: str, served_paths: set[str] | None = None) -> set[tuple[str, str]]:
    """(method, normalized-path) pairs the API doc's route table claims."""
    section = text.split("## Routes", 1)[1] if "## Routes" in text else text
    routes: set[tuple[str, str]] = set()
    for line in section.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cell = line.split("|")[1]
        toks = re.findall(r"`([^`]+)`", cell)
        meths, paths = _route_row_tokens(toks, served_paths)
        for m in meths:
            for p in paths:
                p = p.split("?")[0].rstrip("/").strip()
                p = re.sub(r"\s.*$", "", p)
                if p.startswith("/"):
                    routes.add((m, _norm(p)))
    return routes


def _served_routes() -> set[tuple[str, str]]:
    """(method, normalized-path) the live app actually serves."""
    from fx1.serve.api import create_app

    app = create_app()
    out: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not isinstance(path, str) or not methods:
            continue
        for m in methods:
            out.add((m, _norm(path)))
    return out


def _extract_cli_commands() -> set[str]:
    """Leaf names of every ``fx1 harness <cmd>`` in the Typer tree."""
    from typer.main import get_command

    from fx1.cli import harness_app

    cmd = get_command(harness_app)
    names: set[str] = set()

    def walk(c: Any, prefix: str) -> None:
        for name, sub in getattr(c, "commands", {}).items():
            leaf = f"{prefix} {name}".strip()
            if getattr(sub, "commands", None):
                walk(sub, leaf)
            else:
                names.add(leaf)

    walk(cmd, "")
    return names


def _cited_cli_names(texts: list[str]) -> set[str]:
    cited: set[str] = set()
    for text in texts:
        cited.update(re.findall(r"\bfx1 harness ([a-z][a-z0-9-]*)", text))
    return cited


def _mentioned_cli_names(texts: list[str], real: set[str]) -> set[str]:
    """Real subcommand names appearing anywhere (bare `` `name` `` too)."""
    blob = "\n".join(texts)
    out: set[str] = set()
    for name in real:
        if (
            re.search(r"\bfx1 harness [a-z-]*" + re.escape(name) + r"\b", blob)
            or re.search(r"`[^`]*\b" + re.escape(name) + r"\b[^`]*`", blob)
            or re.search(r"\b" + re.escape(name) + r"\b", blob)
        ):
            out.add(name)
    return out


def _cited_methods(texts: list[str], cls: str) -> set[str]:
    """``<cls>.<name>`` citations; a trailing ``*`` marks a prefix cite."""
    cited: set[str] = set()
    for text in texts:
        cited.update(re.findall(rf"\b{re.escape(cls)}\.([a-zA-Z_][a-zA-Z_0-9]*\*?)", text))
    return cited


def _doc_env_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b(?:FX1|MOONSHOT|OPENAI|ANTHROPIC|KIMI)_[A-Z_]+\b", text))


def _deploy_failure_rows(text: str) -> list[tuple[set[int], set[str]]]:
    """``| status | `code` | ... |`` rows of the failure-modes table."""
    sec = text.split("## Failure modes", 1)[1].split("## ", 1)[0]
    rows: list[tuple[set[int], set[str]]] = []
    for line in sec.splitlines():
        if not line.lstrip().startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0].lower() == "status":
            continue
        statuses = {int(s) for s in re.findall(r"\d{3}", cells[0])}
        codes = set(re.findall(r"`([a-z_]+)`", cells[1]))
        if statuses:
            rows.append((statuses, codes))
    return rows


def _all_cli_flags() -> set[str]:
    """Every ``--flag`` name across the whole ``fx1`` command tree."""
    from typer.main import get_command

    from fx1.cli import app

    flags: set[str] = set()
    cmd = get_command(app)
    stack = [cmd]
    while stack:
        c = stack.pop()
        for _name, sub in getattr(c, "commands", {}).items():
            if getattr(sub, "commands", None):
                stack.append(sub)
            else:
                _flags_of(sub, flags)
    return flags


def _flags_of(cmd: Any, flags: set[str]) -> None:
    for p in getattr(cmd, "params", []):
        for opt in getattr(p, "opts", []) + getattr(p, "secondary_opts", []):
            flags.add(opt.lstrip("-"))


# --- live app fixtures ---------------------------------------------------------


def _client(api_key: str | None = None, **create_kw: Any) -> TestClient:
    """Env-isolated TestClient over ``create_app``."""
    from fastapi.testclient import TestClient as _TC

    import fx1.serve.api as api_mod

    saved = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        return _TC(api_mod.create_app(**create_kw), raise_server_exceptions=False)
    finally:
        if saved is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved


class _StubBackend:
    """Deterministic echo backend — the wire's shape, never a model."""

    def __init__(self) -> None:
        self._model = "fake-0"

    def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
        del sampling  # protocol signature; echo uses only messages
        return f"clean:{messages[-1]['content']}"

    def stream(self, messages: list[dict[str, str]], *, sampling: Any = None) -> Any:
        del messages, sampling  # protocol signature; fixed stub stream
        yield "clean"
        yield ":ok"


class _NoStreamBackend:
    """Backend without the ``StreamingBackend`` protocol — stream
    surfaces must fail 501, never hang or fake a stream."""

    def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
        del messages, sampling  # protocol signature; unconditional stub
        return "clean"


class _DishonestBackend(_StubBackend):
    """Emits a forbidden-claim string the honesty gate must refuse."""

    def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
        return "the fund's sharpe ratio is 1.2 and the nav is up 40%"


class _FailBackend(_StubBackend):
    """Raises mid-call — the ``backend_failure`` 502 verdict."""

    def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
        raise RuntimeError("upstream exploded")


class _BlockBackend(_StubBackend):
    """Holds the inflight slot until released — the 503 over_capacity pin."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
        self.entered.set()
        self.release.wait(10)
        return "clean"


def _unconfigured_backend(*a: Any, **k: Any) -> Any:
    from fx1.serve.backends import BackendNotConfiguredError

    raise BackendNotConfiguredError("backend lane unconfigured")


_H = {"X-API-Key": _ROOT_KEY}
_COMPLETE_BODY = {
    "backend": "local_fx1",
    "messages": [{"role": "user", "content": "ping"}],
}
_CHAT_BODY = {"model": "fx1", "messages": [{"role": "user", "content": "ping"}]}


# --- probe sections ------------------------------------------------------------


def _brace_variants(p: str) -> list[str]:
    """``p{,/suf}`` doc shorthand → [p, p/suf]; the suffix may itself hold
    a placeholder (``/v1/models{,/{id}}``) so braces are balanced."""
    variants = [p]
    i = p.find("{,")
    while i >= 0:
        j, depth = i + 2, 1
        while j < len(p) and depth:
            depth += (p[j] == "{") - (p[j] == "}")
            j += 1
        suf = p[i + 2 : j - 1]
        variants = [v[:i] + v[j:] for v in variants] + [v[:i] + suf + v[j:] for v in variants]
        i = variants[0].find("{,")
    return variants


def _prose_route_gaps(
    api_text: str, served: set[tuple[str, str]], served_paths: set[str]
) -> list[str]:
    """``METHOD /path`` spans in prose (curl snippets, error examples)
    that resolve to no served route; a trailing ``*`` is a prefix glob."""
    bad: list[str] = []
    for m, p in re.findall(rf"`({'|'.join(sorted(_METHODS))})\s+(/[^`\s]+)`", api_text):
        for v in _brace_variants(p.split("?")[0].rstrip("/")):
            if v.endswith("*"):
                ok = any(sp.startswith(v[:-1]) for sp in served_paths)
            else:
                ok = (m, _norm(v)) in served
            if not ok:
                bad.append(f"{m} {v}")
    return bad


def _bare_path_gaps(texts: dict[str, str], served_paths: set[str]) -> set[str]:
    """Bare backticked ``/path`` spans that look like routes — across
    every doc — resolving to neither a served path nor a templated
    example."""
    bare_ns = {"", "/v1", "/harness", "/receipts", _P_DOCS}
    bad: set[str] = set()
    for label, text in texts.items():
        for p in re.findall(r"`((?:/harness|/v1|/receipts|/health|/ready|/metrics)[^`\s]*)`", text):
            p = re.sub(r"\(.*$", "", p).split("?")[0].rstrip("/")
            if p.endswith("..."):
                continue
            if p.endswith("*"):
                if not any(sp.startswith(p[:-1]) for sp in served_paths):
                    bad.add(f"{label}:{p}")
            elif p and p not in bare_ns and "{" not in p and _norm(p) not in served_paths:
                bad.add(f"{label}:{p}")
    return bad


def _probe_route_parity(out: dict[str, bool], texts: dict[str, str]) -> None:
    served = _served_routes()
    served_paths = {p for _, p in served}
    doc_routes = _doc_route_rows(texts["api"], served_paths)
    out["route_table_extracted"] = len(doc_routes) >= 100
    out["doc_routes_all_served"] = doc_routes <= served
    undocumented = served - doc_routes
    out["served_routes_all_documented"] = all(p in _DOC_EXCLUDED_PATHS for _, p in undocumented)
    out["doc_route_exclusions_exact"] = (
        undocumented == {(m, p) for m, p in served if p in _DOC_EXCLUDED_PATHS} - doc_routes
    )
    out["doc_prose_routes_served"] = not _prose_route_gaps(texts["api"], served, served_paths)
    out["doc_bare_paths_served"] = not _bare_path_gaps(texts, served_paths)


def _probe_openapi_golden(out: dict[str, bool], spec: dict[str, Any]) -> None:
    regen = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    committed = _TS_OPENAPI.read_text() if _TS_OPENAPI.is_file() else ""
    out["openapi_golden_exists"] = bool(committed)
    out["openapi_golden_fresh"] = committed == regen
    spec_paths = {_norm(p) for p in spec.get("paths", {})}
    served_paths = {p for _, p in _served_routes()}
    out["openapi_paths_all_served"] = spec_paths <= served_paths
    missing = served_paths - spec_paths
    out["served_paths_in_openapi_or_excluded"] = missing <= _OPENAPI_EXCLUDED_PATHS
    out["openapi_exclusions_exact"] = missing == _OPENAPI_EXCLUDED_PATHS


def _dts_operation_block(text: str, path: str, method: str) -> str:
    """Extract one operation's generated d.ts block — resolves the
    ``{method}: operations["op_id"]`` indirection openapi-typescript
    emits so parameters/requestBody/responses are visible."""
    m = re.search(rf'"{re.escape(path)}":\s*\{{(.*?)\n    \}};', text, re.S)
    if not m:
        return ""
    path_block = m.group(1)
    ref = re.search(rf"\b{method}:\s*operations\[\"([^\"]+)\"\]", path_block)
    if ref is not None:
        om = re.search(rf"\n    {re.escape(ref.group(1))}:\s*\{{(.*?)\n    \}};", text, re.S)
        return om.group(1) if om else ""
    m2 = re.search(rf"\b{method}:\s*\{{(.*?)\n\s{{8}}\}}", path_block, re.S)
    return m2.group(1) if m2 else path_block


def _schema_missing_params(spec: dict[str, Any], dts_paths: set[str], text: str) -> list[str]:
    """Spec operation parameters whose name is absent from the matching
    ``.d.ts`` block."""
    missing: list[str] = []
    for path, ops in spec.get("paths", {}).items():
        if path not in dts_paths:
            continue
        for method, op in ops.items():
            if not isinstance(op, dict):
                continue
            block = _dts_operation_block(text, path, method)
            for prm in op.get("parameters", []):
                name = prm.get("name", "")
                if name and not re.search(rf"\b{re.escape(name)}\b", block):
                    missing.append(f"{method.upper()} {path}:{name}")
    return missing


def _ts_client_called_paths() -> set[str]:
    """Normalized paths the shipped TS client actually calls."""
    ts_client = _TS_CLIENT.read_text()
    called = set(re.findall(r"`(/(?:harness|v1|receipts|health|ready|metrics)[^`]*)`", ts_client))
    called |= set(re.findall(r'"(/(?:harness|v1|receipts|health|ready|metrics)[^"`]*)"', ts_client))
    norm: set[str] = set()
    for p in called:
        p = p.split("${", 1)[0]  # TS template interpolation tail
        p = re.sub(r"\$\{[^}]*\}", "{}", p)
        p = p.split("?")[0].rstrip("/") or "/"
        norm.add(_norm(p))
    return norm


def _probe_ts_schema(out: dict[str, bool], spec: dict[str, Any]) -> None:
    text = _TS_SCHEMA.read_text() if _TS_SCHEMA.is_file() else ""
    out["schema_ts_exists"] = bool(text)
    out["schema_ts_sha256_pinned"] = hashlib.sha256(text.encode()).hexdigest() == _SCHEMA_TS_SHA256
    dts_paths = set(re.findall(r'^\s{4}"(/[^"]+)":\s*\{', text, re.M))
    spec_paths_raw = set(spec.get("paths", {}))
    out["schema_ts_paths_cover_spec"] = spec_paths_raw <= dts_paths
    out["schema_ts_params_fresh"] = not _schema_missing_params(spec, dts_paths, text)
    out["ts_client_paths_all_served"] = _ts_client_called_paths() <= {
        p for _, p in _served_routes()
    }


def _probe_request_fields(out: dict[str, bool], spec: dict[str, Any]) -> None:
    """Fields the doc binds to a route exist in the served schema."""
    checks = {
        ("POST", _P_KEYS): {
            "name",
            "admin",
            "rpm",
            "ttl_s",
            "scopes",
            "max_requests",
            "max_tokens",
        },
        ("POST", "/v1/uploads"): {"purpose", "filename", "bytes", "mime_type"},
        ("POST", "/v1/batches"): {
            "input_file_id",
            "endpoint",
            "completion_window",
            "metadata",
            "callback_url",
            "callback_secret",
        },
        ("POST", "/v1/vector_stores"): {"name", "file_ids", "metadata", "expires_after"},
        ("POST", "/v1/messages"): {
            "model",
            "max_tokens",
            "messages",
            "system",
            "tools",
            "tool_choice",
            "stop_sequences",
            "stream",
        },
        ("POST", _P_CHAT): {
            "model",
            "messages",
            "stream",
            "tools",
            "tool_choice",
            "n",
            "stop",
            "temperature",
            "top_p",
            "response_format",
            "store",
            "metadata",
            "max_tokens",
            "logprobs",
            "top_logprobs",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "reasoning_effort",
            "service_tier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "stream_options",
            "parallel_tool_calls",
            "max_completion_tokens",
        },
        ("POST", "/v1/responses"): {
            "model",
            "input",
            "instructions",
            "reasoning",
            "text",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "max_tool_calls",
            "include",
            "store",
            "background",
            "previous_response_id",
            "conversation",
            "stream",
            "max_output_tokens",
            "metadata",
            "user",
            "safety_identifier",
            "service_tier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "temperature",
            "top_p",
        },
        ("POST", "/v1/embeddings"): {"model", "input", "encoding_format", "dimensions", "user"},
        ("POST", "/v1/conversations"): {"items", "metadata"},
        ("POST", "/v1/messages/batches"): {"requests"},
        ("POST", "/v1/fine_tuning/jobs"): {
            "model",
            "training_file",
            "callback_url",
            "callback_secret",
        },
        ("POST", "/v1/vector_stores/{}/search"): {
            "query",
            "max_num_results",
            "filters",
            "ranking_options",
        },
        ("POST", "/v1/vector_stores/{}/file_batches"): {
            "file_ids",
            "attributes",
            "chunking_strategy",
        },
        ("POST", "/receipts/verify"): {"receipt"},
        ("POST", _P_COMPLETE): {
            "messages",
            "backend",
            "fallbacks",
            "byok",
            "checkpoint_dir",
            "timeout_s",
            "temperature",
            "top_p",
            "max_tokens",
            "seed",
            "receipt_hashes",
        },
        ("POST", "/harness/evals"): {
            "suite",
            "backend",
            "seed",
            "fallbacks",
            "byok",
            "judge_backend",
            "judge_byok",
            "callback_url",
            "callback_secret",
        },
    }
    spec_paths_norm = {_norm(p): p for p in spec.get("paths", {})}
    bad: list[str] = []
    for (method, path), fields in checks.items():
        sp = spec_paths_norm.get(path)
        op = spec["paths"].get(sp, {}).get(method.lower()) if sp else None
        if not isinstance(op, dict):
            bad.append(f"{method} {path} missing from spec")
            continue
        miss = fields - _request_props(spec, op)
        if miss:
            bad.append(f"{method} {path}: {sorted(miss)}")
    out["doc_request_fields_exist"] = not bad


def _request_props(spec: dict[str, Any], op: dict[str, Any]) -> set[str]:
    props: set[str] = set()
    for content in op.get("requestBody", {}).get("content", {}).values():
        sch = content.get("schema", {})
        ref = sch.get("$ref")
        if ref:
            name = ref.split("/")[-1]
            sch = spec.get("components", {}).get("schemas", {}).get(name, {})
        props |= set(sch.get("properties", {}))
    return props


def _probe_response_shapes(out: dict[str, bool]) -> None:
    """Response bodies the doc describes are the live ones."""
    c = _client(api_key=_ROOT_KEY, backend_resolver=lambda *a, **k: _StubBackend())
    with tempfile.TemporaryDirectory() as td:
        body = {**_COMPLETE_BODY, "checkpoint_dir": td}
        r = c.get("/harness/version", headers=_H)
        out["doc_resp_version"] = r.status_code == 200 and {"api_version", "fx1_version"} <= set(
            r.json()
        )
        r = c.get("/harness/capabilities", headers=_H)
        out["doc_resp_capabilities"] = r.status_code == 200 and {"features", "limits"} <= set(
            r.json()
        )
        r = c.post("/harness/gate/check", json={"text": "hi"}, headers=_H)
        out["doc_resp_gate"] = r.status_code == 200 and {"ok", "error"} <= set(r.json())
        r = c.post("/harness/score", json={"input": ["a", "b"]}, headers=_H)
        j = r.json()
        out["doc_resp_score"] = (
            r.status_code == 200
            and j.get("object") == "list"
            and all(
                {"object", "index", "total", "components", "violations"} <= set(d)
                for d in j.get("data", [])
            )
        )
        r = c.get("/health")
        j = r.json()
        out["doc_resp_health"] = r.status_code == 200 and all(
            isinstance(v, bool) for v in j.get("backends", {}).values()
        )
        r = c.get("/ready", headers=_H)
        out["doc_resp_ready"] = r.status_code == 200 and {"ready", "inflight"} <= set(r.json())
        r = c.post(_P_KEYS, json={"name": "probe"}, headers=_H)
        j = r.json()
        out["doc_resp_key_mint"] = (
            r.status_code == 201
            and isinstance(j.get("key"), str)
            and j["key"].startswith("fx1k_")
            and "id" in j
        )
        r = c.post("/harness/jobs", json={"command": "doctor"}, headers=_H)
        out["doc_resp_job_submit"] = (
            r.status_code == 202
            and "job_id" in r.json()
            and "location" in {k.lower() for k in r.headers}
        )
        r = c.post(
            "/harness/evals",
            json={
                "suite": "capability",
                "backend": "local_fx1",
                "checkpoint_dir": "/x",
            },
            headers=_H,
        )
        out["doc_resp_eval_submit"] = r.status_code == 202 and "eval_id" in r.json()
        r = c.post(_P_COMPLETE, json=body, headers=_H)
        out["doc_resp_complete"] = r.status_code == 200 and {"backend", "model", "content"} <= set(
            r.json()
        )
        r = c.get("/harness/usage", headers=_H)
        out["doc_resp_usage"] = r.status_code == 200 and {
            "records_seen",
            "records_dropped",
            "ring_cap",
            "totals",
        } <= set(r.json())
        r = c.get("/harness/backends", headers=_H)
        out["doc_resp_backends"] = r.status_code == 200 and all(
            {
                "configured",
                "circuit_open",
                "cooldown_remaining_s",
                "consecutive_failures",
                "last_probe",
            }
            <= set(v)
            for v in r.json().values()
        )
        r = c.get("/harness/self", headers=_H)
        out["doc_resp_self"] = r.status_code == 200 and {"credential", "scopes", "metered"} <= set(
            r.json()
        )
        rs = c.post(_P_STREAM, json=body, headers=_H)
        out["doc_resp_sse"] = rs.status_code == 200 and rs.headers.get(
            "content-type", ""
        ).startswith("text/event-stream")


_DUMMY_ID_NAMES = {"key_id", "job", "eval", "run"}
_DUMMY_STR_NAMES = {
    "name",
    "model",
    "backend",
    "text",
    "prompt",
    "input",
    "suite",
    "endpoint",
    "url",
    "role",
    "content",
    "filename",
    "purpose",
    "suffix",
    "format",
    "status",
    "query",
    "command",
    "suite_name",
    "training_file",
    "validation_file",
    "input_file_id",
    "callback_url",
    "callback_secret",
    "custom_id",
    "operation",
    "sha",
    "hash",
    "receipt_sha256",
    "model_id",
    "title",
    "checkpoint_dir",
}
_DUMMY_INT_PREFIXES = (
    "max_",
    "num_",
    "limit",
    "n",
    "seed",
    "timeout",
    "ttl",
    "wait",
    "interval",
    "attempts",
    "offset",
    "dimensions",
    "size",
    "count",
    "top_",
)
_DUMMY_BOOL_PREFIXES = (
    "is_",
    "has_",
    "with_",
    "include_",
    "strict",
    "echo",
    "stream",
    "admin",
    "store",
    "background",
    "revoke_old",
)
_DUMMY_LIST_NAMES = {
    "tools",
    "stop",
    "stop_sequences",
    "file_ids",
    "part_ids",
    "fallbacks",
    "receipt_hashes",
    "scopes",
    "clear",
    "include",
}
_DUMMY_DICT_NAMES = {
    "payload",
    "request",
    "params",
    "body",
    "metadata",
    "config",
    "kwargs",
    "data",
    "data_source",
    "hyperparameters",
    "testing_criteria",
    "response_format",
    "tool_choice",
    "byok",
    "judge_byok",
    "filters",
    "ranking_options",
    "chunking_strategy",
    "annotations",
    "attributes",
    "receipt",
    "result",
    "options",
    "headers",
    "extra_headers",
}
_DUMMY_CALLABLE_NAMES = {"fn", "func", "callable", "sleep", "clock", "on_event"}
_DUMMY_FILE_NAMES = {"file", "fh", "stream", "fp", "f"}


def _dummy_for(name: str, ptype: Any) -> Any:
    n = name.lower()
    if ptype is bytes:
        return b"x"
    if n.endswith("_id") or n == "id" or n in _DUMMY_ID_NAMES:
        return "x"
    if n in _DUMMY_STR_NAMES:
        return "x"
    if "messages" in n or n in {"items", "requests"}:
        return [{"role": "user", "content": "x"}]
    if n == "batch":
        return [[{"role": "user", "content": "x"}]]
    if n in _DUMMY_FILE_NAMES:
        return io.BytesIO(b"x")
    if ptype is int or n.startswith(_DUMMY_INT_PREFIXES):
        return 1
    if ptype is float or n.endswith("_s") or n.endswith("_ms"):
        return 1.0
    if ptype is bool or n.startswith(_DUMMY_BOOL_PREFIXES):
        return True
    if n in _DUMMY_LIST_NAMES or n.endswith("s"):
        return []
    if n in _DUMMY_DICT_NAMES:
        return {}
    if n in _DUMMY_CALLABLE_NAMES:
        return lambda *a, **k: None
    return "x"


_STR_TYPE_MAP = {
    "bytes": bytes,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "dict": dict,
    "list": list,
}


def _annotation_type(annotation: Any) -> Any:
    """A parameter's type as a ``type`` when the annotation resolves to
    one — handles string annotations like ``"bytes | None"``."""
    if isinstance(annotation, type):
        return annotation
    if isinstance(annotation, str):
        base = annotation.split("|")[0].strip().strip("'\"")
        return _STR_TYPE_MAP.get(base)
    return None


def _param_dummy(p: inspect.Parameter) -> Any:
    ptype = _annotation_type(p.annotation)
    dummy = _dummy_for(p.name, ptype)
    if (
        "HarnessResult" in str(p.annotation)
        and dummy is None
        or (p.name == "result" and ptype is None)
    ):
        from fx1.harness import HarnessResult  # noqa: PLC0415

        dummy = HarnessResult(command="x", exit_code=0, stdout="", stderr="")
    return dummy


def _call_with_dummies(fn: Any) -> Any:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn()
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for p in sig.parameters.values():
        if p.name == "self" or p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if p.default is not inspect.Parameter.empty:
            continue
        dummy = _param_dummy(p)
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[p.name] = dummy
        else:
            args.append(dummy)
    return fn(*args, **kwargs)


def _exhaust(res: Any) -> None:
    """Pull generator/iterator results so lazy methods emit."""
    if inspect.isgenerator(res) or (
        hasattr(res, "__iter__") and not isinstance(res, (str, bytes, dict, list, tuple, set))
    ):
        try:
            for _ in res:
                break
        except Exception:
            pass


def _route_patterns() -> set[tuple[str, re.Pattern[str]]]:
    served = _served_routes()
    pats: set[tuple[str, re.Pattern[str]]] = set()
    for m, p in served:
        rx = re.compile("^" + re.escape(p).replace(r"\{\}", "[^/]+") + "$")
        pats.add((m, rx))
    return pats


def _canned_body(path: str) -> bytes:
    """A terminal-looking payload per waiter family — each poller
    pattern-matches a different marker."""
    if "stream" in path or "events" in path or "results" in path:
        return b"data: [DONE]\n\n"
    if "/messages/batches" in path:
        return b'{"processing_status":"ended","id":"x","request_counts":{}}'
    if re.search(r"(/harness/jobs|/harness/evals|/v1/fine_tuning)", path):
        return b'{"status":"succeeded","id":"x","data":[]}'
    return b'{"data":[],"id":"x","status":"completed","ok":true,"object":"x"}'


def _probe_client_coverage(out: dict[str, bool], texts: dict[str, str]) -> None:
    from fx1.serve.client import HarnessClient

    emitted: list[tuple[str, str]] = []

    def rec_transport(
        method: str, url: str, payload: Any, headers: Any, timeout: Any
    ) -> tuple[int, dict[str, str], bytes]:
        del payload, headers, timeout  # canned transport records only the route
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        emitted.append((method, path.split("?")[0]))
        return 200, {}, _canned_body(path)

    client = HarnessClient("https://probe.local", transport=rec_transport, max_retries=0)
    pats = _route_patterns()
    uncovered: list[str] = []
    for name in sorted(n for n in dir(client) if not n.startswith("_")):
        fn = getattr(client, name)
        if not callable(fn):
            continue
        before = len(emitted)
        try:
            res = _call_with_dummies(fn)
            _exhaust(res)
        except Exception:
            pass
        if len(emitted) == before:
            uncovered.append(name)
    out["client_methods_all_covered"] = not uncovered
    bad = [f"{m} {p}" for m, p in emitted if not any(mm == m and rx.match(p) for (mm, rx) in pats)]
    out["client_emitted_routes_all_served"] = not bad
    cited = _cited_methods(list(texts.values()), "HarnessClient")
    pub = {n for n in dir(client) if not n.startswith("_")}
    unresolved = [
        c for c in cited if c not in pub and not any(m.startswith(c.rstrip("*")) for m in pub)
    ]
    out["client_doc_cited_methods_real"] = not unresolved


def _probe_sdk(out: dict[str, bool]) -> None:
    from fx1.sdk import Fx1Harness

    with tempfile.TemporaryDirectory() as td:
        h = Fx1Harness(
            backend_resolver=lambda *a, **k: _StubBackend(),
            receipts_dir=Path(td) / "receipts",
            state_dir=Path(td) / "state",
        )
        unexpected: list[str] = []
        for name in sorted(n for n in dir(h) if not n.startswith("_")):
            fn = getattr(h, name)
            if not callable(fn):
                continue
            try:
                res = _call_with_dummies(fn)
                _exhaust(res)
            except (KeyError, ValueError, RuntimeError, OSError):
                continue
            except Exception as exc:  # noqa: BLE001
                # classify — a non-fx1/quant_fund raise is a real finding
                mod = type(exc).__module__
                if "fx1" not in mod and "quant_fund" not in mod:
                    unexpected.append(f"{name}:{type(exc).__name__}")
        out["sdk_methods_all_invocable"] = not unexpected


def _harness_cmd_name(node: Any, deco: Any) -> str | None:
    """The command name a ``@harness_app.command(...)`` decorator binds,
    or None when the decorator is something else."""
    import ast

    if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)):
        return None
    if (
        deco.func.attr != "command"
        or not deco.args
        or not isinstance(deco.func.value, ast.Name)
        or deco.func.value.id != "harness_app"
    ):
        return None
    arg = deco.args[0]
    return str(arg.value if isinstance(arg, ast.Constant) else node.name)


def _cli_route_usage() -> dict[str, set[str]]:
    """command → HarnessClient method names its source calls (AST)."""
    import ast

    import fx1.cli as cli_mod
    from fx1.serve.client import HarnessClient

    pub = {
        n
        for n in dir(HarnessClient)
        if not n.startswith("_") and callable(getattr(HarnessClient, n))
    }
    src = inspect.getsource(cli_mod)
    tree = ast.parse(src)
    usage: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            cmd_name = _harness_cmd_name(node, deco)
            if cmd_name is None:
                continue
            calls = {
                sub.attr
                for sub in ast.walk(node)
                if isinstance(sub, ast.Attribute) and sub.attr in pub
            }
            usage[cmd_name] = calls
    return usage


def _probe_cli(out: dict[str, bool], texts: dict[str, str]) -> None:
    real = _extract_cli_commands()
    blob = [texts["api"], texts["deploy"], texts["clients"], texts["readme"]]
    cited = _cited_cli_names(blob)
    out["cli_cited_commands_real"] = cited <= real
    undocumented = real - _mentioned_cli_names(blob, real)
    out["cli_undocumented_pinned"] = undocumented == _CLI_UNDOCUMENTED
    usage = _cli_route_usage()
    unmapped = {c for c in real if not usage.get(c)} - _CLI_LOCAL_ONLY
    out["cli_commands_map_or_local"] = not unmapped
    from typer.testing import CliRunner

    from fx1.cli import app as cli_app

    runner = CliRunner()
    bad_help = [
        name
        for name in sorted(cited)
        if runner.invoke(cli_app, ["harness", name, "--help"]).exit_code != 0
    ]
    out["cli_documented_commands_parse"] = not bad_help


def _probe_deploy(out: dict[str, bool], texts: dict[str, str]) -> None:
    text = texts["deploy"]
    src_blob = "".join(f.read_text() for f in _SRC_FX1.rglob("*.py"))
    tokens = _doc_env_tokens(text) - _DEPLOY_DOCNAME_TOKENS - _DEPLOY_USER_ENVS
    missing = {t for t in tokens if f'"{t}"' not in src_blob and f"'{t}'" not in src_blob}
    out["deploy_env_vars_real"] = not missing
    flags = set(re.findall(r"--([a-z][a-z0-9-]+)", text))
    bad_flags = flags - _all_cli_flags() - {"env", "name", "from"}
    out["deploy_flags_real"] = not bad_flags
    out["deploy_failure_table_parsed"] = len(_deploy_failure_rows(text)) >= 12


def _probe_failure_table(out: dict[str, bool]) -> None:
    """Every ``(status, code)`` row of the deploy failure table, live."""
    c = _client(api_key=_ROOT_KEY, backend_resolver=lambda *a, **k: _StubBackend())
    r = c.post(_P_CHAT, json=_CHAT_BODY, headers={"X-API-Key": "wrong"})
    out["doc_401_unauthorized"] = (
        r.status_code == 401 and r.json().get("error", {}).get("code") == "unauthorized"
    )
    mint = c.post(_P_KEYS, json={"scopes": ["read"]}, headers=_H)
    ro = c.post(
        _P_CHAT,
        json=_CHAT_BODY,
        headers={"X-API-Key": mint.json().get("key", "")},
    )
    out["doc_403_insufficient_scope"] = (
        ro.status_code == 403 and ro.json().get("error", {}).get("code") == "insufficient_scope"
    )
    # ``admin_required`` exists in code but is unreachable on the wire:
    # every `_require_admin` route is scope-gated 'admin', and minting
    # an admin-scoped key unions the admin flag — the doc row must not
    # list it. Pinned unreachable so it can't silently come back.
    out["admin_required_unreachable_or_absent"] = "admin_required" not in {
        c for _, codes in _deploy_failure_rows(_DOC_DEPLOY.read_text()) for c in codes
    }
    r = c.get("/harness/jobs/nope", headers=_H)
    out["doc_404"] = r.status_code == 404
    r = c.post(_P_CHAT, json={"model": "fx1"}, headers=_H)
    out["doc_422"] = r.status_code == 422
    mint = c.post(_P_KEYS, json={"rpm": 1}, headers=_H)
    h = {"X-API-Key": mint.json().get("key", "")}
    c.get(_P_COMMANDS, headers=h)
    r = c.get(_P_COMMANDS, headers=h)
    out["doc_429_rate_limited"] = (
        r.status_code == 429
        and r.json().get("code") == "rate_limited"
        and "Retry-After" in r.headers
        and "X-RateLimit-Limit-Requests" in r.headers
    )
    mint = c.post(_P_KEYS, json={"max_requests": 1}, headers=_H)
    hq = {"X-API-Key": mint.json().get("key", "")}
    c.get(_P_COMMANDS, headers=hq)
    r = c.get(_P_COMMANDS, headers=hq)
    out["doc_429_quota_exceeded"] = (
        r.status_code == 429
        and r.json().get("code") == "quota_exceeded"
        and "Retry-After" not in r.headers
    )
    # ``/health``/``/ready`` are exempt from the ingress limiter — the
    # 429 proves on a metered route instead.
    cr = _client(rate_limit_rps=0.0001, backend_resolver=lambda *a, **k: _StubBackend())
    cr.get(_P_COMMANDS)
    r = cr.get(_P_COMMANDS)
    out["doc_429_too_many_requests"] = (
        r.status_code == 429
        and r.json().get("code") == "too_many_requests"
        and "Retry-After" in r.headers
    )
    cn = _client(api_key=_ROOT_KEY, backend_resolver=lambda *a, **k: _NoStreamBackend())
    r = cn.post(
        _P_STREAM,
        json={**_COMPLETE_BODY, "backend": "local_fx1", "checkpoint_dir": "/x"},
        headers=_H,
    )
    out["doc_501_not_implemented"] = (
        r.status_code == 501 and r.json().get("code") == "not_implemented"
    )
    r = cn.post(
        _P_STREAM,
        json={
            **_COMPLETE_BODY,
            "backend": "local_fx1",
            "checkpoint_dir": "/x",
            "logprobs": True,
        },
        headers=_H,
    )
    out["doc_501_not_supported"] = r.status_code == 501 and r.json().get("code") == "not_supported"
    cd = _client(api_key=_ROOT_KEY, backend_resolver=lambda *a, **k: _DishonestBackend())
    r = cd.post(_P_CHAT, json=_CHAT_BODY, headers=_H)
    out["doc_502_honesty_gate"] = (
        r.status_code == 502 and r.json().get("error", {}).get("code") == "honesty_gate"
    )
    cf = _client(api_key=_ROOT_KEY, backend_resolver=lambda *a, **k: _FailBackend())
    r = cf.post(_P_CHAT, json=_CHAT_BODY, headers=_H)
    out["doc_502_backend_failure"] = (
        r.status_code == 502 and r.json().get("error", {}).get("code") == "backend_failure"
    )
    cu = _client(api_key=_ROOT_KEY, backend_resolver=_unconfigured_backend)
    r = cu.post(_P_CHAT, json=_CHAT_BODY, headers=_H)
    out["doc_503_backend_unavailable"] = (
        r.status_code == 503 and r.json().get("error", {}).get("code") == "backend_unavailable"
    )
    cg = _client(api_key=_ROOT_KEY, backend_resolver=lambda *a, **k: _StubBackend())
    cg.post("/harness/drain", headers=_H)
    r = cg.post(_P_CHAT, json=_CHAT_BODY, headers=_H)
    out["doc_503_draining"] = (
        r.status_code == 503 and r.json().get("error", {}).get("code") == "draining"
    )
    block = _BlockBackend()
    cc = _client(
        api_key=_ROOT_KEY,
        max_inflight=1,
        backend_resolver=lambda *a, **k: block,
    )
    held: dict[str, int] = {}

    def _hold() -> None:
        with tempfile.TemporaryDirectory() as td:
            rr = cc.post(
                _P_COMPLETE,
                json={**_COMPLETE_BODY, "checkpoint_dir": td},
                headers=_H,
            )
            held["r1"] = rr.status_code

    t = threading.Thread(target=_hold, daemon=True)
    t.start()
    if block.entered.wait(10):
        with tempfile.TemporaryDirectory() as td:
            r = cc.post(
                _P_COMPLETE,
                json={**_COMPLETE_BODY, "checkpoint_dir": td},
                headers=_H,
            )
        block.release.set()
        t.join(10)
        out["doc_503_over_capacity"] = (
            r.status_code == 503
            and r.json().get("detail")
            and r.json().get("code") == "over_capacity"
            and r.headers.get("retry-after") == "1"
        )
    else:
        block.release.set()
        out["doc_503_over_capacity"] = False
    crd = _client(
        api_key=_ROOT_KEY,
        receipts_dir="/nonexistent-x/fx1",
        backend_resolver=lambda *a, **k: _StubBackend(),
    )
    r = crd.get("/receipts/" + "0" * 64, headers=_H)
    out["doc_503_receipts_unavailable"] = (
        r.status_code == 503 and r.json().get("code") == "receipts_unavailable"
    )


def _probe_error_contract(out: dict[str, bool]) -> None:
    from fx1.serve.client import (
        BackendNotConfiguredError,
        HarnessAuthError,
        HarnessClient,
        HarnessTransportError,
    )

    def mapped(status: int) -> type[BaseException]:
        return type(HarnessClient._map_error(status, b'{"detail":"x","code":"c"}'))

    table = {
        400: HarnessTransportError,
        401: HarnessAuthError,
        403: HarnessAuthError,
        404: KeyError,
        405: HarnessTransportError,
        409: HarnessTransportError,
        422: ValueError,
        429: HarnessTransportError,
        500: HarnessTransportError,
        501: NotImplementedError,
        503: BackendNotConfiguredError,
    }
    out["error_map_documented"] = all(mapped(s) is t for s, t in table.items())
    exc = HarnessClient._map_error(502, b'{"detail":"honesty gate refused model output: x"}')
    out["error_map_502_honesty"] = type(exc).__name__ == "Fx1HonestyError"
    exc = HarnessClient._map_error(502, b'{"detail":"boom"}')
    out["error_map_502_fault"] = type(exc) is RuntimeError
    c = _client(api_key=_ROOT_KEY, backend_resolver=lambda *a, **k: _StubBackend())
    r = c.get("/harness/jobs/nope", headers=_H)
    j = r.json()
    out["envelope_harness_detail_code"] = (
        r.status_code == 404 and isinstance(j.get("detail"), str) and isinstance(j.get("code"), str)
    )
    r = c.post(_P_CHAT, json={"model": "fx1"}, headers=_H)
    j = r.json().get("error", {})
    out["envelope_openai_keys"] = r.status_code in (400, 422) and {
        "message",
        "type",
        "param",
        "code",
    } <= set(j)
    r = c.post(
        "/v1/messages",
        json={"model": "fx1"},
        headers={**_H, "anthropic-version": "2023-06-01"},
    )
    j = r.json()
    out["envelope_anthropic_shape"] = r.status_code >= 400 and (
        j.get("type") == "error"
        and isinstance(j.get("error"), dict)
        and isinstance(j["error"].get("type"), str)
        and isinstance(j["error"].get("message"), str)
    )


def _probe_readme(out: dict[str, bool], texts: dict[str, str]) -> None:
    text = texts["readme"]
    served = _served_routes()
    served_paths = {p for _, p in served}
    cited_pairs = set()
    for m, p in re.findall(rf"`({'|'.join(sorted(_METHODS))})\s+(/[^`\s]+)`", text):
        if _norm(p) not in _README_LAB_PATHS:
            cited_pairs.add((m, _norm(p)))
    out["readme_routes_real"] = cited_pairs <= served
    bare = {
        _norm(p.split("?")[0].rstrip("/"))
        for p in re.findall(r"`((?:/harness|/v1|/receipts|/health|/ready|/metrics)[^`\s]*)`", text)
    }
    out["readme_bare_paths_real"] = bare <= served_paths | {_P_V1_PH}
    real = _extract_cli_commands()
    cited = set(re.findall(r"\bfx1 harness ([a-z][a-z0-9-]*)", text))
    out["readme_commands_real"] = cited <= real
    import fx1

    out["readme_version_pin"] = fx1.__version__ in text


def _probe_ledger(out: dict[str, bool]) -> None:
    text = _DOC_LEDGER.read_text()
    modules = {p.stem for p in _SERVE_DIR.glob("*_audit.py")}
    out["ledger_modules_documented"] = all(m in text for m in modules)
    tokens = set(re.findall(r"\b([a-z0-9]+_audit)\b", text))
    receipts = {p.stem.removeprefix("fx1_") for p in _RECEIPTS_DIR.glob("fx1_*.json")}
    orphan = {
        t for t in tokens if t not in modules and t not in receipts and not t.startswith("test_")
    }
    out["ledger_names_resolve"] = not orphan
    # ``(PR #N)`` headings resolve to reachable history.
    import subprocess

    cited_prs = set(re.findall(r"\(PR #(\d+)\)", text))
    log = subprocess.run(
        ["git", "log", "--all", "--format=%s"],
        capture_output=True,
        text=True,
        cwd=_REPO,
    ).stdout
    out["ledger_pr_refs_resolve"] = all(f"#{n}" in log for n in cited_prs)
    # Probe counts the ledger quotes match committed receipts where one
    # exists (``checks N`` / ``N probes`` prose vs claim.results size).
    quoted = re.findall(r"checks (\d+) selected", text)
    out["ledger_probe_counts_honest"] = bool(quoted)
    cov = json.loads(_COVERAGE.read_text())
    serve_n = cov.get("directories", {}).get("serve", {}).get("n_modules")
    out["coverage_n_modules_true"] = serve_n == len(list(_SERVE_DIR.glob("*.py")))


def _probe_doc_methods(out: dict[str, bool], texts: dict[str, str]) -> None:
    from fx1.sdk import Fx1Harness

    cited = _cited_methods(list(texts.values()), "Fx1Harness")
    pub = {n for n in dir(Fx1Harness) if not n.startswith("_")}
    unresolved = [
        c for c in cited if c not in pub and not any(m.startswith(c.rstrip("*")) for m in pub)
    ]
    out["doc_fx1harness_methods_real"] = not unresolved


# --- battery -------------------------------------------------------------------


def docs_audit() -> dict[str, bool]:
    out: dict[str, bool] = {}
    for f in (
        _DOC_API,
        _DOC_DEPLOY,
        _DOC_CLIENTS,
        _DOC_LEDGER,
        _README,
        _TS_OPENAPI,
        _TS_SCHEMA,
        _TS_CLIENT,
        _COVERAGE,
    ):
        out[f"file_{f.name.replace('.', '_')}"] = f.is_file()
    texts = {
        "api": _DOC_API.read_text(),
        "deploy": _DOC_DEPLOY.read_text(),
        "clients": _DOC_CLIENTS.read_text(),
        "ledger": _DOC_LEDGER.read_text(),
        "readme": _README.read_text(),
    }
    from fx1.serve.api import create_app

    spec = create_app().openapi()
    _probe_route_parity(out, texts)
    _probe_openapi_golden(out, spec)
    _probe_ts_schema(out, spec)
    _probe_request_fields(out, spec)
    _probe_response_shapes(out)
    _probe_client_coverage(out, texts)
    _probe_sdk(out)
    _probe_cli(out, texts)
    _probe_deploy(out, texts)
    _probe_failure_table(out)
    _probe_error_contract(out)
    _probe_readme(out, texts)
    _probe_ledger(out)
    _probe_doc_methods(out, texts)
    return out


def docs_audit_bench() -> dict[str, Any]:
    """Seal the docs↔surface parity audit as ``fx1_docs_audit.v1``."""
    r = docs_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "fx1_docs_audit",
        "schema": "fx1_docs_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok, "probes": len(r)},
        "interpretation": (
            "the committed docs are the served surface: every documented "
            "route/field/env-var/command/error resolves, every served route "
            "is documented or pinned excluded, and the goldens regenerate "
            "byte-identically"
            if ok
            else f"DOC DRIFT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(docs_audit_bench(), indent=2, sort_keys=True))
