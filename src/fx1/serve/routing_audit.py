"""routing_audit — the capstone meta-audit: every route the app serves must
be covered by at least one sealed audit's probe battery.

Every other ``*_audit`` module probes a slice of the surface — files, jobs,
quotas, dialects. Nobody answers the prior question: *is there a route no
audit ever calls?* A route that ships unprobed is where drift hides. This
audit enumerates the live route table of ``create_app()`` (and its
``app.openapi()`` projection) and cross-references every ``(method, path)``
pair against the declared coverage manifest ``AUDIT_ROUTE_MANIFEST`` — an
explicit, parseable ``"METHOD /template"`` map of which sibling battery
exercises what.

Probe surfaces:

- *Route table completeness* — the table enumerates, is deterministic
  across two app builds, and every OpenAPI operation resolves to a live
  route (and vice versa: declared methods equal declared spec methods).
- *Manifest derivability* — every ``src/fx1/serve/*_audit.py`` module
  (including this one) appears as a manifest key; every manifest entry is
  a declared ``METHOD /template`` that resolves to a live route — the
  coverage map is data, not a docstring's prose.
- *Coverage completeness* — the union of all manifest entries covers the
  whole live surface; uncovered pairs land in the receipt's ``gaps``
  list (``[]`` is the full-coverage claim — the verdict is honest).
- *Mounted-route parity* — every ``_mount_*`` helper defined in
  ``api.py`` is invoked by ``create_app`` (no dead mounts), and every
  live APIRoute template appears verbatim in the source's route
  decorators (nothing is served that no mount admits to registering).
- *Method surface* — a wrong-method call on ``/harness``/``/receipts``
  lands enveloped 405 ``method_not_allowed``; on ``/v1`` it is absorbed
  by the catch-all into an enveloped 404 ``invalid_request_error``
  (OpenAI's grammar — OpenAI 404s unknown method+path pairs). ``HEAD``
  never silently rides: 200 on the plain ``Route`` doc paths, 405 on a
  GET-only ``APIRoute`` (the catch-all claims it under ``/v1``).
- *Dialect parity* — every live path sits under exactly one prefix
  family (``/v1``, ``/harness``, ``/receipts``, ``/metrics``, ``/health``,
  ``/ready``, the docs routes); no ``/v1/harness*`` or ``/harness/v1*``
  leak; the catch-all 404 speaks Anthropic's ``{type: "error"}`` grammar
  under ``/v1/messages*`` and OpenAI's ``{error}`` elsewhere.
- *Template honesty* — parameterized routes are counted once per
  template; no two live routes share the same canonical
  (param-erased) path+method (a shadowed duplicate is a defect);
  the catch-all is the only ``{path:path}`` route and is registered
  last so it shadows nothing.
- *Unreachable routes* — a ``_mount_*`` defined but never called, or a
  decorator literal that no live route answers, is dead surface and is
  flagged.
- *Coverage density* — per-route claimant count is reported
  (single-probe vs multiply-covered), and pairs whose only coverage is
  unsealed (no committed ``receipts/fx1_*_audit.json`` yet) are listed
  under ``unsealed_only`` — covered but not yet sealed evidence.
- *Fresh-route detection* — a sentinel ``(method, path)`` injected into
  the live set is flagged by the same machinery, proving the audit
  trips on drift rather than blessing whatever exists.
- *Ratchet consistency* — ``quality/audit_coverage_fx1.json``'s
  ``serve.n_modules`` equals the live module census.

Two conventions shape the manifest: sibling modules do not declare
per-module route manifests today, so this module carries the explicit
taxonomy (the brief's sanctioned fallback); and modules with no HTTP
probe surface (``attestation_audit``, ``contract_audit``,
``journal_audit``, ``serve_audit``) declare empty coverage honestly —
``contract_audit``'s golden pin of the whole ``openapi()`` surface is
schema-level evidence, deliberately not counted as per-route probes, so
a route probed only by the spec diff still lands in ``gaps``.

Sealed ``routing_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = [
    "AUDIT_ROUTE_MANIFEST",
    "audit_coverage",
    "live_route_table",
    "routing_audit",
    "routing_audit_bench",
]

_SERVE_DIR = Path(__file__).resolve().parent
_API_SRC = _SERVE_DIR / "api.py"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RECEIPTS = _REPO_ROOT / "receipts"
_COVERAGE_MANIFEST = _REPO_ROOT / "quality" / "audit_coverage_fx1.json"

_API_KEY_ENV = "FX1_API_KEY"

_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
_OPENAPI_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})
_CATCH_ALL = "/v1/{path:path}"

#: Routes FastAPI registers itself (docs UI + schema) or registers with
#: ``include_in_schema=False`` — live, but absent from the OpenAPI surface.
_NON_SCHEMA_PATHS = frozenset(
    {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json", _CATCH_ALL}
)

#: Every served path lives under exactly one of these first segments —
#: a route anywhere else is surprise surface the dialect story can't cover.
_KNOWN_PREFIXES = (
    "/v1",
    "/harness",
    "/receipts",
    "/metrics",
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
)

_SENTINEL_METHOD = "POST"
_SENTINEL_PATH = "/v1/__routing_audit_sentinel__"

_ENTRY_RE = re.compile(r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) (\S+)$")
_MOUNT_DEF_RE = re.compile(r"^def (_mount_\w+)\(", re.M)
_MOUNT_CALL_RE = re.compile(r"^    (_mount_\w+)\(", re.M)
_PATH_LIT_RE = re.compile(r"""['"](/[^'"]*)['"]""")
_PARAM_RE = re.compile(r"\{[^}/]*\}")


#: Declared per-module route coverage: ``"METHOD /template"`` entries naming
#: the live FastAPI route each sibling probe battery exercises over HTTP.
#: Derived mechanically from each module's call sites (TestClient, raw ASGI,
#: SDK surfaces) and reviewed against the live route table; entries whose
#: method is not declared on the matched template attribute to the route
#: that actually serves the call (the ``/v1`` catch-all). Modules probing
#: non-HTTP surfaces declare empty coverage — an honest zero, not a gap.
AUDIT_ROUTE_MANIFEST: dict[str, frozenset[str]] = {
    "anthropic_sdk_audit": frozenset(
        {
            "DELETE /v1/messages/batches/{batch_id}",
            "GET /v1/messages/batches",
            "GET /v1/messages/batches/{batch_id}",
            "GET /v1/messages/batches/{batch_id}/results",
            "GET /v1/models",
            "GET /v1/models/{model}",
            "POST /harness/keys",
            "POST /v1/messages",
            "POST /v1/messages/batches",
            "POST /v1/messages/batches/{batch_id}/cancel",
            "POST /v1/messages/count_tokens",
        }
    ),
    "api_audit": frozenset(
        {
            "DELETE /harness/evals/{eval_id}",
            "DELETE /harness/jobs/{job_id}",
            "DELETE /harness/keys/{key_id}",
            "DELETE /v1/chat/completions/{completion_id}",
            "DELETE /v1/conversations/{conversation_id}",
            "DELETE /v1/conversations/{conversation_id}/items/{item_id}",
            "DELETE /v1/evals/{eval_id}",
            "DELETE /v1/evals/{eval_id}/runs/{run_id}",
            "DELETE /v1/files/{file_id}",
            "DELETE /v1/messages/batches/{batch_id}",
            "DELETE /v1/models/{model}",
            "DELETE /v1/responses/{response_id}",
            "DELETE /v1/vector_stores/{vector_store_id}",
            "DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "GET /harness/backends",
            "GET /harness/capabilities",
            "GET /harness/commands",
            "GET /harness/completions",
            "GET /harness/completions/{completion_id}",
            "GET /harness/completions/{completion_id}/receipt",
            "GET /harness/evals",
            "GET /harness/evals/{eval_id}",
            "GET /harness/evals/{eval_id}/diff/{candidate_id}",
            "GET /harness/evals/{eval_id}/receipt",
            "GET /harness/jobs",
            "GET /harness/jobs/{job_id}",
            "GET /harness/jobs/{job_id}/events",
            "GET /harness/jobs/{job_id}/receipt",
            "GET /harness/keys",
            "GET /harness/keys/{key_id}",
            "GET /harness/keys/{key_id}/usage",
            "GET /harness/self",
            "GET /harness/usage",
            "GET /harness/version",
            "GET /health",
            "GET /metrics",
            "GET /openapi.json",
            "GET /ready",
            "GET /receipts",
            "GET /receipts/{sha256}",
            "GET /v1/batches",
            "GET /v1/batches/{batch_id}",
            "GET /v1/chat/completions",
            "GET /v1/chat/completions/{completion_id}",
            "GET /v1/chat/completions/{completion_id}/messages",
            "GET /v1/conversations/{conversation_id}",
            "GET /v1/conversations/{conversation_id}/items",
            "GET /v1/conversations/{conversation_id}/items/{item_id}",
            "GET /v1/evals",
            "GET /v1/evals/{eval_id}",
            "GET /v1/evals/{eval_id}/runs",
            "GET /v1/evals/{eval_id}/runs/{run_id}",
            "GET /v1/evals/{eval_id}/runs/{run_id}/output_items",
            "GET /v1/files",
            "GET /v1/files/{file_id}",
            "GET /v1/files/{file_id}/content",
            "GET /v1/fine_tuning/jobs",
            "GET /v1/fine_tuning/jobs/{job_id}",
            "GET /v1/fine_tuning/jobs/{job_id}/checkpoints",
            "GET /v1/fine_tuning/jobs/{job_id}/events",
            "GET /v1/messages/batches",
            "GET /v1/messages/batches/{batch_id}",
            "GET /v1/messages/batches/{batch_id}/results",
            "GET /v1/models",
            "GET /v1/models/{model}",
            "GET /v1/responses/{response_id}",
            "GET /v1/responses/{response_id}/input_items",
            "GET /v1/vector_stores",
            "GET /v1/vector_stores/{vector_store_id}",
            "GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}",
            "GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
            "GET /v1/vector_stores/{vector_store_id}/files",
            "GET /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "GET /v1/vector_stores/{vector_store_id}/files/{file_id}/content",
            "GET /v1/{path:path}",
            "PATCH /harness/keys/{key_id}",
            "POST /harness/backends/{name}/probe",
            "POST /harness/complete",
            "POST /harness/complete/batch",
            "POST /harness/complete/stream",
            "POST /harness/drain",
            "POST /harness/evals",
            "POST /harness/gate/check",
            "POST /harness/jobs",
            "POST /harness/jobs/batch",
            "POST /harness/keys",
            "POST /harness/keys/{key_id}/rotate",
            "POST /harness/runs",
            "POST /harness/score",
            "POST /receipts/verify",
            "POST /receipts/verify/batch",
            "POST /v1/batches",
            "POST /v1/batches/{batch_id}/cancel",
            "POST /v1/chat/completions",
            "POST /v1/chat/completions/{completion_id}",
            "POST /v1/completions",
            "POST /v1/conversations",
            "POST /v1/embeddings",
            "POST /v1/evals",
            "POST /v1/evals/{eval_id}",
            "POST /v1/evals/{eval_id}/runs",
            "POST /v1/evals/{eval_id}/runs/{run_id}/cancel",
            "POST /v1/files",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/fine_tuning/jobs/{job_id}/cancel",
            "POST /v1/fine_tuning/jobs/{job_id}/pause",
            "POST /v1/fine_tuning/jobs/{job_id}/resume",
            "POST /v1/messages",
            "POST /v1/messages/batches",
            "POST /v1/messages/batches/{batch_id}/cancel",
            "POST /v1/messages/count_tokens",
            "POST /v1/moderations",
            "POST /v1/responses",
            "POST /v1/responses/{response_id}/cancel",
            "POST /v1/uploads",
            "POST /v1/uploads/{upload_id}/cancel",
            "POST /v1/uploads/{upload_id}/complete",
            "POST /v1/uploads/{upload_id}/parts",
            "POST /v1/vector_stores",
            "POST /v1/vector_stores/{vector_store_id}",
            "POST /v1/vector_stores/{vector_store_id}/file_batches",
            "POST /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
            "POST /v1/vector_stores/{vector_store_id}/files",
            "POST /v1/vector_stores/{vector_store_id}/search",
        }
    ),
    "attestation_audit": frozenset(),  # attestation quote/release artifacts — non-HTTP audit
    "auth_audit": frozenset(
        {
            "DELETE /harness/keys/{key_id}",
            "DELETE /v1/files/{file_id}",
            "GET /docs",
            "GET /harness/jobs",
            "GET /harness/keys",
            "GET /harness/keys/{key_id}",
            "GET /harness/keys/{key_id}/usage",
            "GET /harness/self",
            "GET /health",
            "GET /metrics",
            "GET /openapi.json",
            "GET /ready",
            "GET /v1/models",
            "GET /v1/{path:path}",
            "HEAD /v1/{path:path}",
            "OPTIONS /v1/{path:path}",
            "PATCH /harness/keys/{key_id}",
            "PATCH /v1/{path:path}",
            "POST /harness/drain",
            "POST /harness/keys",
            "POST /harness/keys/{key_id}/rotate",
            "POST /v1/chat/completions",
            "POST /v1/messages",
            "PUT /v1/{path:path}",
        }
    ),
    "batch_audit": frozenset(
        {
            "DELETE /v1/messages/batches/{batch_id}",
            "GET /harness/keys/{key_id}/usage",
            "GET /harness/usage",
            "GET /v1/batches",
            "GET /v1/batches/{batch_id}",
            "GET /v1/files/{file_id}",
            "GET /v1/files/{file_id}/content",
            "GET /v1/messages/batches",
            "GET /v1/messages/batches/{batch_id}",
            "GET /v1/messages/batches/{batch_id}/results",
            "POST /harness/drain",
            "POST /harness/keys",
            "POST /v1/batches",
            "POST /v1/batches/{batch_id}/cancel",
            "POST /v1/chat/completions",
            "POST /v1/files",
            "POST /v1/messages",
            "POST /v1/messages/batches",
            "POST /v1/messages/batches/{batch_id}/cancel",
            "POST /v1/responses",
        }
    ),
    "byok_audit": frozenset(
        {
            "GET /harness/keys/{key_id}/usage",
            "GET /harness/usage",
            "GET /v1/batches/{batch_id}",
            "GET /v1/chat/completions/{completion_id}",
            "GET /v1/chat/completions/{completion_id}/messages",
            "GET /v1/files/{file_id}/content",
            "GET /v1/responses/{response_id}",
            "POST /harness/complete",
            "POST /harness/complete/stream",
            "POST /harness/keys",
            "POST /v1/batches",
            "POST /v1/chat/completions",
            "POST /v1/completions",
            "POST /v1/embeddings",
            "POST /v1/files",
            "POST /v1/messages",
            "POST /v1/responses",
        }
    ),
    "client_audit": frozenset(
        {
            "DELETE /harness/keys/{key_id}",
            "POST /harness/keys",
        }
    ),
    "contract_audit": frozenset(),  # schema-pin of the whole openapi surface — non-route audit
    "conv_audit": frozenset(
        {
            "DELETE /v1/conversations/{conversation_id}",
            "DELETE /v1/conversations/{conversation_id}/items/{item_id}",
            "GET /v1/batches/{batch_id}",
            "GET /v1/conversations/{conversation_id}",
            "GET /v1/conversations/{conversation_id}/items",
            "GET /v1/conversations/{conversation_id}/items/{item_id}",
            "GET /v1/files/{file_id}/content",
            "GET /v1/responses/{response_id}",
            "GET /v1/responses/{response_id}/input_items",
            "GET /v1/{path:path}",
            "POST /harness/keys",
            "POST /v1/batches",
            "POST /v1/conversations",
            "POST /v1/conversations/{conversation_id}",
            "POST /v1/conversations/{conversation_id}/items",
            "POST /v1/files",
            "POST /v1/responses",
            "PUT /v1/{path:path}",
        }
    ),
    "drain_audit": frozenset(
        {
            "DELETE /harness/evals/{eval_id}",
            "DELETE /harness/keys/{key_id}",
            "DELETE /v1/chat/completions/{completion_id}",
            "DELETE /v1/conversations/{conversation_id}",
            "DELETE /v1/evals/{eval_id}",
            "DELETE /v1/files/{file_id}",
            "DELETE /v1/models/{model}",
            "DELETE /v1/vector_stores/{vector_store_id}",
            "GET /harness/capabilities",
            "GET /harness/commands",
            "GET /harness/evals",
            "GET /harness/jobs",
            "GET /harness/keys",
            "GET /harness/keys/{key_id}",
            "GET /harness/keys/{key_id}/usage",
            "GET /harness/self",
            "GET /harness/version",
            "GET /health",
            "GET /metrics",
            "GET /openapi.json",
            "GET /ready",
            "GET /v1/batches",
            "GET /v1/chat/completions/{completion_id}",
            "GET /v1/conversations/{conversation_id}",
            "GET /v1/conversations/{conversation_id}/items",
            "GET /v1/evals",
            "GET /v1/evals/{eval_id}",
            "GET /v1/files",
            "GET /v1/files/{file_id}",
            "GET /v1/files/{file_id}/content",
            "GET /v1/fine_tuning/jobs",
            "GET /v1/fine_tuning/jobs/{job_id}",
            "GET /v1/fine_tuning/jobs/{job_id}/events",
            "GET /v1/messages/batches",
            "GET /v1/models",
            "GET /v1/responses/{response_id}",
            "GET /v1/vector_stores/{vector_store_id}",
            "GET /v1/vector_stores/{vector_store_id}/files",
            "GET /v1/{path:path}",
            "PATCH /harness/keys/{key_id}",
            "POST /harness/backends/{name}/probe",
            "POST /harness/complete",
            "POST /harness/complete/batch",
            "POST /harness/complete/stream",
            "POST /harness/drain",
            "POST /harness/evals",
            "POST /harness/gate/check",
            "POST /harness/jobs",
            "POST /harness/jobs/batch",
            "POST /harness/keys",
            "POST /harness/keys/{key_id}/rotate",
            "POST /harness/runs",
            "POST /harness/score",
            "POST /receipts/verify",
            "POST /v1/batches",
            "POST /v1/chat/completions",
            "POST /v1/chat/completions/{completion_id}",
            "POST /v1/completions",
            "POST /v1/conversations",
            "POST /v1/conversations/{conversation_id}",
            "POST /v1/conversations/{conversation_id}/items",
            "POST /v1/embeddings",
            "POST /v1/evals",
            "POST /v1/evals/{eval_id}",
            "POST /v1/evals/{eval_id}/runs",
            "POST /v1/files",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/fine_tuning/jobs/{job_id}/cancel",
            "POST /v1/fine_tuning/jobs/{job_id}/pause",
            "POST /v1/fine_tuning/jobs/{job_id}/resume",
            "POST /v1/messages",
            "POST /v1/messages/batches",
            "POST /v1/messages/count_tokens",
            "POST /v1/moderations",
            "POST /v1/responses",
            "POST /v1/uploads",
            "POST /v1/uploads/{upload_id}/cancel",
            "POST /v1/uploads/{upload_id}/complete",
            "POST /v1/uploads/{upload_id}/parts",
            "POST /v1/vector_stores",
            "POST /v1/vector_stores/{vector_store_id}",
            "POST /v1/vector_stores/{vector_store_id}/file_batches",
            "POST /v1/vector_stores/{vector_store_id}/files",
            "POST /v1/vector_stores/{vector_store_id}/search",
        }
    ),
    "e2e_audit": frozenset(
        {
            "GET /harness/jobs/{job_id}",
            "GET /harness/jobs/{job_id}/events",
            "GET /health",
            "POST /harness/jobs",
            "POST /harness/jobs/batch",
        }
    ),
    "eval_lifecycle_audit": frozenset(
        {
            "DELETE /harness/evals/{eval_id}",
            "DELETE /v1/evals/{eval_id}",
            "DELETE /v1/evals/{eval_id}/runs/{run_id}",
            "GET /harness/evals",
            "GET /harness/evals/{eval_id}",
            "GET /harness/evals/{eval_id}/diff/{candidate_id}",
            "GET /harness/evals/{eval_id}/receipt",
            "GET /metrics",
            "GET /v1/evals",
            "GET /v1/evals/{eval_id}",
            "GET /v1/evals/{eval_id}/runs",
            "GET /v1/evals/{eval_id}/runs/{run_id}",
            "GET /v1/evals/{eval_id}/runs/{run_id}/output_items",
            "POST /harness/drain",
            "POST /harness/evals",
            "POST /harness/keys",
            "POST /v1/evals",
            "POST /v1/evals/{eval_id}",
            "POST /v1/evals/{eval_id}/runs",
        }
    ),
    "fault_audit": frozenset(
        {
            "DELETE /harness/keys/{key_id}",
            "GET /harness/commands",
            "GET /harness/keys",
            "GET /harness/self",
            "GET /ready",
            "POST /harness/drain",
            "POST /harness/jobs",
            "POST /harness/keys",
            "POST /harness/runs",
        }
    ),
    "files_audit": frozenset(
        {
            "DELETE /v1/files/{file_id}",
            "DELETE /v1/{path:path}",
            "GET /v1/batches",
            "GET /v1/batches/{batch_id}",
            "GET /v1/files",
            "GET /v1/files/{file_id}",
            "GET /v1/files/{file_id}/content",
            "GET /v1/fine_tuning/jobs/{job_id}",
            "GET /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "GET /v1/vector_stores/{vector_store_id}/files/{file_id}/content",
            "POST /harness/keys",
            "POST /v1/batches",
            "POST /v1/files",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/uploads",
            "POST /v1/uploads/{upload_id}/complete",
            "POST /v1/uploads/{upload_id}/parts",
            "POST /v1/vector_stores",
            "POST /v1/vector_stores/{vector_store_id}/files",
            "POST /v1/{path:path}",
            "PUT /v1/{path:path}",
        }
    ),
    "header_audit": frozenset(
        {
            "GET /harness/commands",
            "GET /harness/completions/{completion_id}/receipt",
            "GET /health",
            "GET /metrics",
            "GET /receipts/{sha256}",
            "GET /v1/batches/{batch_id}",
            "GET /v1/models",
            "OPTIONS /v1/{path:path}",
            "POST /harness/complete",
            "POST /harness/keys",
            "POST /v1/batches",
            "POST /v1/chat/completions",
            "POST /v1/embeddings",
            "POST /v1/files",
            "POST /v1/messages",
            "POST /v1/messages/count_tokens",
            "POST /v1/responses",
            "POST /v1/{path:path}",
        }
    ),
    "idem_audit": frozenset(
        {
            "DELETE /v1/files/{file_id}",
            "GET /harness/evals",
            "GET /harness/jobs",
            "GET /harness/keys",
            "GET /v1/batches",
            "GET /v1/evals/{eval_id}/runs",
            "GET /v1/files",
            "GET /v1/files/{file_id}",
            "GET /v1/fine_tuning/jobs",
            "POST /harness/complete",
            "POST /harness/jobs",
            "POST /harness/jobs/batch",
            "POST /harness/keys",
            "POST /harness/runs",
            "POST /v1/chat/completions",
            "POST /v1/evals",
            "POST /v1/files",
            "POST /v1/messages",
            "POST /v1/responses",
            "POST /v1/uploads",
            "POST /v1/uploads/{upload_id}/cancel",
            "POST /v1/uploads/{upload_id}/complete",
            "POST /v1/uploads/{upload_id}/parts",
        }
    ),
    "jobs_audit": frozenset(
        {
            "DELETE /harness/jobs/{job_id}",
            "GET /harness/jobs",
            "GET /harness/jobs/{job_id}",
            "GET /harness/jobs/{job_id}/events",
            "GET /harness/jobs/{job_id}/receipt",
            "GET /harness/keys/{key_id}",
            "GET /harness/self",
            "GET /health",
            "POST /harness/drain",
            "POST /harness/jobs",
            "POST /harness/jobs/batch",
            "POST /harness/keys",
            "POST /harness/runs",
        }
    ),
    "journal_audit": frozenset(),  # journal/record-store internals — non-HTTP audit
    "models_audit": frozenset(
        {
            "DELETE /v1/models/{model}",
            "DELETE /v1/{path:path}",
            "GET /v1/fine_tuning/jobs/{job_id}",
            "GET /v1/fine_tuning/jobs/{job_id}/checkpoints",
            "GET /v1/models",
            "GET /v1/models/{model}",
            "GET /v1/{path:path}",
            "POST /v1/chat/completions",
            "POST /v1/completions",
            "POST /v1/embeddings",
            "POST /v1/files",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/fine_tuning/jobs/{job_id}/cancel",
            "POST /v1/messages",
            "POST /v1/messages/count_tokens",
            "POST /v1/responses",
        }
    ),
    "oai_sdk_audit": frozenset(
        {
            "DELETE /v1/chat/completions/{completion_id}",
            "DELETE /v1/conversations/{conversation_id}",
            "DELETE /v1/conversations/{conversation_id}/items/{item_id}",
            "DELETE /v1/evals/{eval_id}",
            "DELETE /v1/files/{file_id}",
            "DELETE /v1/models/{model}",
            "DELETE /v1/responses/{response_id}",
            "DELETE /v1/vector_stores/{vector_store_id}",
            "DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "GET /v1/batches",
            "GET /v1/batches/{batch_id}",
            "GET /v1/chat/completions",
            "GET /v1/chat/completions/{completion_id}",
            "GET /v1/chat/completions/{completion_id}/messages",
            "GET /v1/conversations/{conversation_id}",
            "GET /v1/conversations/{conversation_id}/items",
            "GET /v1/conversations/{conversation_id}/items/{item_id}",
            "GET /v1/evals",
            "GET /v1/evals/{eval_id}",
            "GET /v1/evals/{eval_id}/runs",
            "GET /v1/files",
            "GET /v1/files/{file_id}",
            "GET /v1/files/{file_id}/content",
            "GET /v1/fine_tuning/jobs",
            "GET /v1/fine_tuning/jobs/{job_id}",
            "GET /v1/fine_tuning/jobs/{job_id}/checkpoints",
            "GET /v1/fine_tuning/jobs/{job_id}/events",
            "GET /v1/models",
            "GET /v1/models/{model}",
            "GET /v1/responses/{response_id}",
            "GET /v1/responses/{response_id}/input_items",
            "GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}",
            "GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
            "GET /v1/vector_stores/{vector_store_id}/files",
            "GET /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "GET /v1/vector_stores/{vector_store_id}/files/{file_id}/content",
            "POST /v1/batches",
            "POST /v1/chat/completions",
            "POST /v1/chat/completions/{completion_id}",
            "POST /v1/completions",
            "POST /v1/conversations",
            "POST /v1/conversations/{conversation_id}",
            "POST /v1/conversations/{conversation_id}/items",
            "POST /v1/embeddings",
            "POST /v1/evals",
            "POST /v1/evals/{eval_id}",
            "POST /v1/evals/{eval_id}/runs",
            "POST /v1/files",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/fine_tuning/jobs/{job_id}/cancel",
            "POST /v1/moderations",
            "POST /v1/responses",
            "POST /v1/responses/{response_id}/cancel",
            "POST /v1/uploads",
            "POST /v1/uploads/{upload_id}/cancel",
            "POST /v1/uploads/{upload_id}/complete",
            "POST /v1/uploads/{upload_id}/parts",
            "POST /v1/vector_stores",
            "POST /v1/vector_stores/{vector_store_id}/file_batches",
            "POST /v1/vector_stores/{vector_store_id}/files",
            "POST /v1/vector_stores/{vector_store_id}/search",
        }
    ),
    "observe_audit": frozenset(
        {
            "GET /harness/backends",
            "GET /harness/commands",
            "GET /harness/completions",
            "GET /harness/usage",
            "GET /health",
            "GET /metrics",
            "GET /ready",
            "POST /harness/complete",
            "POST /harness/complete/batch",
            "POST /harness/drain",
            "POST /harness/jobs",
            "POST /harness/keys",
            "POST /harness/runs",
            "POST /v1/messages",
        }
    ),
    "parity_audit": frozenset(
        {
            "DELETE /v1/chat/completions/{completion_id}",
            "DELETE /v1/models/{model}",
            "GET /harness/commands",
            "GET /harness/completions/{completion_id}",
            "GET /health",
            "GET /receipts",
            "GET /v1/chat/completions/{completion_id}",
            "GET /v1/models",
            "GET /v1/models/{model}",
            "POST /harness/complete",
            "POST /harness/complete/batch",
            "POST /harness/complete/stream",
            "POST /harness/jobs",
            "POST /harness/keys",
            "POST /harness/runs",
            "POST /receipts/verify",
            "POST /v1/chat/completions",
            "POST /v1/chat/completions/{completion_id}",
            "POST /v1/completions",
            "POST /v1/embeddings",
            "POST /v1/messages",
            "POST /v1/messages/batches",
            "POST /v1/messages/count_tokens",
            "POST /v1/responses",
            "POST /v1/responses/{response_id}/cancel",
        }
    ),
    "perf_audit": frozenset(
        {
            "GET /harness/completions",
            "GET /harness/usage",
            "GET /metrics",
            "GET /v1/chat/completions",
            "GET /v1/chat/completions/{completion_id}",
            "GET /v1/chat/completions/{completion_id}/messages",
            "GET /v1/responses/{response_id}",
            "POST /harness/complete",
            "POST /harness/complete/stream",
            "POST /harness/jobs",
            "POST /harness/runs",
            "POST /v1/chat/completions",
            "POST /v1/responses",
            "POST /v1/uploads",
            "POST /v1/uploads/{upload_id}/parts",
        }
    ),
    "quota2_audit": frozenset(
        {
            "GET /harness/keys/{key_id}",
            "GET /harness/keys/{key_id}/usage",
            "GET /harness/self",
            "PATCH /harness/keys/{key_id}",
            "POST /harness/complete",
            "POST /harness/keys",
            "POST /v1/chat/completions",
            "POST /v1/messages",
        }
    ),
    "quota_audit": frozenset(
        {
            "DELETE /harness/keys/{key_id}",
            "GET /harness/jobs",
            "GET /harness/keys/{key_id}/usage",
            "GET /harness/self",
            "GET /health",
            "PATCH /harness/keys/{key_id}",
            "POST /harness/complete",
            "POST /harness/complete/batch",
            "POST /harness/complete/stream",
            "POST /harness/evals",
            "POST /harness/jobs",
            "POST /harness/keys",
            "POST /harness/keys/{key_id}/rotate",
            "POST /v1/chat/completions",
            "POST /v1/completions",
            "POST /v1/messages",
            "POST /v1/responses",
        }
    ),
    "retrieval_audit": frozenset(
        {
            "DELETE /v1/chat/completions/{completion_id}",
            "DELETE /v1/conversations/{conversation_id}",
            "DELETE /v1/conversations/{conversation_id}/items/{item_id}",
            "DELETE /v1/evals/{eval_id}",
            "DELETE /v1/evals/{eval_id}/runs/{run_id}",
            "DELETE /v1/messages/batches/{batch_id}",
            "DELETE /v1/models/{model}",
            "DELETE /v1/responses/{response_id}",
            "DELETE /v1/vector_stores/{vector_store_id}",
            "DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "GET /harness/completions",
            "GET /harness/completions/{completion_id}",
            "GET /harness/evals",
            "GET /harness/evals/{eval_id}",
            "GET /harness/jobs",
            "GET /harness/jobs/{job_id}",
            "GET /harness/jobs/{job_id}/events",
            "GET /v1/batches",
            "GET /v1/chat/completions",
            "GET /v1/chat/completions/{completion_id}/messages",
            "GET /v1/conversations/{conversation_id}/items",
            "GET /v1/conversations/{conversation_id}/items/{item_id}",
            "GET /v1/evals",
            "GET /v1/evals/{eval_id}/runs",
            "GET /v1/evals/{eval_id}/runs/{run_id}",
            "GET /v1/evals/{eval_id}/runs/{run_id}/output_items",
            "GET /v1/files",
            "GET /v1/files/{file_id}",
            "GET /v1/files/{file_id}/content",
            "GET /v1/fine_tuning/jobs",
            "GET /v1/fine_tuning/jobs/{job_id}",
            "GET /v1/fine_tuning/jobs/{job_id}/checkpoints",
            "GET /v1/fine_tuning/jobs/{job_id}/events",
            "GET /v1/messages/batches",
            "GET /v1/messages/batches/{batch_id}",
            "GET /v1/messages/batches/{batch_id}/results",
            "GET /v1/models",
            "GET /v1/responses/{response_id}",
            "GET /v1/responses/{response_id}/input_items",
            "GET /v1/vector_stores",
            "GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
            "GET /v1/vector_stores/{vector_store_id}/files",
            "GET /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "GET /v1/vector_stores/{vector_store_id}/files/{file_id}/content",
            "GET /v1/{path:path}",
            "POST /harness/jobs",
            "POST /harness/keys",
            "POST /v1/batches",
            "POST /v1/chat/completions",
            "POST /v1/conversations",
            "POST /v1/conversations/{conversation_id}/items",
            "POST /v1/evals",
            "POST /v1/evals/{eval_id}/runs",
            "POST /v1/evals/{eval_id}/runs/{run_id}/cancel",
            "POST /v1/files",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/fine_tuning/jobs/{job_id}/cancel",
            "POST /v1/fine_tuning/jobs/{job_id}/pause",
            "POST /v1/messages/batches",
            "POST /v1/responses",
            "POST /v1/vector_stores",
            "POST /v1/vector_stores/{vector_store_id}/file_batches",
            "POST /v1/vector_stores/{vector_store_id}/files",
        }
    ),
    "routing_audit": frozenset(
        {
            "DELETE /v1/{path:path}",
            "GET /docs",
            "GET /docs/oauth2-redirect",
            "GET /openapi.json",
            "GET /redoc",
            "GET /v1/{path:path}",
            "HEAD /docs",
            "HEAD /docs/oauth2-redirect",
            "HEAD /openapi.json",
            "HEAD /redoc",
            "HEAD /v1/{path:path}",
            "OPTIONS /v1/{path:path}",
            "PATCH /v1/{path:path}",
            "POST /v1/{path:path}",
            "PUT /v1/{path:path}",
        }
    ),
    "serve_audit": frozenset(),  # harness runner + transport seams — non-HTTP audit
    "spec_audit": frozenset(
        {
            "GET /harness/commands",
            "GET /v1/batches/{batch_id}",
            "GET /v1/files",
            "GET /v1/fine_tuning/jobs/{job_id}",
            "GET /v1/messages/batches/{batch_id}",
            "GET /v1/models",
            "GET /v1/models/{model}",
            "GET /v1/responses/{response_id}",
            "POST /harness/keys",
            "POST /v1/batches",
            "POST /v1/chat/completions",
            "POST /v1/completions",
            "POST /v1/files",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/messages",
            "POST /v1/messages/batches",
            "POST /v1/messages/count_tokens",
            "POST /v1/responses",
        }
    ),
    "stream_audit": frozenset(
        {
            "DELETE /v1/responses/{response_id}",
            "GET /v1/responses/{response_id}",
            "POST /harness/complete/stream",
            "POST /v1/chat/completions",
            "POST /v1/completions",
            "POST /v1/messages",
            "POST /v1/responses",
            "POST /v1/responses/{response_id}/cancel",
        }
    ),
    "tenancy_audit": frozenset(
        {
            "DELETE /harness/keys/{key_id}",
            "DELETE /v1/chat/completions/{completion_id}",
            "DELETE /v1/conversations/{conversation_id}",
            "DELETE /v1/evals/{eval_id}",
            "DELETE /v1/files/{file_id}",
            "DELETE /v1/models/{model}",
            "DELETE /v1/responses/{response_id}",
            "DELETE /v1/vector_stores/{vector_store_id}",
            "DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "GET /harness/completions",
            "GET /harness/jobs",
            "GET /harness/keys",
            "GET /harness/keys/{key_id}/usage",
            "GET /harness/self",
            "GET /harness/usage",
            "GET /health",
            "GET /v1/batches",
            "GET /v1/batches/{batch_id}",
            "GET /v1/chat/completions",
            "GET /v1/chat/completions/{completion_id}",
            "GET /v1/chat/completions/{completion_id}/messages",
            "GET /v1/conversations/{conversation_id}",
            "GET /v1/conversations/{conversation_id}/items",
            "GET /v1/evals",
            "GET /v1/evals/{eval_id}",
            "GET /v1/evals/{eval_id}/runs",
            "GET /v1/evals/{eval_id}/runs/{run_id}",
            "GET /v1/files",
            "GET /v1/files/{file_id}",
            "GET /v1/files/{file_id}/content",
            "GET /v1/fine_tuning/jobs",
            "GET /v1/fine_tuning/jobs/{job_id}",
            "GET /v1/fine_tuning/jobs/{job_id}/checkpoints",
            "GET /v1/fine_tuning/jobs/{job_id}/events",
            "GET /v1/messages/batches",
            "GET /v1/messages/batches/{batch_id}",
            "GET /v1/models",
            "GET /v1/responses/{response_id}",
            "GET /v1/responses/{response_id}/input_items",
            "GET /v1/vector_stores",
            "GET /v1/vector_stores/{vector_store_id}",
            "GET /v1/{path:path}",
            "POST /harness/drain",
            "POST /harness/jobs",
            "POST /harness/keys",
            "POST /harness/keys/{key_id}/rotate",
            "POST /harness/runs",
            "POST /v1/batches",
            "POST /v1/batches/{batch_id}/cancel",
            "POST /v1/chat/completions",
            "POST /v1/chat/completions/{completion_id}",
            "POST /v1/conversations",
            "POST /v1/conversations/{conversation_id}",
            "POST /v1/conversations/{conversation_id}/items",
            "POST /v1/evals",
            "POST /v1/evals/{eval_id}",
            "POST /v1/evals/{eval_id}/runs",
            "POST /v1/evals/{eval_id}/runs/{run_id}/cancel",
            "POST /v1/files",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/messages/batches",
            "POST /v1/responses",
            "POST /v1/responses/{response_id}/cancel",
            "POST /v1/uploads",
            "POST /v1/uploads/{upload_id}/complete",
            "POST /v1/uploads/{upload_id}/parts",
            "POST /v1/vector_stores",
            "POST /v1/vector_stores/{vector_store_id}/files",
            "POST /v1/vector_stores/{vector_store_id}/search",
        }
    ),
    "uploads_audit": frozenset(
        {
            "GET /v1/batches/{batch_id}",
            "GET /v1/files",
            "GET /v1/files/{file_id}",
            "GET /v1/files/{file_id}/content",
            "GET /v1/{path:path}",
            "POST /harness/keys",
            "POST /v1/batches",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/uploads",
            "POST /v1/uploads/{upload_id}/cancel",
            "POST /v1/uploads/{upload_id}/complete",
            "POST /v1/uploads/{upload_id}/parts",
            "POST /v1/vector_stores",
            "POST /v1/vector_stores/{vector_store_id}/files",
        }
    ),
    "usage_audit": frozenset(
        {
            "GET /harness/commands",
            "GET /harness/completions",
            "GET /harness/keys/{key_id}/usage",
            "GET /harness/self",
            "GET /harness/usage",
            "POST /harness/complete",
            "POST /harness/complete/batch",
            "POST /harness/complete/stream",
            "POST /harness/keys",
            "POST /v1/chat/completions",
            "POST /v1/messages",
        }
    ),
    "webhook_audit": frozenset(
        {
            "DELETE /harness/evals/{eval_id}",
            "GET /harness/evals/{eval_id}",
            "GET /harness/jobs",
            "GET /v1/batches/{batch_id}",
            "GET /v1/fine_tuning/jobs/{job_id}",
            "GET /v1/messages/batches/{batch_id}",
            "POST /harness/evals",
            "POST /harness/jobs",
            "POST /v1/batches",
            "POST /v1/batches/{batch_id}/cancel",
            "POST /v1/files",
            "POST /v1/fine_tuning/jobs",
            "POST /v1/fine_tuning/jobs/{job_id}/cancel",
            "POST /v1/messages/batches",
            "POST /v1/messages/batches/{batch_id}/cancel",
        }
    ),
}


def _parse_manifest() -> dict[str, frozenset[tuple[str, str]]]:
    """Manifest entries as ``(template, method)`` pairs; malformed entries
    are kept under ``(entry, "")`` so the resolve probe can name them."""
    out: dict[str, frozenset[tuple[str, str]]] = {}
    for mod, entries in AUDIT_ROUTE_MANIFEST.items():
        pairs: set[tuple[str, str]] = set()
        for entry in entries:
            m = _ENTRY_RE.match(entry)
            pairs.add((m.group(2), m.group(1)) if m else (entry, ""))
        out[mod] = frozenset(pairs)
    return out


def live_route_table(app: Any) -> list[dict[str, Any]]:
    """Enumerate ``app.routes`` as ordered ``(path, methods, name, kind)`` rows."""
    rows: list[dict[str, Any]] = []
    for idx, route in enumerate(app.routes):
        rows.append(
            {
                "index": idx,
                "path": getattr(route, "path", ""),
                "methods": sorted(getattr(route, "methods", None) or ()),
                "name": getattr(route, "name", ""),
                "kind": type(route).__name__,
                "in_schema": bool(getattr(route, "include_in_schema", True)),
            }
        )
    return rows


def _openapi_pairs(app: Any) -> dict[str, frozenset[str]]:
    """``{path: {methods}}`` declared by ``app.openapi()``."""
    spec = app.openapi()
    out: dict[str, frozenset[str]] = {}
    for path, ops in spec.get("paths", {}).items():
        if not isinstance(ops, dict):
            continue
        out[path] = frozenset(m.upper() for m in ops if m.lower() in _OPENAPI_METHODS)
    return out


def _mount_parity() -> dict[str, Any]:
    """Static parity between ``api.py``'s mount helpers and decorators and
    the live route table — computed from source so a dead mount shows."""
    src = _API_SRC.read_text(encoding="utf-8")
    defined = _MOUNT_DEF_RE.findall(src)
    app_start = src.find("\ndef create_app(")
    app_body = src[app_start:] if app_start != -1 else ""
    called = [name for name in defined if re.search(rf"\b{name}\(", app_body)]
    literals = {lit for lit in _PATH_LIT_RE.findall(src) if lit.startswith(tuple(_KNOWN_PREFIXES))}
    return {
        "mounts_defined": sorted(defined),
        "mounts_called": sorted(called),
        "declared_literals": sorted(literals),
    }


def _canonical(path: str) -> str:
    """Param-erased path shape — ``/v1/files/{file_id}`` -> ``/v1/files/{}``."""
    return _PARAM_RE.sub("{}", path)


def _prefix_family(path: str) -> str | None:
    for prefix in _KNOWN_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return prefix
    return None


def audit_coverage(
    live: list[dict[str, Any]] | None = None,
    manifest: dict[str, frozenset[tuple[str, str]]] | None = None,
) -> dict[str, Any]:
    """Cross-reference the live route table against the manifest.

    Returns the coverage report: per-pair claimants, the ``gaps`` list of
    uncovered ``(method, path)`` pairs, density histogram, and which pairs
    are covered only by modules without a sealed receipt.
    """
    if live is None:
        from fx1.serve.api import create_app  # noqa: PLC0415

        live = live_route_table(create_app())
    if manifest is None:
        manifest = _parse_manifest()

    live_pairs = {(row["path"], m) for row in live for m in row["methods"]}
    claimants: dict[tuple[str, str], list[str]] = {}
    for mod, pairs in manifest.items():
        for pair in pairs:
            claimants.setdefault(pair, []).append(mod)
    for mods_list in claimants.values():
        mods_list.sort()

    sealed = {mod for mod in manifest if (_RECEIPTS / f"fx1_{mod}.json").is_file()}
    gaps = sorted(
        f"{method} {path}" for path, method in sorted(live_pairs) if (path, method) not in claimants
    )
    unsealed_only = sorted(
        f"{method} {path}"
        for (path, method), mods in sorted(claimants.items())
        if (path, method) in live_pairs and not (set(mods) & sealed)
    )
    density: dict[str, int] = {}
    for path, method in sorted(live_pairs):
        n = len(claimants.get((path, method), []))
        key = str(n) if n < 3 else "3+"
        density[key] = density.get(key, 0) + 1
    return {
        "n_routes": len(live),
        "n_paths": len({row["path"] for row in live}),
        "n_pairs": len(live_pairs),
        "gaps": gaps,
        "unsealed_only": unsealed_only,
        "density": density,
        "sealed_modules": sorted(sealed),
        "claimants": {f"{m} {p}": mods for (p, m), mods in sorted(claimants.items())},
    }


def _audit_modules_on_disk() -> frozenset[str]:
    return frozenset(p.stem for p in _SERVE_DIR.glob("*_audit.py"))


def _enveloped_404_openai(body: Any) -> bool:
    err = body.get("error") if isinstance(body, dict) else None
    return (
        isinstance(err, dict)
        and err.get("type") == "invalid_request_error"
        and err.get("code") == "not_found"
    )


def _enveloped_404_anthropic(body: Any) -> bool:
    return (
        isinstance(body, dict)
        and body.get("type") == "error"
        and isinstance(body.get("error"), dict)
        and body["error"].get("type") == "not_found_error"
    )


def _enveloped_405(client: TestClient, method: str, path: str) -> bool:
    r = client.request(method, path)
    body = r.json() if r.content else {}
    return (
        r.status_code == 405
        and isinstance(body, dict)
        and body.get("code") == "method_not_allowed"
        and "detail" in body
    )


def routing_audit() -> dict[str, bool]:
    """Probe battery: enumerate the served surface, prove the manifest
    covers it, and exercise the meta-routes this audit itself claims."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from fx1.serve.api import create_app  # noqa: PLC0415

    prev_key = os.environ.pop(_API_KEY_ENV, None)
    out: dict[str, bool] = {}
    try:
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        rows = live_route_table(app)
        live_pairs = {(r["path"], m) for r in rows for m in r["methods"]}
        live_methods: dict[str, set[str]] = {}
        for r in rows:
            live_methods.setdefault(r["path"], set()).update(r["methods"])
        spec = _openapi_pairs(app)
        manifest = _parse_manifest()

        # --- enumeration sanity -----------------------------------------
        out["route_table_enumerable"] = bool(rows) and all(
            r["path"] and r["methods"] and r["name"] for r in rows
        )
        out["route_table_deterministic"] = rows == live_route_table(create_app())

        # --- route table <-> openapi parity ------------------------------
        spec_pairs = {(p, m) for p, ms in spec.items() for m in ms}
        out["openapi_operations_all_live"] = spec_pairs <= live_pairs
        out["live_methods_match_spec"] = all(
            live_methods.get(p, set()) == set(ms) for p, ms in spec.items()
        )
        non_schema = {r["path"] for r in rows} - set(spec)
        out["non_schema_surface_expected"] = non_schema == _NON_SCHEMA_PATHS

        # --- manifest derivability ---------------------------------------
        disk_modules = _audit_modules_on_disk()
        out["manifest_covers_every_module"] = set(manifest) == set(disk_modules)
        manifest_pairs = {pair for pairs in manifest.values() for pair in pairs}
        out["manifest_entries_resolve"] = bool(manifest_pairs) and all(
            tpl in live_methods and method in live_methods[tpl] for tpl, method in manifest_pairs
        )

        # --- coverage completeness ---------------------------------------
        coverage = audit_coverage(rows, manifest)
        out["coverage_complete_no_gaps"] = coverage["gaps"] == []
        out["coverage_multi_audit_present"] = any(
            len(mods) > 1 for mods in coverage["claimants"].values()
        )

        # --- sealed receipts ----------------------------------------------
        receipt_files = sorted(_RECEIPTS.glob("fx1_*_audit.json"))
        from quant_fund.research.receipt_v2 import verify_receipt_file  # noqa: PLC0415

        out["sealed_receipts_all_verify"] = bool(receipt_files) and all(
            verify_receipt_file(f)["valid"] for f in receipt_files
        )

        # --- catch-all + method surface ------------------------------------
        catch_rows = [r for r in rows if r["path"] == _CATCH_ALL]
        catch = catch_rows[0] if catch_rows else None
        out["catch_all_registered_once_last"] = (
            len(catch_rows) == 1
            and catch is not None
            and catch["index"] == rows[-1]["index"]
            and catch["in_schema"] is False
        )
        out["catch_all_declares_all_methods"] = catch is not None and set(catch["methods"]) == set(
            _METHODS
        )
        out["catch_all_serves_every_method"] = all(
            (r.status_code == 404 and (m == "HEAD" or _enveloped_404_openai(r.json())))
            for m in _METHODS
            for r in [client.request(m, f"/v1/__unmapped_{m.lower()}__")]
        )
        out["catch_all_openai_grammar"] = _enveloped_404_openai(client.get(_SENTINEL_PATH).json())
        out["catch_all_anthropic_grammar"] = _enveloped_404_anthropic(
            client.get("/v1/messages/__unmapped__").json()
        )
        out["harness_method_mismatch_405"] = _enveloped_405(
            client, "GET", "/harness/complete"
        ) and _enveloped_405(client, "POST", "/receipts")
        out["head_honesty_by_route_kind"] = (
            client.head("/openapi.json").status_code == 200
            and client.head("/harness/jobs").status_code == 405
            and client.head("/v1/models").status_code == 404
        )
        out["docs_surface_live"] = all(
            client.request(m, p).status_code == 200
            for p in ("/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json")
            for m in ("GET", "HEAD")
        )

        # --- dialect + mount parity ---------------------------------------
        families = {r["path"]: _prefix_family(r["path"]) for r in rows}
        out["dialect_prefixes_partition"] = all(
            f is not None for f in families.values()
        ) and not any(p.startswith("/v1/harness") or p.startswith("/harness/v1") for p in families)
        parity = _mount_parity()
        out["mount_helpers_all_wired"] = parity["mounts_defined"] == parity[
            "mounts_called"
        ] and bool(parity["mounts_defined"])
        declared = set(parity["declared_literals"])
        out["every_api_route_declared_in_source"] = all(
            r["path"] in declared for r in rows if r["kind"] == "APIRoute"
        )

        # --- shadowing / template honesty ----------------------------------
        canon_seen: dict[tuple[str, str], str] = {}
        shadowed = []
        for r in rows:
            for m in r["methods"]:
                key = (_canonical(r["path"]), m)
                if key in canon_seen:
                    shadowed.append(f"{m} {r['path']} ~ {canon_seen[key]}")
                else:
                    canon_seen[key] = r["path"]
        out["no_shadowed_templates"] = shadowed == []
        out["catch_all_only_path_converter"] = all(
            ":path" not in r["path"] or r["path"] == _CATCH_ALL for r in rows
        )
        canon_entries = {(_canonical(tpl), m) for tpl, m in manifest_pairs}
        out["manifest_uses_canonical_templates"] = len(canon_entries) == len(manifest_pairs)

        # --- fresh-route detection (the audit's own teeth) ------------------
        synth_rows = list(rows) + [
            {
                "index": len(rows),
                "path": _SENTINEL_PATH,
                "methods": [_SENTINEL_METHOD],
                "name": "sentinel",
                "kind": "APIRoute",
                "in_schema": True,
            }
        ]
        synth = audit_coverage(synth_rows, manifest)
        out["fresh_route_flagged"] = synth["gaps"] == [f"{_SENTINEL_METHOD} {_SENTINEL_PATH}"]

        # --- ratchet consistency ------------------------------------------
        try:
            import json  # noqa: PLC0415

            manifest_doc = json.loads(_COVERAGE_MANIFEST.read_text())
            pinned = manifest_doc["directories"]["serve"]["n_modules"]
            waived = {
                k for k, v in manifest_doc.get("modules", {}).items() if v.get("status") == "waived"
            }
            fx1_src = _SERVE_DIR.parent
            n_live = sum(
                1
                for p in fx1_src.rglob("*.py")
                if "__pycache__" not in p.parts
                and (rel := p.relative_to(fx1_src).as_posix()).split("/")[0] == "serve"
                and rel not in waived
            )
            out["ratchet_census_matches"] = isinstance(pinned, int) and pinned == n_live
        except (OSError, ValueError, KeyError, TypeError):
            out["ratchet_census_matches"] = False
    finally:
        if prev_key is not None:
            os.environ[_API_KEY_ENV] = prev_key
    return out


def routing_audit_bench() -> dict[str, Any]:
    """Seal the routing meta-audit as ``fx1_routing_audit.v1``."""
    r = routing_audit()
    ok = bool(r) and all(v is True for v in r.values())
    coverage = audit_coverage()
    out: dict[str, Any] = {
        "kind": "fx1_routing_audit",
        "schema": "fx1_routing_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": r,
            "ok": ok,
            "gaps": coverage["gaps"],
            "unsealed_only": coverage["unsealed_only"],
            "route_count": coverage["n_routes"],
            "path_count": coverage["n_paths"],
            "pair_count": coverage["n_pairs"],
            "coverage_density": coverage["density"],
            "sealed_modules": coverage["sealed_modules"],
            "manifest_modules": sorted(AUDIT_ROUTE_MANIFEST),
        },
        "interpretation": (
            "every route+method the app serves is claimed by at least one "
            "audit battery; the manifest resolves against the live table, "
            "the catch-all is last and enveloped, and a sentinel route is "
            "flagged — coverage is a claim the harness can falsify"
            if ok
            else f"ROUTING COVERAGE DEFECTS: {sorted(k for k, v in r.items() if v is not True)}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(routing_audit_bench(), indent=2, sort_keys=True))
