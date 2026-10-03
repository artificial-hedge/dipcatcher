/**
 * Typed client for the dipcatcher fx-1 harness API.
 *
 * Zero runtime dependencies: paths and response types are derived from the
 * generated `schema.d.ts` (openapi-typescript) so this client cannot drift
 * from the committed `openapi.json`. Works in any environment with `fetch`
 * (Node 18+, browsers, Deno, Bun).
 *
 * The harness binds loopback-only unless `FX1_API_KEY` is set; when it is,
 * pass `apiKey` — it is sent as `X-API-Key`. See docs/FX1_HARNESS_API.md for
 * the wire contract (idempotency, drain, webhooks, version negotiation).
 */

import type { components, paths } from "./schema.d.ts";

export type Backend = components["schemas"]["CompleteRequest"]["backend"];
export type CapabilitiesResponse =
  components["schemas"]["CapabilitiesResponse"];
export type ChatMessage = components["schemas"]["ChatMessage"];
export type BackendProbeRequest = components["schemas"]["BackendProbeRequest"];
export type BackendProbeResponse =
  components["schemas"]["BackendProbeResponse"];
export type GateCheckRequest = components["schemas"]["GateCheckRequest"];
export type GateCheckResponse = components["schemas"]["GateCheckResponse"];
export type ScoreRequest = components["schemas"]["ScoreRequest"];
export type ScoreItem = components["schemas"]["ScoreItem"];
export type ScoreResponse = components["schemas"]["ScoreResponse"];
export type ModerationRequest = components["schemas"]["ModerationRequest"];
export type ModerationResult = components["schemas"]["ModerationResult"];
export type ModerationResponse = components["schemas"]["ModerationResponse"];
export type CompleteRequest = components["schemas"]["CompleteRequest"];
export type CompleteResponse = components["schemas"]["CompleteResponse"];
export type CompleteBatchRequest =
  components["schemas"]["CompleteBatchRequest"];
export type CompleteBatchResponse =
  components["schemas"]["CompleteBatchResponse"];
export type CompletionRecord = components["schemas"]["CompletionRecord"];
export type CompletionListResponse =
  components["schemas"]["CompletionListResponse"];
export type DrainResponse = components["schemas"]["DrainResponse"];
export type EvalListResponse = components["schemas"]["EvalListResponse"];
export type EvalRecord = components["schemas"]["EvalRecord"];
export type EvalSubmitRequest =
  components["schemas"]["EvalSubmitRequest"];
export type EvalSubmitResponse =
  components["schemas"]["EvalSubmitResponse"];
export type EvalSuiteName = EvalSubmitRequest["suite"];
export type HarnessCommandItem =
  components["schemas"]["HarnessCommandItem"];
export type HarnessCommandListResponse =
  components["schemas"]["HarnessCommandListResponse"];
export type HarnessRole = components["schemas"]["HarnessRole"];
export type HarnessRunRequest = components["schemas"]["HarnessRunRequest"];
export type HarnessRunResponse = components["schemas"]["HarnessRunResponse"];
export type HealthResponse = components["schemas"]["HealthResponse"];
export type JobBatchRequest = components["schemas"]["JobBatchRequest"];
export type JobBatchResponse = components["schemas"]["JobBatchResponse"];
export type JobListResponse = components["schemas"]["JobListResponse"];
export type JobStatusResponse = components["schemas"]["JobStatusResponse"];
export type JobSubmitResponse = components["schemas"]["JobSubmitResponse"];
export type MetricsResponse = components["schemas"]["MetricsResponse"];
export type OpenAIChatRequest = components["schemas"]["OpenAIChatRequest"];
export type OpenAIChatResponse =
  components["schemas"]["OpenAIChatResponse"];
export type OpenAIModelList = components["schemas"]["OpenAIModelList"];
export type OpenAIModel = components["schemas"]["OpenAIModel"];
export type OpenAIResponseRequest =
  components["schemas"]["OpenAIResponseRequest"];
export type OpenAIBatchRequest = components["schemas"]["OpenAIBatchRequest"];
export type OpenAIEmbeddingRequest =
  components["schemas"]["OpenAIEmbeddingRequest"];
export type OpenAIEmbeddingItem =
  components["schemas"]["OpenAIEmbeddingItem"];
export type OpenAIEmbeddingResponse =
  components["schemas"]["OpenAIEmbeddingResponse"];
export type ReadyResponse = components["schemas"]["ReadyResponse"];

/** The OpenAI `file` object as served by POST/GET /v1/files. */
export interface OpenAIFileObject {
  id: string;
  object: "file";
  purpose: string;
  filename: string;
  bytes: number;
  created_at: number;
  status: string;
}

/** GET /v1/files listing envelope. */
export interface OpenAIFileList {
  object: "list";
  data: OpenAIFileObject[];
}

/** The OpenAI `batch` object as served by /v1/batches. */
export interface OpenAIBatchObject {
  id: string;
  object: "batch";
  endpoint: string;
  errors: unknown;
  input_file_id: string;
  completion_window: string;
  status: string;
  output_file_id: string | null;
  error_file_id: string | null;
  created_at: number;
  in_progress_at: number | null;
  expires_at: number | null;
  finalizing_at: number | null;
  completed_at: number | null;
  failed_at: number | null;
  expired_at: number | null;
  cancelling_at: number | null;
  cancelled_at: number | null;
  request_counts: { total: number; completed: number; failed: number };
  metadata: Record<string, string> | null;
}

/** GET /v1/batches listing envelope. */
export interface OpenAIBatchList {
  object: "list";
  data: OpenAIBatchObject[];
  first_id: string | null;
  last_id: string | null;
  has_more: boolean;
}

/** One line of a batch output file. */
export interface OpenAIBatchOutputLine {
  id: string;
  custom_id: string;
  response: {
    status_code: number;
    request_id: string;
    body: Record<string, unknown>;
  };
  error: unknown;
}
export type ReceiptIndexItem = components["schemas"]["ReceiptIndexItem"];
export type ReceiptIndexResponse =
  components["schemas"]["ReceiptIndexResponse"];
export type ReceiptVerifyRequest =
  components["schemas"]["ReceiptVerifyRequest"];
export type ReceiptVerifyResponse =
  components["schemas"]["ReceiptVerifyResponse"];
export type ReceiptVerifyBatchRequest =
  components["schemas"]["ReceiptVerifyBatchRequest"];
export type ReceiptVerifyBatchResponse =
  components["schemas"]["ReceiptVerifyBatchResponse"];
export type VersionResponse = components["schemas"]["VersionResponse"];

/** Result of GET /receipts/{sha256}: sealed bytes plus the live-verifier flag. */
export interface FetchedReceipt {
  /** The sealed receipt document, verbatim. */
  receipt: Record<string, unknown>;
  /** Server-side re-verify verdict (X-Fx1-Receipt-Valid response header). */
  valid: boolean;
}

/** JSON body of a 200 GET response for a path in the OpenAPI spec. */
type GetJson<P extends keyof paths> =
  paths[P]["get"] extends { responses: { 200: { content: { "application/json": infer B } } } }
    ? B
    : never;

export interface HarnessApiClientOptions {
  /** Default `http://127.0.0.1:8011` (the `fx1 harness serve` default). */
  baseUrl?: string;
  /** Sent as `X-API-Key` when the server runs with `FX1_API_KEY` set. */
  apiKey?: string;
  /** Injectable fetch implementation (defaults to global fetch). */
  fetch?: typeof fetch;
  /**
   * Wire contract this client speaks. `checkCompat` compares the server's
   * `/harness/version` `api_version` against it. Bump with the server.
   */
  expectedApiVersion?: string;
  /**
   * Retried attempts after the first (default `0` — retries off).
   * Idempotent calls — GETs, keyed submissions, drain, receipt verifies —
   * retry by default; unkeyed writes only retry under `retryWrites`.
   */
  maxRetries?: number;
  /** Initial backoff in ms, doubling per attempt (default `100`). */
  retryBackoffMs?: number;
  /** Also retry unkeyed writes (default `false`). */
  retryWrites?: boolean;
  /**
   * Cap on one retry wait in ms (default `5000`). A server `Retry-After`
   * beyond the budget ends the loop instead of sleeping past it.
   */
  maxRetryWaitMs?: number;
  /**
   * Consecutive transport faults that open the circuit (default `0` —
   * off). While open, calls fail fast with `HarnessTransportError` until
   * `circuitResetMs` elapses and a half-open probe closes or re-opens it.
   */
  circuitBreakerThreshold?: number;
  /** Circuit-open window in ms (default `30000`). */
  circuitResetMs?: number;
  /** Injectable async sleep — tests pass a recorder (default: timer). */
  sleep?: (ms: number) => Promise<void>;
  /** Injectable clock in ms for the breaker (default `Date.now`). */
  now?: () => number;
}

/** Error raised for any non-2xx harness response. */
export class HarnessApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  /** Machine-readable `code` field from the `{detail, code}` envelope. */
  readonly code: string | null;

  constructor(status: number, detail: unknown) {
    const message =
      typeof detail === "object" && detail !== null && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `HTTP ${status}`;
    super(message);
    this.name = "HarnessApiError";
    this.status = status;
    this.detail = detail;
    this.code =
      typeof detail === "object" && detail !== null && "code" in detail
        ? String((detail as { code: unknown }).code)
        : null;
  }
}

/**
 * Raised when the harness is unreachable, retries are exhausted, or the
 * circuit breaker is open — the request never produced an HTTP response
 * (or never will, for a fail-fast open circuit).
 */
export class HarnessTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HarnessTransportError";
  }
}

/** Raised by `checkCompat` when the server's wire contract differs. */
export class HarnessCompatError extends Error {
  readonly report: CompatReport;

  constructor(report: CompatReport) {
    super(
      `harness wire contract mismatch: client speaks api_version ` +
        `${report.clientApiVersion}, server speaks ` +
        `${report.serverApiVersion ?? "unversioned"}`,
    );
    this.name = "HarnessCompatError";
    this.report = report;
  }
}

export interface CompatReport {
  compatible: boolean;
  clientApiVersion: string;
  serverApiVersion: string | null;
  serverFx1Version: string | null;
}

/** One parsed SSE frame. `id` is the frame's `id:` field — the
 * completion stream numbers frames by chunk index, so a dropped keyed
 * stream resumes via `chatCompletionStream`'s `lastEventId`. */
export interface SseEvent {
  event: string;
  data: string;
  id?: string;
}

const DEFAULT_API_VERSION = "1";

const defaultSleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

export class HarnessApiClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly fetchImpl: typeof fetch;
  private readonly expectedApiVersion: string;
  private readonly maxRetries: number;
  private readonly retryBackoffMs: number;
  private readonly retryWrites: boolean;
  private readonly maxRetryWaitMs: number;
  private readonly cbThreshold: number;
  private readonly cbResetMs: number;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly now: () => number;
  private cbFailures = 0;
  private cbOpenUntil = 0;
  /** `X-Fx1-Api-Version` stamped by the most recent response. */
  lastApiVersion: string | null = null;

  constructor(options: HarnessApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:8011").replace(
      /\/+$/,
      "",
    );
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? fetch;
    this.expectedApiVersion = options.expectedApiVersion ?? DEFAULT_API_VERSION;
    this.maxRetries = options.maxRetries ?? 0;
    this.retryBackoffMs = options.retryBackoffMs ?? 100;
    this.retryWrites = options.retryWrites ?? false;
    this.maxRetryWaitMs = options.maxRetryWaitMs ?? 5000;
    this.cbThreshold = options.circuitBreakerThreshold ?? 0;
    this.cbResetMs = options.circuitResetMs ?? 30000;
    this.sleep = options.sleep ?? defaultSleep;
    this.now = options.now ?? (() => Date.now());
    for (const [name, v] of Object.entries({
      maxRetries: this.maxRetries,
      retryBackoffMs: this.retryBackoffMs,
      maxRetryWaitMs: this.maxRetryWaitMs,
      circuitBreakerThreshold: this.cbThreshold,
      circuitResetMs: this.cbResetMs,
    })) {
      if (v < 0 || (name !== "maxRetries" && name !== "circuitBreakerThreshold" && v <= 0))
        throw new RangeError(`${name} out of range: ${v}`);
    }
  }

  private headers(extra?: Record<string, string>): Record<string, string> {
    const h: Record<string, string> = { ...extra };
    if (this.apiKey !== undefined) h["X-API-Key"] = this.apiKey;
    return h;
  }

  private stampVersion(res: Response): void {
    const v = res.headers.get("x-fx1-api-version");
    if (v !== null) this.lastApiVersion = v;
  }

  private async parse(res: Response): Promise<unknown> {
    this.stampVersion(res);
    const text = await res.text();
    if (!res.ok) {
      let detail: unknown = text;
      try {
        detail = JSON.parse(text);
      } catch {
        /* non-JSON error body */
      }
      throw new HarnessApiError(res.status, detail);
    }
    if (!text) return null;
    return JSON.parse(text);
  }

  private static retryAfterMs(res: Response): number | null {
    const raw = res.headers.get("retry-after");
    if (raw === null) return null;
    const secs = Number(raw);
    return Number.isFinite(secs) && secs >= 0 ? secs * 1000 : null;
  }

  /** 429 always retries; 503 only when it carries Retry-After (the
   * in-flight cap — a backend-misconfig 503 never will). */
  private static retryable(res: Response): boolean {
    return (
      res.status === 429 ||
      (res.status === 503 && HarnessApiClient.retryAfterMs(res) !== null)
    );
  }

  private cbTrip(): void {
    if (this.cbThreshold === 0) return;
    this.cbFailures += 1;
    if (this.cbFailures >= this.cbThreshold) {
      this.cbOpenUntil = this.now() + this.cbResetMs;
      this.cbFailures = 0;
    }
  }

  private cbReset(): void {
    this.cbFailures = 0;
    this.cbOpenUntil = 0;
  }

  /**
   * Send one request with the retry/circuit policy. `idempotent` marks
   * calls the server can safely see twice (GETs, keyed deduped POSTs,
   * drain, receipt verifies); unkeyed writes retry only under
   * `retryWrites`.
   */
  private async send(init: {
    method: string;
    path: string;
    body?: unknown;
    /** Raw fetch body (multipart uploads) — sent verbatim, never JSON'd. */
    rawBody?: BodyInit;
    idempotent?: boolean;
    headers?: Record<string, string>;
  }): Promise<Response> {
    if (this.cbThreshold > 0 && this.now() < this.cbOpenUntil)
      throw new HarnessTransportError(
        `circuit open for ${this.baseUrl} — fail fast`,
      );
    const retries =
      init.idempotent === true || this.retryWrites ? this.maxRetries : 0;
    let backoff = this.retryBackoffMs;
    try {
      for (let attempt = 0; ; attempt++) {
        let res: Response;
        try {
          res = await this.fetchImpl(this.baseUrl + init.path, {
            method: init.method,
            headers: this.headers(init.headers),
            ...(init.rawBody !== undefined
              ? { body: init.rawBody }
              : init.body !== undefined
                ? { body: JSON.stringify(init.body) }
                : {}),
          });
          this.stampVersion(res);
        } catch (exc) {
          if (attempt >= retries) {
            throw new HarnessTransportError(
              `harness unreachable at ${init.path}: ${String(exc)}`,
            );
          }
          await this.sleep(backoff);
          backoff *= 2;
          continue;
        }
        if (HarnessApiClient.retryable(res) && attempt < retries) {
          const wait = HarnessApiClient.retryAfterMs(res);
          if (wait !== null && wait > this.maxRetryWaitMs) break;
          await this.sleep(Math.min(wait ?? backoff, this.maxRetryWaitMs));
          backoff *= 2;
          continue;
        }
        this.cbReset();
        return res;
      }
    } catch (exc) {
      if (exc instanceof HarnessTransportError) this.cbTrip();
      throw exc;
    }
    throw new HarnessTransportError(
      `harness ${init.method} ${init.path} exhausted ${retries} retries`,
    );
  }

  private async get<P extends keyof paths>(
    path: string,
  ): Promise<GetJson<P>> {
    const res = await this.send({ method: "GET", path, idempotent: true });
    return (await this.parse(res)) as GetJson<P>;
  }

  private async post(
    path: string,
    body: unknown,
    idempotencyKey?: string,
    opts?: { idempotent?: boolean },
  ): Promise<unknown> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (idempotencyKey !== undefined) headers["Idempotency-Key"] = idempotencyKey;
    const res = await this.send({
      method: "POST",
      path,
      body: body ?? {},
      // A keyed POST dedupes server-side — safe to retry by construction.
      idempotent: opts?.idempotent ?? idempotencyKey !== undefined,
      headers,
    });
    return this.parse(res);
  }

  // ---- introspection ----------------------------------------------------

  /** GET /health — presence booleans only; never env values. */
  health(): Promise<HealthResponse> {
    return this.get("/health");
  }

  /** GET /ready — 503 once drain latches. */
  ready(): Promise<ReadyResponse> {
    return this.get("/ready");
  }

  /** GET /metrics — JSON ops counters, or Prometheus text when asked. */
  async metrics(format?: "json" | "prom"): Promise<MetricsResponse | string> {
    if (format === "prom") {
      const res = await this.send({
        method: "GET",
        path: "/metrics?format=prom",
        idempotent: true,
        headers: { Accept: "text/plain" },
      });
      if (!res.ok) throw new HarnessApiError(res.status, await res.text());
      return res.text();
    }
    return this.get("/metrics");
  }

  /** GET /harness/version — wire contract + package version. */
  version(): Promise<VersionResponse> {
    return this.get("/harness/version");
  }

  /**
   * Compare the server's wire contract to this client's. With `strict`
   * (default) a mismatch raises `HarnessCompatError` whose `.report` carries
   * the full comparison. A peer too old to have `/harness/version` reports
   * `serverApiVersion: null` — incompatible under strict.
   */
  async checkCompat(strict = true): Promise<CompatReport> {
    let serverApiVersion: string | null = null;
    let serverFx1Version: string | null = null;
    try {
      const v = await this.version();
      serverApiVersion = v.api_version ?? null;
      serverFx1Version = v.fx1_version ?? null;
    } catch (err) {
      if (!(err instanceof HarnessApiError && err.status === 404)) throw err;
    }
    const report: CompatReport = {
      compatible: serverApiVersion === this.expectedApiVersion,
      clientApiVersion: this.expectedApiVersion,
      serverApiVersion,
      serverFx1Version,
    };
    if (strict && !report.compatible) throw new HarnessCompatError(report);
    return report;
  }

  /**
   * GET /harness/capabilities — the server's self-describing feature
   * flags and effective limits (batch caps, store bounds, rate limit).
   * Self-configure from this instead of hardcoding server internals.
   */
  capabilities(): Promise<CapabilitiesResponse> {
    return this.get("/harness/capabilities");
  }

  /** GET /harness/commands — registered commands, `role` filters. */
  commands(role?: HarnessRole): Promise<HarnessCommandListResponse> {
    return this.get(role ? `/harness/commands?role=${role}` : "/harness/commands");
  }

  // ---- synchronous work --------------------------------------------------

  /**
   * POST /harness/runs — execute a registered command. `idempotencyKey` (or a
   * request-level `idempotency_key`) dedupes retries server-side.
   */
  run(
    request: HarnessRunRequest,
    idempotencyKey?: string,
  ): Promise<HarnessRunResponse> {
    return this.post("/harness/runs", request, idempotencyKey) as Promise<HarnessRunResponse>;
  }

  /** POST /harness/complete — one gated completion. */
  complete(
    request: CompleteRequest,
    idempotencyKey?: string,
  ): Promise<CompleteResponse> {
    return this.post("/harness/complete", request, idempotencyKey) as Promise<CompleteResponse>;
  }

  /** POST /harness/complete/batch — up to 64 conversations, one backend. */
  completeBatch(
    request: CompleteBatchRequest,
    idempotencyKey?: string,
  ): Promise<CompleteBatchResponse> {
    return this.post("/harness/complete/batch", request, idempotencyKey) as Promise<CompleteBatchResponse>;
  }

  /**
   * POST /harness/backends/{name}/probe — one live gated completion;
   * the verdict (`ok:false` for an unconfigured/unreachable backend) is a
   * payload, not a wire fault. Bypasses and never feeds the circuit.
   */
  probeBackend(
    name: "hosted_k3" | "local_fx1" | "byok",
    request?: BackendProbeRequest | null,
  ): Promise<BackendProbeResponse> {
    return this.post(
      `/harness/backends/${encodeURIComponent(name)}/probe`,
      request ?? null,
    ) as Promise<BackendProbeResponse>;
  }

  /**
   * POST /harness/gate/check — pre-flight text through the honesty gate
   * without spending model tokens; a refusal rides `ok:false`.
   */
  checkText(text: string): Promise<GateCheckResponse> {
    return this.post("/harness/gate/check", {
      text,
    } satisfies GateCheckRequest) as Promise<GateCheckResponse>;
  }

  /**
   * POST /harness/score — run text through the deterministic reward
   * contract; returns the per-item breakdowns (total, components,
   * violations). Advisory: never spends model tokens.
   */
  score(request: ScoreRequest): Promise<ScoreResponse> {
    return this.post("/harness/score", request) as Promise<ScoreResponse>;
  }

  /**
   * POST /v1/moderations — OpenAI-compatible moderation over the honesty
   * gate: per-input {flagged, categories, category_scores} and a
   * content-derived `modr-<sha256>` id. Advisory: never spends model
   * tokens.
   */
  moderations(request: ModerationRequest): Promise<ModerationResponse> {
    return this.post("/v1/moderations", request) as Promise<ModerationResponse>;
  }

  /**
   * GET /harness/completions/{id} — one recorded call from the server's
   * completion log (hashes, usage, verdict; 404 on unknown ids).
   */
  completion(completionId: string): Promise<CompletionRecord> {
    return this.get(
      `/harness/completions/${encodeURIComponent(completionId)}`,
    ) as Promise<CompletionRecord>;
  }

  /**
   * GET /harness/completions/{id}/receipt — the logged call as a sealed
   * `fx1_completion_record.v1` document (POST it to /receipts/verify).
   */
  completionReceipt(
    completionId: string,
  ): Promise<Record<string, unknown>> {
    return this.get(
      `/harness/completions/${encodeURIComponent(completionId)}/receipt`,
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * GET /harness/completions — newest-first window on the completion log.
   */
  completions(filter?: {
    limit?: number;
    backend?: "hosted_k3" | "local_fx1" | "byok";
  }): Promise<CompletionListResponse> {
    const q = new URLSearchParams();
    if (filter?.limit !== undefined) q.set("limit", String(filter.limit));
    if (filter?.backend) q.set("backend", filter.backend);
    const suffix = q.size ? `?${q.toString()}` : "";
    return this.get(`/harness/completions${suffix}`) as Promise<CompletionListResponse>;
  }

  // ---- OpenAI-compatible ingress (/v1) ------------------------------------

  /**
   * GET /v1/models — the OpenAI `list` envelope: `fx1` (the default link)
   * plus the backend names a request's `model` may carry.
   */
  async listModels(): Promise<OpenAIModelList> {
    const res = await this.send({
      method: "GET",
      path: "/v1/models",
      idempotent: true,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as OpenAIModelList;
  }

  /**
   * GET /v1/models/{model} — OpenAI's `models.retrieve`: one card for a
   * listed id; unknown ids throw the 404-class error (`model_not_found`),
   * never a fabricated card.
   */
  async retrieveModel(model: string): Promise<OpenAIModel> {
    const res = await this.send({
      method: "GET",
      path: `/v1/models/${encodeURIComponent(model)}`,
      idempotent: true,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as OpenAIModel;
  }

  /**
   * POST /v1/chat/completions — the OpenAI chat surface over the gated
   * pipeline. Non-streaming only (`stream: true` is rejected here; use
   * `chatCompletionStream`). `headers` carries the fx1 selectors —
   * `X-Fx1-Backend`, `X-Fx1-Fallbacks`, `X-Fx1-Byok-*` — for callers that
   * can't put the `fx1` extension object in the body. Returns the
   * `chat.completion` envelope plus the `X-Fx1-Completion-Id` handle that
   * links the call to its sealed receipt.
   */
  async chatCompletion(
    request: OpenAIChatRequest,
    headers?: Record<string, string>,
    idempotencyKey?: string,
  ): Promise<{ response: OpenAIChatResponse; completionId: string | null }> {
    const res = await this.send({
      method: "POST",
      path: "/v1/chat/completions",
      body: { ...request, stream: false },
      // A keyed call dedupes server-side — safe for the retry policy.
      idempotent: idempotencyKey !== undefined,
      headers: {
        "Content-Type": "application/json",
        ...headers,
        ...(idempotencyKey !== undefined
          ? { "Idempotency-Key": idempotencyKey }
          : {}),
      },
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return {
      response: (await res.json()) as OpenAIChatResponse,
      completionId: res.headers.get("X-Fx1-Completion-Id"),
    };
  }

  /**
   * POST /v1/chat/completions with `stream: true` — SSE
   * `chat.completion.chunk` frames. `onChunk` receives each parsed chunk
   * (an `include_usage` terminal chunk carries `choices: []` + `usage`);
   * the promise resolves with the `X-Fx1-Completion-Id` handle.
   *
   * Frames carry SSE `id:` equal to the chunk index — `lastEventId`
   * resumes a dropped keyed stream: resend the same request with the
   * same `idempotencyKey` and the last received index; the pinned
   * response regenerates byte-identically and already-delivered frames
   * are dropped. Resume without the key fails closed (400); a key with
   * no pinned stream gets 409.
   */
  async chatCompletionStream(
    request: OpenAIChatRequest,
    onChunk: (chunk: Record<string, unknown>) => void,
    headers?: Record<string, string>,
    idempotencyKey?: string,
    lastEventId?: number,
  ): Promise<string | null> {
    const res = await this.send({
      method: "POST",
      path: "/v1/chat/completions",
      body: { ...request, stream: true },
      // Keyed streams replay byte-identically — safe to retry.
      idempotent: idempotencyKey !== undefined,
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...headers,
        ...(idempotencyKey !== undefined
          ? { "Idempotency-Key": idempotencyKey }
          : {}),
        ...(lastEventId !== undefined
          ? { "Last-Event-ID": String(lastEventId) }
          : {}),
      },
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    for await (const ev of readSse(res)) {
      if (ev.data === "[DONE]") break;
      onChunk(JSON.parse(ev.data) as Record<string, unknown>);
    }
    return res.headers.get("X-Fx1-Completion-Id");
  }

  /**
   * POST /v1/responses — the OpenAI Responses surface over the gated
   * pipeline. Non-streaming only (`stream: true` is rejected here; use
   * `responsesCreateStream`). `input` is a string or message-item list
   * (`function_call`/`function_call_output` items carry a tool history);
   * `instructions` prepends a system turn; `text.format` is the
   * post-validated structured-output channel. `tools` takes the
   * flattened Responses spec (`{type: "function", name, description,
   * parameters}`), `tool_choice` is `"none" | "auto" | "required"` or
   * `{type: "function", name}` — calls land in `output` as
   * `function_call` items. Returns the `response` object plus the
   * `X-Fx1-Completion-Id` handle.
   */
  async responsesCreate(
    request: OpenAIResponseRequest,
    headers?: Record<string, string>,
    idempotencyKey?: string,
  ): Promise<{ response: Record<string, unknown>; completionId: string | null }> {
    const res = await this.send({
      method: "POST",
      path: "/v1/responses",
      body: { ...request, stream: false },
      idempotent: idempotencyKey !== undefined,
      headers: {
        "Content-Type": "application/json",
        ...headers,
        ...(idempotencyKey !== undefined
          ? { "Idempotency-Key": idempotencyKey }
          : {}),
      },
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return {
      response: (await res.json()) as Record<string, unknown>,
      completionId: res.headers.get("X-Fx1-Completion-Id"),
    };
  }

  /**
   * POST /v1/responses with `stream: true` — SSE frames in the Responses
   * event grammar (`response.created` … `response.completed`; there is
   * no `[DONE]` sentinel — the completed event is terminal). `onEvent`
   * receives each parsed payload (every payload carries `type`).
   * `lastEventId` resumes a dropped keyed stream exactly like
   * `chatCompletionStream` — frames carry `id:` equal to their index.
   */
  async responsesCreateStream(
    request: OpenAIResponseRequest,
    onEvent: (payload: Record<string, unknown>) => void,
    headers?: Record<string, string>,
    idempotencyKey?: string,
    lastEventId?: number,
  ): Promise<string | null> {
    const res = await this.send({
      method: "POST",
      path: "/v1/responses",
      body: { ...request, stream: true },
      idempotent: idempotencyKey !== undefined,
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...headers,
        ...(idempotencyKey !== undefined
          ? { "Idempotency-Key": idempotencyKey }
          : {}),
        ...(lastEventId !== undefined
          ? { "Last-Event-ID": String(lastEventId) }
          : {}),
      },
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    let terminal = false;
    for await (const ev of readSse(res)) {
      const payload = JSON.parse(ev.data) as Record<string, unknown>;
      onEvent(payload);
      if (payload.type === "response.completed") {
        terminal = true;
        break;
      }
    }
    if (!terminal)
      throw new HarnessApiError(
        0,
        "responses stream ended before response.completed",
      );
    return res.headers.get("X-Fx1-Completion-Id");
  }

  /**
   * POST /v1/embeddings — the OpenAI embeddings surface over the link
   * chain (`model` forwards verbatim to the provider; a link without the
   * embeddings channel answers 501). Returns the `list` envelope and the
   * completion-log id from `X-Fx1-Completion-Id` — the call is recorded
   * exactly like a completion (input/output digests bind the wire).
   */
  async embeddingsCreate(
    request: OpenAIEmbeddingRequest,
    headers?: Record<string, string>,
    idempotencyKey?: string,
  ): Promise<{
    response: OpenAIEmbeddingResponse;
    completionId: string | null;
  }> {
    const res = await this.send({
      method: "POST",
      path: "/v1/embeddings",
      body: request,
      idempotent: idempotencyKey !== undefined,
      headers: {
        "Content-Type": "application/json",
        ...headers,
        ...(idempotencyKey !== undefined
          ? { "Idempotency-Key": idempotencyKey }
          : {}),
      },
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return {
      response: (await res.json()) as OpenAIEmbeddingResponse,
      completionId: res.headers.get("X-Fx1-Completion-Id"),
    };
  }

  // ---- files + batches -----------------------------------------------------

  /**
   * POST /v1/files — upload a batch-input JSONL (multipart). `content` is
   * the raw JSONL bytes; only `purpose: "batch"` and `.jsonl` filenames
   * are served (fail-closed server-side).
   */
  async uploadFile(
    content: string | Uint8Array | Blob,
    filename = "input.jsonl",
    purpose = "batch",
  ): Promise<OpenAIFileObject> {
    const form = new FormData();
    form.append("purpose", purpose);
    const blob =
      typeof content === "string"
        ? new Blob([content], { type: "application/jsonl" })
        : content instanceof Uint8Array
          ? new Blob([content as BlobPart], { type: "application/jsonl" })
          : content;
    form.append("file", blob, filename);
    const res = await this.send({
      method: "POST",
      path: "/v1/files",
      rawBody: form,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as OpenAIFileObject;
  }

  /** GET /v1/files — newest-first listing. */
  async files(): Promise<OpenAIFileObject[]> {
    const res = await this.send({
      method: "GET",
      path: "/v1/files",
      idempotent: true,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return ((await res.json()) as OpenAIFileList).data;
  }

  /** GET /v1/files/{id} — one file's card. */
  async file(fileId: string): Promise<OpenAIFileObject> {
    const res = await this.send({
      method: "GET",
      path: `/v1/files/${encodeURIComponent(fileId)}`,
      idempotent: true,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as OpenAIFileObject;
  }

  /** GET /v1/files/{id}/content — the raw bytes (JSONL in, JSONL out). */
  async fileContent(fileId: string): Promise<string> {
    const res = await this.send({
      method: "GET",
      path: `/v1/files/${encodeURIComponent(fileId)}/content`,
      idempotent: true,
    });
    if (!res.ok) {
      let detail: unknown = await res.text();
      try {
        detail = JSON.parse(detail as string);
      } catch {
        /* non-JSON body */
      }
      throw new HarnessApiError(res.status, detail);
    }
    return res.text();
  }

  /** DELETE /v1/files/{id}. */
  async deleteFile(fileId: string): Promise<Record<string, unknown>> {
    const res = await this.send({
      method: "DELETE",
      path: `/v1/files/${encodeURIComponent(fileId)}`,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as Record<string, unknown>;
  }

  /**
   * POST /v1/batches — run an uploaded file as one batch through the
   * gated pipeline. `endpoint` is `/v1/chat/completions` or
   * `/v1/responses`; the submitter's `X-Fx1-*` headers route every line.
   * `idempotencyKey` replays the submit envelope (shared /v1 idem space).
   */
  async createBatch(
    inputFileId: string,
    endpoint: "/v1/chat/completions" | "/v1/responses",
    metadata?: Record<string, string>,
    idempotencyKey?: string,
  ): Promise<OpenAIBatchObject> {
    const body: Record<string, unknown> = {
      input_file_id: inputFileId,
      endpoint,
      completion_window: "24h",
      ...(metadata !== undefined ? { metadata } : {}),
    };
    const res = await this.send({
      method: "POST",
      path: "/v1/batches",
      body,
      idempotent: idempotencyKey !== undefined,
      headers: {
        "Content-Type": "application/json",
        ...(idempotencyKey !== undefined
          ? { "Idempotency-Key": idempotencyKey }
          : {}),
      },
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as OpenAIBatchObject;
  }

  /** GET /v1/batches/{id} — status + request counts. */
  async batch(batchId: string): Promise<OpenAIBatchObject> {
    const res = await this.send({
      method: "GET",
      path: `/v1/batches/${encodeURIComponent(batchId)}`,
      idempotent: true,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as OpenAIBatchObject;
  }

  /** GET /v1/batches — newest-first page (`after` = last id seen). */
  async batches(filter?: {
    limit?: number;
    after?: string;
  }): Promise<OpenAIBatchList> {
    const q = new URLSearchParams();
    if (filter?.limit !== undefined) q.set("limit", String(filter.limit));
    if (filter?.after) q.set("after", filter.after);
    const qs = q.toString();
    const res = await this.send({
      method: "GET",
      path: `/v1/batches${qs ? `?${qs}` : ""}`,
      idempotent: true,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as OpenAIBatchList;
  }

  /**
   * POST /v1/batches/{id}/cancel — cooperative cancel; the worker checks
   * between lines and lands 'cancelled' with partial output written.
   */
  async cancelBatch(batchId: string): Promise<OpenAIBatchObject> {
    const res = await this.send({
      method: "POST",
      path: `/v1/batches/${encodeURIComponent(batchId)}/cancel`,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as OpenAIBatchObject;
  }

  /**
   * Poll `batch` until a terminal status; resolves with the batch object
   * on 'completed', throws `HarnessApiError` on failed/expired/cancelled,
   * `HarnessTransportError` on timeout — the `waitRun` contract.
   */
  async waitBatch(
    batchId: string,
    pollMs = 500,
    timeoutMs?: number,
  ): Promise<OpenAIBatchObject> {
    const deadline =
      timeoutMs === undefined ? undefined : this.now() + timeoutMs;
    for (;;) {
      const b = await this.batch(batchId);
      if (b.status === "completed") return b;
      if (["failed", "expired", "cancelled"].includes(b.status))
        throw new HarnessApiError(409, `batch ${batchId} ${b.status}`);
      const remaining =
        deadline === undefined ? pollMs : deadline - this.now();
      if (remaining <= 0)
        throw new HarnessTransportError(
          `batch ${batchId} still ${b.status} after ${timeoutMs}ms`,
        );
      await this.sleep(Math.min(pollMs, remaining));
    }
  }

  // ---- /v1 retrieval -------------------------------------------------------

  /**
   * GET /v1/chat/completions/{id} — the stored `chat.completion` envelope
   * (404 when evicted, deleted, or the call went out with `store: false`).
   */
  async retrieveChatCompletion(
    completionId: string,
  ): Promise<Record<string, unknown>> {
    const res = await this.send({
      method: "GET",
      path: `/v1/chat/completions/${encodeURIComponent(completionId)}`,
      idempotent: true,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as Record<string, unknown>;
  }

  /** DELETE /v1/chat/completions/{id} — drop the stored envelope. */
  async deleteChatCompletion(
    completionId: string,
  ): Promise<Record<string, unknown>> {
    const res = await this.send({
      method: "DELETE",
      path: `/v1/chat/completions/${encodeURIComponent(completionId)}`,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as Record<string, unknown>;
  }

  /** GET /v1/responses/{id} — the stored `response` object. */
  async retrieveResponse(
    responseId: string,
  ): Promise<Record<string, unknown>> {
    const res = await this.send({
      method: "GET",
      path: `/v1/responses/${encodeURIComponent(responseId)}`,
      idempotent: true,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as Record<string, unknown>;
  }

  /** DELETE /v1/responses/{id} — drop the stored envelope. */
  async deleteResponse(
    responseId: string,
  ): Promise<Record<string, unknown>> {
    const res = await this.send({
      method: "DELETE",
      path: `/v1/responses/${encodeURIComponent(responseId)}`,
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    return (await res.json()) as Record<string, unknown>;
  }

  // ---- async jobs --------------------------------------------------------

  /** POST /harness/jobs — 202 + job id. */
  submitRun(
    request: HarnessRunRequest,
    idempotencyKey?: string,
  ): Promise<JobSubmitResponse> {
    return this.post("/harness/jobs", request, idempotencyKey) as Promise<JobSubmitResponse>;
  }

  /** POST /harness/jobs/batch — up to 64 run-requests in one call. */
  submitBatch(request: JobBatchRequest): Promise<JobBatchResponse> {
    return this.post("/harness/jobs/batch", request) as Promise<JobBatchResponse>;
  }

  /** GET /harness/jobs/{id} — poll one job. */
  job(jobId: string): Promise<JobStatusResponse> {
    return this.get(`/harness/jobs/${encodeURIComponent(jobId)}`);
  }

  /**
   * GET /harness/jobs/{id}/receipt — the job's ledger record as a sealed
   * `fx1_job_record.v1` document (POST it to /receipts/verify).
   */
  jobReceipt(jobId: string): Promise<Record<string, unknown>> {
    return this.get(
      `/harness/jobs/${encodeURIComponent(jobId)}/receipt`,
    ) as Promise<Record<string, unknown>>;
  }

  /** GET /harness/jobs — list/filter. */
  jobs(filter?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<JobListResponse> {
    const q = new URLSearchParams();
    if (filter?.status) q.set("status", filter.status);
    if (filter?.limit !== undefined) q.set("limit", String(filter.limit));
    if (filter?.offset !== undefined) q.set("offset", String(filter.offset));
    const suffix = q.size ? `?${q.toString()}` : "";
    return this.get(`/harness/jobs${suffix}`);
  }

  /** DELETE /harness/jobs/{id} — cancel a queued job. */
  async cancelJob(jobId: string): Promise<JobStatusResponse> {
    const res = await this.send({
      method: "DELETE",
      path: `/harness/jobs/${encodeURIComponent(jobId)}`,
      idempotent: true,
    });
    return (await this.parse(res)) as JobStatusResponse;
  }

  /**
   * Poll a job until terminal. `pollMs` defaults to 500ms; `timeoutS` bounds
   * the wait (0 = forever). Terminal states are returned, not thrown.
   */
  async waitJob(
    jobId: string,
    opts: { pollMs?: number; timeoutS?: number } = {},
  ): Promise<JobStatusResponse> {
    const pollMs = opts.pollMs ?? 500;
    const deadline =
      opts.timeoutS === undefined || opts.timeoutS === 0
        ? Infinity
        : Date.now() + opts.timeoutS * 1000;
    for (;;) {
      const j = await this.job(jobId);
      if (
        j.status === "succeeded" ||
        j.status === "failed" ||
        j.status === "cancelled"
      ) {
        return j;
      }
      if (Date.now() >= deadline) return j;
      await new Promise((r) => setTimeout(r, pollMs));
    }
  }

  // ---- eval submissions ------------------------------------------------------

  /**
   * POST /harness/evals — submit a seeded eval suite against a backend
   * chain (202). `idempotencyKey` dedupes retries (same body replays the
   * stored eval_id). Evals run under the temperature=0 decode pin and
   * meter under `eval:{suite}:{backend}`.
   */
  submitEval(
    request: EvalSubmitRequest,
    idempotencyKey?: string,
  ): Promise<EvalSubmitResponse> {
    return this.post("/harness/evals", request, idempotencyKey) as Promise<EvalSubmitResponse>;
  }

  /** GET /harness/evals/{id} — the live eval record. */
  eval(evalId: string): Promise<EvalRecord> {
    return this.get(`/harness/evals/${encodeURIComponent(evalId)}`) as Promise<EvalRecord>;
  }

  /** GET /harness/evals — list/filter (status, suite). */
  evals(filter?: {
    status?: string;
    suite?: EvalSuiteName;
    limit?: number;
  }): Promise<EvalListResponse> {
    const q = new URLSearchParams();
    if (filter?.status) q.set("status", filter.status);
    if (filter?.suite) q.set("suite", filter.suite);
    if (filter?.limit !== undefined) q.set("limit", String(filter.limit));
    const suffix = q.size ? `?${q.toString()}` : "";
    return this.get(`/harness/evals${suffix}`) as Promise<EvalListResponse>;
  }

  /**
   * GET /harness/evals/{id}/receipt — the terminal record as a sealed
   * `fx1_eval_record.v1` document (409 while non-terminal; POST the doc
   * to /receipts/verify).
   */
  evalReceipt(evalId: string): Promise<Record<string, unknown>> {
    return this.get(
      `/harness/evals/${encodeURIComponent(evalId)}/receipt`,
    ) as Promise<Record<string, unknown>>;
  }

  /** DELETE /harness/evals/{id} — cooperative cancel of a queued eval. */
  async cancelEval(evalId: string): Promise<EvalRecord> {
    const res = await this.send({
      method: "DELETE",
      path: `/harness/evals/${encodeURIComponent(evalId)}`,
      idempotent: true,
    });
    return (await this.parse(res)) as EvalRecord;
  }

  /**
   * Poll an eval until terminal. `pollMs` defaults to 500ms; `timeoutS`
   * bounds the wait (0 = forever). Terminal records are returned, not
   * thrown — `status` + `error` carry the verdict.
   */
  async waitEval(
    evalId: string,
    opts: { pollMs?: number; timeoutS?: number } = {},
  ): Promise<EvalRecord> {
    const pollMs = opts.pollMs ?? 500;
    const deadline =
      opts.timeoutS === undefined || opts.timeoutS === 0
        ? Infinity
        : Date.now() + opts.timeoutS * 1000;
    for (;;) {
      const e = await this.eval(evalId);
      if (
        e.status === "succeeded" ||
        e.status === "failed" ||
        e.status === "cancelled"
      ) {
        return e;
      }
      if (Date.now() >= deadline) return e;
      await new Promise((r) => setTimeout(r, pollMs));
    }
  }

  // ---- receipt store --------------------------------------------------------

  /** GET /receipts — index the server's sealed-receipt store. */
  receipts(): Promise<ReceiptIndexResponse> {
    return this.get("/receipts");
  }

  /**
   * GET /receipts/{sha256} — the sealed receipt verbatim plus the server's
   * live re-verify verdict (X-Fx1-Receipt-Valid). Unknown hashes reject with
   * a 404 HarnessApiError; malformed digests with 422.
   */
  async receipt(sha256: string): Promise<FetchedReceipt> {
    const res = await this.send({
      method: "GET",
      path: `/receipts/${sha256}`,
      idempotent: true,
    });
    const doc = (await this.parse(res)) as Record<string, unknown>;
    return {
      receipt: doc,
      valid: res.headers.get("x-fx1-receipt-valid") === "true",
    };
  }

  // ---- receipts ----------------------------------------------------------

  /** POST /receipts/verify — verify one receipt payload. */
  verifyReceipt(receipt: Record<string, unknown>): Promise<ReceiptVerifyResponse> {
    return this.post(
      "/receipts/verify",
      {
        receipt,
      } satisfies ReceiptVerifyRequest,
      undefined,
      { idempotent: true },
    ) as Promise<ReceiptVerifyResponse>;
  }

  /** POST /receipts/verify/batch — up to 64, order-preserved. */
  verifyReceipts(
    receipts: Record<string, unknown>[],
  ): Promise<ReceiptVerifyBatchResponse> {
    return this.post(
      "/receipts/verify/batch",
      {
        receipts,
      } satisfies ReceiptVerifyBatchRequest,
      undefined,
      { idempotent: true },
    ) as Promise<ReceiptVerifyBatchResponse>;
  }

  // ---- ops ---------------------------------------------------------------

  /**
   * POST /harness/drain — latch the one-way drain. `waitS` blocks until
   * in-flight empties; `drained` reports which outcome happened.
   */
  drain(waitS?: number): Promise<DrainResponse> {
    const suffix = waitS !== undefined ? `?wait_s=${waitS}` : "";
    return this.post(`/harness/drain${suffix}`, {}, undefined, {
      idempotent: true,
    }) as Promise<DrainResponse>;
  }

  // ---- streaming ----------------------------------------------------------

  /**
   * POST /harness/complete/stream — SSE `token` frames + a `final` frame with
   * the gated `CompleteResponse`, then `[DONE]`. `onEvent` receives each
   * parsed frame; the promise resolves with the final response.
   */
  async streamComplete(
    request: CompleteRequest,
    onEvent: (event: SseEvent) => void,
  ): Promise<CompleteResponse | null> {
    const res = await this.send({
      method: "POST",
      path: "/harness/complete/stream",
      body: request,
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    let final: CompleteResponse | null = null;
    for await (const ev of readSse(res)) {
      onEvent(ev);
      if (ev.event === "final") final = JSON.parse(ev.data) as CompleteResponse;
    }
    return final;
  }

  /**
   * GET /harness/jobs/{id}/events — SSE `job` frames carrying each
   * `JobStatusResponse` snapshot until the job goes terminal.
   */
  async streamJob(
    jobId: string,
    onEvent: (job: JobStatusResponse) => void,
    timeoutS?: number,
  ): Promise<JobStatusResponse | null> {
    const suffix = timeoutS !== undefined ? `?timeout_s=${timeoutS}` : "";
    const res = await this.send({
      method: "GET",
      path: `/harness/jobs/${encodeURIComponent(jobId)}/events${suffix}`,
      idempotent: true,
      headers: { Accept: "text/event-stream" },
    });
    if (!res.ok) throw new HarnessApiError(res.status, await res.json());
    let last: JobStatusResponse | null = null;
    for await (const ev of readSse(res)) {
      if (ev.event !== "job") continue;
      last = JSON.parse(ev.data) as JobStatusResponse;
      onEvent(last);
    }
    return last;
  }

  // ---- webhooks -----------------------------------------------------------

  /**
   * Verify a signed `callback_url` delivery:
   * `X-Fx1-Webhook-Signature` is HMAC-SHA256 hex over
   * `X-Fx1-Webhook-Timestamp` + `"."` + body. Mirrors
   * `fx1.serve.webhooks.verify_webhook` (300s freshness window).
   */
  static async verifyWebhook(opts: {
    body: string;
    signature: string;
    timestamp: string;
    secret: string;
    toleranceS?: number;
  }): Promise<boolean> {
    const toleranceS = opts.toleranceS ?? 300;
    const ts = Number(opts.timestamp);
    if (!Number.isFinite(ts)) return false;
    if (Math.abs(Date.now() / 1000 - ts) > toleranceS) return false;
    const key = await crypto.subtle.importKey(
      "raw",
      new TextEncoder().encode(opts.secret),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"],
    );
    const sig = await crypto.subtle.sign(
      "HMAC",
      key,
      new TextEncoder().encode(`${opts.timestamp}.${opts.body}`),
    );
    const hex = [...new Uint8Array(sig)]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    return hex === opts.signature.toLowerCase();
  }
}

/** Minimal SSE frame reader over a `fetch` response body. */
async function* readSse(res: Response): AsyncGenerator<SseEvent> {
  if (!res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let event = "message";
  let id: string | undefined;
  let dataLines: string[] = [];
  const flush = (): SseEvent | null => {
    if (dataLines.length === 0) {
      event = "message";
      id = undefined;
      return null;
    }
    const out: SseEvent = { event, data: dataLines.join("\n") };
    if (id !== undefined) out.id = id;
    event = "message";
    id = undefined;
    dataLines = [];
    return out;
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      for (;;) {
        const nl = buf.indexOf("\n");
        if (nl < 0) break;
        const line = buf.slice(0, nl).replace(/\r$/, "");
        buf = buf.slice(nl + 1);
        if (line === "") {
          const ev = flush();
          if (ev) yield ev;
        } else if (line.startsWith(":")) {
          continue; // keepalive comment
        } else if (line.startsWith("event:")) {
          event = line.slice(6).trim();
        } else if (line.startsWith("id:")) {
          id = line.slice(3).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).replace(/^ /, ""));
        }
      }
    }
    const ev = flush();
    if (ev) yield ev;
  } finally {
    reader.releaseLock();
  }
}
