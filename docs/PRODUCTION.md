# Production checklist

Operational checklist for deploying and running dipcatcher's research surface.
This is a **research and simulated-paper** deployment: there is no live broker
connectivity, no order-placement path, and no live-trading claim. The
five minimum-evidence conditions that would have to hold before any live claim
are listed in [`INSTITUTIONAL_READINESS.md`](INSTITUTIONAL_READINESS.md); none
hold today.

Use this page as the short checklist. Process start, stop, rollback, the
kill-switch file, and log location live in
[`PRODUCTION_RUNBOOK.md`](PRODUCTION_RUNBOOK.md). The longer procedures live
in [`OPERATIONS_RUNBOOK.md`](OPERATIONS_RUNBOOK.md) (deploy, rollback, incident
response, key rotation, checkpoint spine) and
[`FX1_DEPLOY.md`](FX1_DEPLOY.md) (fx-1 harness service launch, durability,
managed keys).

## 1. Deploy

### Pre-deploy

- [ ] Python 3.12+ and `uv` are installed on the target host.
- [ ] `uv sync --frozen --all-groups --all-extras` succeeds. `uv.lock` is
      authoritative; a drifted lock fails the frozen sync rather than
      re-resolving.
- [ ] `make lint` and `make typecheck` pass.
- [ ] `make test` passes (the PR-gate lane: not network, not slow).
- [ ] `uv run dipcatcher doctor --config configs/research.yaml` exits `0`
      after the first research run has produced a data manifest and a valid
      receipt. On a fresh tree it exits `1` by design — run research first.
- [ ] `uv run dipcatcher verify-research` exits `0`.

```bash
uv sync --frozen --all-groups --all-extras
make lint
make typecheck
make test
uv run dipcatcher research --config configs/research.yaml
uv run dipcatcher verify-research
uv run dipcatcher doctor --config configs/research.yaml
```

### Research API

`dipcatcher api` serves the read-only research/forecast service and binds
loopback by default:

```bash
uv run dipcatcher api --host 127.0.0.1 --port 8000
```

- [ ] The bind host is `127.0.0.1` unless the service is behind an
      authenticated proxy.
- [ ] A non-loopback bind is refused unless `QUANT_API_KEY` is set; when the
      key is set, every non-public route requires `X-API-Key`.
- [ ] `GET /health` answers; `GET /ready` is treated as fail-closed and must
      pass before traffic is cut over.
- [ ] Request bodies stay under the streamed 64 KiB cap, and `POST /backtest`
      stays within its 30-decision-date bound; larger studies run through the
      CLI.

### Container

The image is built from the repository `Dockerfile` (pinned base, non-root
user, `STOPSIGNAL SIGTERM`, liveness healthcheck). Start, stop, and rollback
commands are in [`PRODUCTION_RUNBOOK.md`](PRODUCTION_RUNBOOK.md).

```bash
docker build -t dipcatcher:local .
docker run --rm -p 127.0.0.1:8000:8000 \
  -e QUANT_API_KEY \
  -e DIPCATCHER_BIND_HOST=0.0.0.0 \
  dipcatcher:local
```

- [ ] The image contains the installed runtime, `src/`, `configs/`, the
      liveness script, and the SLO file. It ships **no** `receipts/` and no
      generated `data/`. `/app/data` is writable so readiness can create the
      lake directories. `/ready` stays fail-closed until a data manifest and
      a research receipt exist.
- [ ] A published host port reaches the process only when it binds
      `0.0.0.0`. That bind requires `QUANT_API_KEY`. The default bind is
      loopback. The healthcheck runs inside the container and calls
      `127.0.0.1`.
- [ ] The container healthcheck is allowed to complete before the image is
      promoted (CI does the same). Liveness is `GET /health`. Readiness is
      `GET /ready`.

### fx-1 harness service

The harness is deployed separately from the research API; follow
[`FX1_DEPLOY.md`](FX1_DEPLOY.md) for process launch, durable `--state-dir`,
managed-key bootstrap, and drain-before-shutdown. The canonical launcher is
the only path that enforces the non-loopback auth precondition:

```bash
uv run fx1 harness serve --host 127.0.0.1 --port 8011
```

### Static explorer

The receipt explorer in `web/` is a static build with no backend. Deploy the
build output to any static host; it is read-only and renders fields as stored.

```bash
cd web
npm ci
npm run build
```

### Post-deploy

- [ ] `uv run dipcatcher verify-research` and `make receipts-reverify` pass on
      the deployed tree.
- [ ] Rollback path is known and tested: see "Rollback" in
      [`OPERATIONS_RUNBOOK.md`](OPERATIONS_RUNBOOK.md). Receipts are immutable
      evidence — retract with `uv run dipcatcher tombstone`, never edit or
      delete them.
- [ ] No model checkpoint is deployed from this repository; there is no model
      rollback procedure because there is no released fx-1 checkpoint.

## 2. Configuration via environment

Configuration files live under `configs/` and use an `inherit:` chain rooted
at `base.yaml`. Secrets are supplied through the environment only. Copy
`.env.example` to `.env` (gitignored) and fill values locally; never commit
`.env`, keys, or signed state. Diagnostics such as `dipcatcher doctor` and
`fx1 doctor` report presence flags, never values.

| Variable | Default | Purpose |
|---|---|---|
| `QUANT_DATA_ROOT` | `data` | Data root; `doctor` requires containment under it. |
| `MLFLOW_TRACKING_URI` | `./mlruns` | Local MLflow tracking store. |
| `QUANT_API_KEY` | unset | Non-loopback `dipcatcher api` guard and `X-API-Key` auth for that service. |
| `RESEARCH_API_KEY` | unset | Same guard/auth for the read-only research API (`quant_fund.api.research_api`) when run directly. |
| `QUANT_VENDOR_API_KEY` | unset | Licensed vendor adapter; unset means fail-closed. |
| `GATE_SIGNING_KEY` | unset | Ed25519 repo gate key (`priv_hex:pub_hex`, key-file path, or bare priv hex) for pin/checkpoint signing. |
| `GATE_SIGNING_KEY_NEW` | unset | Incoming key for `dipcatcher rotate-key`; one-shot per rotation. |
| `GATE_QUORUM_KEYS` | unset | Quorum signing pairs for `dipcatcher quorum-sign` when no key file is given. |
| `PROOFCORE_SIGNING_KEY` | unset | Proof-core artifact signing/verification key. |
| `WITNESS_SIGNING_KEY` | unset | ECDSA P-256 witness key for Rekor-anchored checkpoints. |
| `FX1_SIGNING_KEY` | unset | When set, local fx-1 serving requires signed checkpoints. |
| `FX1_API_KEY` | unset | Required before the fx-1 harness service will bind non-loopback. |
| `FX1_API_HOST` / `FX1_API_PORT` | `127.0.0.1` / `8011` | Harness bind address. |
| `MOONSHOT_API_KEY` | unset | Hosted Kimi K3 eval only (`make fx1-eval`). |
| `DIPCATCHER_OBSERVE` | unset | Opt-in audit ledger, metrics, and the alert loop. |
| `DIPCATCHER_SIGSTORE_ID_TOKEN` | unset | Keyless signing of ledger tree heads; no token, no bundle. |
| `DIPCATCHER_SIGSTORE_INSTANCE` | `production` | Sigstore instance selector. |

The process contract for bind, port, logging, shutdown, metrics, the SLO
file, and the kill-switch file is
[`PRODUCTION_RUNBOOK.md`](PRODUCTION_RUNBOOK.md). The same names are listed
in `.env.example`. Do not keep a second copy of those defaults here.

- [ ] Every secret above is injected by the host's secret store, not baked
      into the image or committed.
- [ ] `configs/` is the only directory the API will read configs from; config
      paths outside it are rejected.
- [ ] `runtime.allow_live` stays `false`; a live endpoint refuses at
      configuration time.

## 3. CI

22 workflow files live under `.github/workflows/`. The gates that matter for a
deployment:

| Workflow | Gate |
|---|---|
| `ci.yml` | Lint (`make lint`), typecheck, sharded tests on Python 3.12/3.13, coverage floor (81%), wheel/sdist build + install smoke. |
| `fx1.yml` | fx-1 lint, types, tests, honesty inheritance, corpus-contract smoke. |
| `docs.yml` | `mkdocs build --strict`. |
| `atlas.yml` | Regenerates and diff-checks the architecture diagrams. |
| `arch_guards.yml` | Import-layer boundaries from `configs/arch_boundaries.toml`. |
| `proofcore.yml` | PROOFCORE contracts, provenance DB, layering gate. |
| `release.yml` | Release build/publish, keyed on a `v*.*.*` tag. |

- [ ] Before merging: `make ci` (lint + typecheck + coverage) or the
      individual targets, plus `make fx1-gate` when touching the fx-1 lane.
- [ ] `make epoch-consistency EPOCH_BASE=<base-ref>` passes in the PR lane;
      it needs a base ref and skips without one.
- [ ] CI status is read from `gh run list`, not from the badge. The README's
      CI note records a measured window (2026-10-08) in which the pipeline was
      not returning green verdicts; treat defined-but-unobserved gates as
      unconfirmed.
- [ ] No `v*` release tag has been cut, so `release.yml` has never executed
      and there is no published install artifact. `install.sh` defaults to the
      `v0.4.0` GitHub release asset; until that tag exists, use the
      clone-and-`uv` path above.

## 4. Pin re-signing requires the key holder

The integrity chain is append-only. Any change under a chained directory
(`docs/`, `src/`, `configs/`, `tests/`, `receipts/`, `artifacts/`,
`.github/`, and the rest listed in `quality/epoch_heads.json`) stales epoch
verification until the chain is re-stamped and the pins are re-signed.

Authoring order, when authorized (from
[`OPERATIONS_RUNBOOK.md`](OPERATIONS_RUNBOOK.md)):

```bash
make stamp-epochs     # re-stamp every corpus-epoch chain + the head pin
make sign-pins        # Ed25519-sign the pins; needs GATE_SIGNING_KEY or --key-file
make anchor-pins      # RFC 3161 anchor; network, quiet points only
make checkpoint       # sign the pin state; run last, binds the current signature
```

**This is a key-holder operation.** `make sign-pins`,`make checkpoint`, `make rotate-key`, and `dipcatcher quorum-sign` need `GATE_SIGNING_KEY` (or a key
file) and, for quorum work, `GATE_QUORUM_KEYS`. A contributor without the key
cannot re-sign and must not try:

- [ ] Do **not** edit, regenerate, or delete `gate_pins.sig`, any `*.sig`,
      signing keys, quorum/checkpoint/crown-jewel pin files, or
      `quality/epoch_heads.json`.
- [ ] Do **not** run `stamp-epochs`, `sign-pins`, `anchor-pins`, `checkpoint`,
      `witness-checkpoint`, `rotate-key`, or `dipcatcher quorum-sign` to make a
      local tree look clean.
- [ ] Hand the change to the key holder. A tree that fails epoch or pin
      verification between a content change and re-signing is the expected
      handoff state, not a reason to touch the pin files.
- [ ] After re-signing, verify with `make epoch-consistency EPOCH_BASE=<ref>`,
      `make checkpoint-chain`, `make verify-rotations`, `make verify-witness`,
      and `make evidence-audit`.

**Never commit a secret.** If a key is exposed, follow "Gate-key rotation" in
[`OPERATIONS_RUNBOOK.md`](OPERATIONS_RUNBOOK.md); rollback of a key is
revocation, not an edit to the signed files.

## 5. Ops drills and what is still the owner's

This lane is research and simulated paper. The lab compose file has no broker.
Nothing here is a live-trading claim.

Verified in this tree, by the drills in [`OPS_RUNBOOK_PROD.md`](OPS_RUNBOOK_PROD.md):

- [x] Every Prometheus alert under `deploy/observability/` routes through a
      local Alertmanager webhook, is deduped, and resolves.
      `make -f deploy/prod.mk prod-alert-drill`.
- [x] API, broker, and webhook HMAC secrets overlap on the existing paths
      and the previous token still works until retire.
      `make -f deploy/prod.mk prod-rotate-drill`.
- [x] Lab soak under synthetic `/health` load records p50, p99, and the
      error rate, and checks RSS and file-descriptor slope.
      `make -f deploy/prod.mk prod-soak`. The measured figures are in
      the capacity soak section of the ops runbook. No leak was found.
- [x] `make -f deploy/prod.mk prod-smoke` and
      `make -f deploy/prod.mk prod-dr-drill` still pass on this host.
      The measured RTO is in the disaster-recovery section of the ops runbook.
- [x] `prod-rollback-drill` restores N after a broken N+1 fails the canary,
      with the ledger digest unchanged. The measured time is in the ops runbook.
- [x] Reconcile, backup age, and canary gauges are published by the jobs,
      and `prod-signal-drill` shows those three alerts fire.
- [x] `prod-redaction-audit` scans the soak and the rotation drill.
      The measured soak and the empty finding list are in the ops runbook.
- [x] Incident pages in `docs/incidents/` cover every alert and are linked
      from the Prometheus annotations.
- [x] `prod-order-load` holds pre-trade limits and the kill switch at 10x
      the production request rate on the paper and loopback paths.
- [x] `prod-migrate-drill` carries the ledger, journal, rate store, and
      audit log across a config bump and back. A newer state schema fails
      closed. The measured digests are in the ops runbook.
- [x] `prod-disk-drill` rejects new paper orders when the ledger filesystem
      is full or read-only, pages `DipcatcherWriteHalt`, and accepts again
      when the write lands.
- [x] `prod-time-drill` keeps the rate window, the HTTP bucket, and the
      kill file honest across a wall-clock step, and pages
      `DipcatcherDataStale` and `DipcatcherKillSwitchBlocking`.
- [x] `prod-fs-readonly` pages `DipcatcherFilesystemReadOnly` for a
      read-only non-temporary mount. `DipcatcherDiskPressure` stays quiet
      on that series. promtool exit was 0.
- [x] `prod-shutdown-drill` finishes SIGTERM and SIGINT with exit 0 and
      the same ledger digest on resume. SIGKILL mid-replace leaves a
      complete ledger, and restart applies the journaled intent once.
- [x] `prod-backup-restore-drill` restores a snapshot taken while the
      paper loop is running. Digests match. A stale restore fails closed
      instead of double-submitting.
- [x] `prod-paper-soak` holds a simulated paper loop for 3600 seconds
      and checks RSS, open file descriptors, and thread count. The
      measured series is in the ops runbook.
- [x] `prod-split-brain-drill` keeps two paper processes on one state
      directory from both submitting. A SIGKILL leaves the lock file in
      place, and the next acquirer increments the fencing token.
- [x] `prod-dependency-drill` fails closed on DNS, an expired certificate,
      a broker that is down at session open, and a broker slower than the
      timeout. Recovery accepts the order once. promtool exit was 0.
- [x] `prod-audit-tamper-drill` halts startup when a paper journal or
      audit line is truncated, reordered, or edited. The chain covers new
      runtime lines. No signing key is used.
- [x] `prod-log-journal-drill` rotates a service log under a size cap
      without renaming the order journal or blocking the next submit.

Still the owner's. This slice does not do them:

- [ ] Re-sign. `docs/`, `src/`, and `tests/` moved, so epoch verification is
      stale until the key holder runs `stamp-epochs`, `sign-pins`,
      `anchor-pins`, and `checkpoint`. Do not run those here.
- [ ] fx-1 weights. No fine-tuned weights are published. `fx1.__version__`
      is the harness API, not a model release.
- [ ] A `v*` tag. `release.yml` runs only on a tag, and no tag has been cut.
- [ ] Dockerfile `USER` and `HEALTHCHECK`. Both Dockerfiles already run as
      `dipcatcher` and define a `HEALTHCHECK`. This lane does not edit them.
      Any further image user or healthcheck change stays with the owner.
- [ ] The virtualenv bump. `uv.lock` already pins virtualenv `21.14.12`
      under the constraint `virtualenv>=21.14.9`. This lane does not edit
      the lock. A further bump stays with the owner.
