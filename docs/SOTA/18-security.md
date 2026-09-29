# 18 — Security & Supply-Chain Hardening

Lane deliverable, 2026-09-28. **Read-only analysis**: nothing was modified
except this file. Findings are pinned to `HEAD = 3dafeb7` ("fix(ci): archive
decided reality study") — the working tree was observed mid-flux during this
session (`pyproject.toml`/`uv.lock` identity oscillating between `fx-1` and a
stale `dipcatcher` copy; a `stash@{0}: pre-sync-20260928` exists), so re-verify
line numbers against the tree you adopt from. Cross-cutting policy already
lives in `SECURITY.md`; this doc is the SOTA-practice survey + concrete gap
list + adoption plan, and complements `docs/AUDIT_FRONTIER.md` (call-site
audit) and `docs/MODEL_CARDS.md` (artifact trust discussion).

> Integration note: `docs.yml` runs `mkdocs build --strict` and
> `validation.nav.omitted_files: warn`. A new file under `docs/` that is not in
> `mkdocs.yml` nav will fail the strict docs job. Adding this page (or the
> whole `SOTA/` section) to nav is a one-line follow-up for whoever lands it.

---

## 1. Posture scorecard — what already exists (strong)

The repo is materially ahead of typical Python research codebases on supply
chain. Verified in-tree:

| Control | Evidence |
|---|---|
| Lockfile authoritative, hash-covered | `uv.lock` (2,689 `hash =` entries / 2,504 PyPI URLs); `uv sync --frozen` in every CI job; `uv lock --check` in `ci.yml` audit job + `docs.yml` + pre-commit `uv-lock-check` hook |
| Dependency vuln gate | `ci.yml` `audit` job: `uv export ... | pip-audit==2.10.1 --strict --format json`, artifact uploaded; mirrored by `make audit` |
| PR dependency review | `.github/workflows/dependency-review.yml` (`fail-on-severity: moderate`, OpenSSF scorecard check, dependency-graph fallback gate) |
| Secrets scanning, two layers | `.pre-commit-config.yaml`: gitleaks hook (staged diff, `.gitleaks.toml` narrow allowlists) + custom zero-dep staged-diff scanner `scripts/secret_scan.py`; `.github/workflows/secret-scan.yml`: gitleaks 8.30.1 over full reachable history, **binary pinned by SHA-256**, `--redact`, `permissions: {}` |
| Secret hygiene in code | `.env.example` is 100% commented placeholders; credentials env-only (`src/fx1/serve/backends.py` raises if `MOONSHOT_API_KEY` unset, never hardcodes); `fx1 doctor`/`dipcatcher doctor` report presence flags, never values; child processes get a **scrubbed env** (`src/fx1/data/sources/base.py` `default_runner`: PATH/HOME/LANG + allowlisted credential values only, nothing on argv) |
| API auth fail-closed | `src/quant_fund/api/app.py`: `/health` only public route; `QUANT_API_KEY` compared with `hmac.compare_digest`; when unset, non-loopback clients are refused (403); 64 KiB request cap; Dockerfile binds `127.0.0.1` by default |
| Actions pinning | **All** `uses:` refs in 16 workflows are full 40-char commit SHAs with version comments |
| Least privilege | No `pull_request_target`, no `workflow_run`, no `issue_comment`, no `secrets: inherit` anywhere; `permissions: {}` or `contents: read` at workflow level; `write` grants scoped to jobs that need them (code-scanning, attestations, pages); `persist-credentials: false` on most checkouts |
| Script-injection surface | Only 3 `${{ }}` interpolations inside/near `run:`; two are `workflow_dispatch` inputs behind `if: github.event_name == 'workflow_dispatch'` (privileged actors only) — see finding F3 |
| SLSA / provenance / signing | `release.yml`: Sigstore sign (`sigstore==4.5.0` pinned) + **identity verification** against the workflow OIDC identity, `actions/attest-build-provenance` + `actions/attest` (SBOM attestation), SHA256SUMS; `ci.yml` package job attests on `main` pushes only (fork PRs can't mint id-tokens — correct) |
| SBOM | CycloneDX 1.5 generated from `uv.lock` (`uv export --format cyclonedx1.5 --frozen --no-dev --all-extras`) with an in-line structural validation step; attested |
| PyPI publishing | Triple gate: tag push `v*.*.*` **and** `vars.PYPI_TRUSTED_PUBLISHING == 'true'` **and** GitHub Environment `pypi` as Trusted Publisher. OIDC only — no stored PyPI token |
| SAST / scorecard | CodeQL `security-extended` + custom `.github/codeql/codeql-config.yml` (`src`, `tests`, `scripts`; ignores `third_party`); Bandit 1.9.4 pinned in CI + pre-commit (`make security`); OpenSSF Scorecard weekly + on push/PR, SARIF → code scanning |
| Hardened checkpoint loader (fx1) | `src/fx1/forecast/artifacts.py`: JSON/ONNX/weights-only-torch by default; pickle/joblib/full-module-torch require `allow_unsafe_deserialization=true` **and** a pinned 64-hex `trusted_checkpoint_sha256`; hash checked on the exact bytes deserialized (`compare_digest`); `torch.load(..., weights_only=True)` default path; lock pins torch 2.14.0 (≥2.6, so `weights_only` defaults true even if a caller omits it) |
| Checkpoint release signing | `src/fx1/serve/signing.py`: HMAC-SHA256 over a per-file SHA-256 manifest; `LocalFx1Backend` refuses unsigned checkpoints when `FX1_SIGNING_KEY` is set; docstring notes cosign/Sigstore-compatible shape for future keyless migration |
| Container hygiene | `Dockerfile`: both stages digest-pinned (`python:3.12-slim@sha256:7838...`), multi-stage (no uv/pip in runtime), non-root uid 10001, `USER dipcatcher`, loopback bind; `.dockerignore` present |
| Vendored-model pinning | `src/quant_fund/models/robinhood_plus/constants.py` pins Kronos Hub revisions to 40-char commits; `src/quant_fund/hedge_lab/zoo.py` `snapshot_download(revision=...)`; `src/quant_fund/models/kronos.py` `load_local_predictor` requires local dirs, `local_files_only=True`, optional SHA-256 digest via `validate_local_artifact` |
| numpy deserialization | Every first-party `np.load` call site passes `allow_pickle=False` explicitly (~20 sites in `src/` and `scripts/`), except two cache reads that rely on the safe-by-default (numpy ≥1.16.3) — see F1 |

`gitleaks.toml` allowlists are correctly narrow: per-path + per-rule regex
conditions bound to content-digest fixtures and one synthetic canary
(`tests/fx1/test_sources.py`), with an explicit comment refusing blanket
hex allowlists. This is the right way to do allowlists.

---

## 2. Findings from repo inspection

Severity is impact-if-exploited × likelihood given this repo's threat model
(research harness, no broker connectivity, hosted-API key is the crown jewel).

### F1 — joblib (pickle) deserialization on the harness model path, sidecar optional (MEDIUM)

- `src/quant_fund/models/base.py:210` (`load_joblib_artifact`) and `:275`
  (`JoblibMixin.load`): `joblib.load(path)` runs **after** an *optional*
  `<artifact>.sha256` sidecar check — a missing sidecar loads anyway (legacy
  compatibility). `docs/MODEL_CARDS.md:88` already states the honest
  limitation: the sidecar detects corruption/replacement against a trusted
  sidecar but "does **not** authenticate artifact origin".
- `src/quant_fund/pipeline/forecast/artifacts.py:188`
  (`_paper_challenger_stamp`): bare `joblib.load(path)` on
  `data/metadata/ranker_*.joblib` with **no** sidecar check at all;
  broad `except (TypeError, ValueError, OSError, AttributeError)` around it.
- `src/quant_fund/models/robinhood_plus/engine.py:87`: `joblib.load(path)`.
- Contrast: the **fx1** loader (`src/fx1/forecast/artifacts.py`) is the
  hardened design — opt-in + pinned digest + hash-the-bytes-you-deserialize.
  The harness path never adopted it.
- joblib files are pickle under the hood; loading an attacker-writable
  `.joblib` = arbitrary code execution. Today `data/` is gitignored and local,
  so exploitation requires local write access — hence MEDIUM not HIGH. But the
  repo's own receipts philosophy ("reproducible from a receipt hash") argues
  for the same fail-closed digest rule on model artifacts.
- Also: `tests/support/session_cache.py` pickles synthetic-lake results into a
  shared `~/.cache/dipcatcher` slot (`pickle.loads(blob_path.read_bytes())`,
  lines ~275–428). Test-only, synthetic-gated, but a shared world-readable
  cache dir + `pickle.loads` is a classic local privesg shape on multi-user
  hosts. CI restores this cache (`actions/cache` key
  `dipcatcher-session-*`) — cache poisoning → pickle RCE on the next runner is
  the theoretical chain (GitHub-hosted runner caches are scoped to the repo,
  which limits it).

### F2 — no first-party safetensors policy; Hub download path unpinned (MEDIUM)

- safetensors appears only in vendored Kronos (`third_party/kronos/requirements.txt`
  `safetensors==0.6.2`) and test guards checking `model.safetensors` exists.
  No first-party save/load path uses it; torch checkpoints rely on
  `weights_only=True`, which narrows but does not fully eliminate risk
  (PyTorch docs; HuggingFace guidance: prefer safetensors).
- `src/quant_fund/models/robinhood_plus/torch_backend.py:141-142`:
  `KronosTokenizer.from_pretrained(tok_id)` / `Kronos.from_pretrained(model_id)`
  are called with the **repo id** from `VARIANT_HUB` and **no `revision=`**
  — despite `constants.py:38` documenting that revisions are pinned "because a
  moving `main` would silently mutate the weights under a frozen evaluation".
  The pinned revisions are used by `zoo.py` `snapshot_download` but not here.
  With `allow_network=True` this silently resolves `main` head. Pin
  `revision=spec["tokenizer_rev"]` / `spec["model_rev"]` (or route through the
  already-downloaded local dirs, as `models/kronos.py` does).

### F3 — GitHub Actions residual gaps (LOW–MEDIUM)

- **`persist-credentials: false` missing on 19 of 43 checkout steps.**
  Whole workflows without it: `proofcore.yml` (7 checkouts), `simtest.yml` (2),
  `replay_viz.yml` (2), `web_explorer.yml` (2); plus 6 stragglers in `ci.yml`
  (`mc-engine-smoke`, `formal`, `stress-smoke`, `market-sim`, and two others).
  The GITHUB_TOKEN stays in `.git/config` for the job — the exact thing
  zizmor's `artipacked` audit flags.
- **Script-injection pattern (theoretical):** `simtest.yml:65` interpolates
  `${{ inputs.seeds }} / ${{ inputs.days }} / ${{ inputs.base_seed }}`
  directly into a `run:` line. `workflow_dispatch` requires write access, so
  this is defense-in-depth only — but the canonical fix (env-var indirection)
  is free. `ci.yml:199` (`suite="${{ inputs.suite }}"`) is a choice input
  (closed set) — acceptable.
- **No workflow linter** (zizmor / actionlint) and **no runtime egress
  monitoring** (StepSecurity harden-runner) in any workflow.
- **CodeQL covers Python only** (`languages: python`) while the tree carries
  TypeScript (`web/`, `replay/`, both with `package-lock.json` and Playwright
  e2e) and Rust (`rust/quant_core`, built in CI via maturin). Rust gets
  `cargo clippy -D warnings` but no `cargo audit`/`cargo deny`; JS gets no SAST
  and no dependency vuln scan at all (see F4).

### F4 — npm/JS supply chain is a blind spot (MEDIUM)

- `web/` and `replay/` each have committed `package-lock.json`; CI runs
  `npm ci` (correct, lockfile-respecting) but:
  - `.github/dependabot.yml` registers only `uv`, `pip`, `github-actions` —
    **no `npm` ecosystem entries** for `web/` or `replay/`.
  - No `npm audit` / osv-scanner / Snyk step anywhere.
  - No `npm` min-release-age cooldown (`.npmrc` absent), and no Dependabot
    `cooldown:` config on any ecosystem — freshly published (potentially
    registry-takeover) versions are adoptable same-day. Post-Shai-Hulud this
    is the standard 2026 hardening item.

### F5 — Python audit gate is single-source; no malware/slopsquat screen (LOW)

- CI runs `pip-audit --strict` with the **default PyPA advisory service**
  (`-s pypi`). OSV aggregates PyPA advisories plus other feeds and carries the
  `MAL-*` malicious-package records; running one service misses advisories the
  other has. `uv audit` (OSV-native, reads `uv.lock` directly, 4–10× faster)
  and `UV_MALWARE_CHECK=1` (pre-install OSV malware gate on sync) exist in the
  pinned uv (0.12.16 has `uv audit`) but are unused. Caveats: uv's malware
  check is preview, and `uv audit` ships all locked names to OSV (fine here —
  no private packages).
- No slopsquatting/hallucinated-dependency gate. Low exposure (deps are
  human-curated, lockfile-frozen) but this repo is built with heavy LLM
  assistance — exactly the workflow slopgate-class tools target.

### F6 — workspace hygiene: stray duplicate trees (LOW, operational)

- Untracked `src/src/` (426 files — an **older** `quant_fund` copy, e.g.
  `src/src/quant_fund/models/base.py:105` `joblib.load` with no manifest
  verification) and `tests/tests/` sit in the working tree; `.gitignore` does
  not exclude them; a root-level `NUL` file, `*.prof`, and ~20 `probe_*.py` /
  `run_*.ps1` strays are untracked. AGENTS.md already says the `tests/tests`
  mirror was dropped. Risk: an agent or human edits/commits the wrong tree, or
  a future `git add -A` commits 400+ stale files (the `INFLIGHT` purge-incident
  note shows this tree has been bitten by mass-history operations before).
  Recommend: delete or `.gitignore` `src/src/` + `tests/tests/`, and add a CI
  guard that fails on their reappearance.

### F7 — vendored third_party trust boundary (INFO)

- `third_party/kronos` is git-tracked (93 files) and uses `pickle.load` in its
  finetune scripts, with its own loose `requirements.txt` (`torch>=2.0.0`).
  `SECURITY.md` scopes third_party findings out, and CodeQL ignores the path —
  but `torch_backend.py:_ensure_kronos_on_path()` **prepends**
  `third_party/kronos_src`/`Kronos`/`kronos` to `sys.path` and imports
  `model.kronos` from it. The vendored code executes with full trust whenever
  the robinhood+ torch backend runs. Mitigation that exists: revisions pinned
  at fetch time in `constants.py`/`zoo.py`; keep it that way and never vendor
  from an unpinned source. The vendored `requirements.txt` is not covered by
  pip-audit (audit exports only the project lock).

### F8 — minor dependency hygiene (INFO)

- `python-dotenv` and `pydantic-settings` are declared core deps but **zero
  first-party imports** (`git grep` over `*.py`: `dotenv` 0, `pydantic_settings`
  0). `clarabel` shows 0 direct imports too (may be a cvxpy solver plugin —
  verify before removing). Unused deps are attack surface; `.env.example`'s
  "Copy to .env" workflow is served by shell env or `--env-file`, not code.
  Removing `python-dotenv` (or actually using it with a documented load path)
  would close the loop between docs and code.
- `mkdocs.yml` loads MathJax from `cdn.jsdelivr.net` (third-party CDN, unpinned
  `@3` tag) on published docs pages — supply-chain-adjacent for the docs site
  only.

### What was checked and found clean

- No `shell=True`, no `eval(`/`exec(` on dynamic input, no `verify=False`, no
  `trust_remote_code=True`, no `allow_pickle=True` anywhere in first-party
  `src/` or `scripts/` (grep-verified; the only `pickle.loads` are the fx1
  hardened loader and the test session cache, above).
- No `curl | sh` install patterns in `scripts/`; the one curl-download
  (gitleaks binary in `secret-scan.yml`) is checksum-pinned.
- `urlopen` uses are annotated/pinned-host (`fx1/serve/backends.py:57`
  Moonshot API URL constant; `data/adapters/hf_ohlcv_1m.py:602` injectable
  opener), with `# noqa: S310 / # nosec B310` — bandit's URL-open rule is
  consciously waived at exactly 2 sites, both justified in-line.
- No `.env` on disk; `.env` gitignored; only `.env.example` tracked.
- `MOONSHOT_API_KEY` handling matches best practice: env-only, fail-fast
  RuntimeError, never logged (doctor prints set/unset), `Authorization` header
  built per-request, no key in argv or config files.

---

## 3. Practice summaries (researched, with citations)

### 3.1 Python dependency auditing — pip-audit / OSV / uv audit

- **Scan resolved versions, not ranges.** A `requirements.txt` line
  `requests>=2.0` tells a scanner nothing; the lockfile is the truth. This
  repo does it right (`uv export` from `uv.lock` → pip-audit).
- **pip-audit** (PyPA): default service is the PyPI Advisory Database;
  `-s osv` switches to OSV (aggregates PyPA + GHSA + MAL records); `--strict`
  fails on collection errors; formats include `cyclonedx-json`; docs state it
  "is not a malware-defense tool". Run both services periodically — coverage
  differs. (https://github.com/pypa/pip-audit/blob/main/README.md,
  https://pypi.org/project/pip-audit/2.10.1/)
- **uv audit** (2026, preview): OSV-native, reads `uv.lock` directly (no
  re-resolution), 4–10× faster than equivalent pip-audit runs; also surfaces
  PEP 792 "adverse" project statuses. `UV_MALWARE_CHECK=1` turns every
  `uv sync` into a pre-install gate against OSV `MAL-` advisories — the
  failure mode shifts from "installed something malicious" to "install blocked
  pending review". Caveats: preview API; names are sent to OSV (point
  `--service-url` at a proxy if that matters); lockfile installs bypass
  index-level quarantine metadata, which is exactly what the malware check
  compensates for. (https://astral.sh/blog/uv-audit)
- **osv-scanner** reads `uv.lock` directly and emits SARIF → good independent
  second opinion feeding GitHub code scanning.
- **Dependabot/Renovate strategy:** update PRs are maintenance, not an audit
  gate. 2026 best practice adds a **cooldown**: Dependabot
  `cooldown.default-days` (+ semver-specific overrides) and Renovate
  `minimumReleaseAge` delay adopting freshly published versions, blunting
  registry-takeover attacks (Shai-Hulud class). Known gap: Dependabot cooldown
  doesn't cover transitives
  (https://github.com/dependabot/dependabot-core/issues/14683); package-manager
  level gates (npm `.npmrc` `min-release-age`, pnpm `minimumReleaseAge`, Yarn
  `npmMinimalAgeGate`, Bun `minimumReleaseAge`) run at **install time** and do
  cover transitives. (https://craigory.dev/blog/2026-05-29/package-manager-release-cooldown/)

### 3.2 Secrets scanning — gitleaks / detect-secrets / TruffleHog

- Layered strategy is the 2026 consensus: **gitleaks** at pre-commit + CI
  (fast, offline, regex+entropy, 150+ rules, TOML allowlists) — this repo
  already runs both layers with narrow allowlists, which is the hard part done
  right. **detect-secrets** is the brownfield-baseline tool (records existing
  findings in `.secrets.baseline`, blocks only new ones) — not needed here
  because history is already clean-scanned. **TruffleHog** adds *live
  verification* (calls provider APIs; `--only-verified` ≈ rotate-now list) —
  best as a scheduled history sweep, the one layer this repo lacks.
- Pre-commit is a fast feedback loop, not the authoritative gate: `--no-verify`
  bypasses it, so CI history scanning must remain blocking (it is here).
- Detection ≠ remediation: a committed secret requires rotation +
  `git filter-repo` history purge, not just deletion.
  (https://blog.servarat.net/choosing-a-secrets-scanner-gitleaks-vs-trufflehog-vs-detect-secrets-vs-gitguardian/,
  https://www.systemshardening.com/articles/cicd/secret-scanning-cicd/)

### 3.3 SLSA framework

- SLSA v1.1+ Build track: **L1** provenance exists; **L2** provenance is
  generated *and signed by the build platform* (not user-controlled steps);
  **L3** hardened platform — isolation between builds and signing keys
  inaccessible to user steps. v1.2 (Nov 2025) adds a Source track.
  (https://slsa.dev/spec/v1.1/levels)
- On GitHub Actions, `actions/attest-build-provenance` (OIDC `id-token: write`
  + `attestations: write`) mints in-toto/SLSA provenance into the immutable
  Attestation API — this repo already does it in `ci.yml` (main pushes) and
  `release.yml` (tags), which is the practical L3-equivalent for GHA-hosted
  builds. `slsa-framework/slsa-github-generator` is the alternative for
  release-asset provenance.
- PyPI attestations: produced by `actions/attest` or sigstore-python under a
  Trusted Publisher identity, uploaded via `gh-action-pypi-publish` with
  `attestations: true` — release.yml already wires all three.
  (https://docs.pypi.org/attestations/producing-attestations/,
  https://github.com/sigstore/sigstore-python)

### 3.4 SBOM — CycloneDX for Python

- CycloneDX (OWASP) is the more common Python-ecosystem SBOM; SPDX (LF) is the
  alternative. Best practice: generate **at build time** from the **lockfile**,
  include hashes/pURLs, version it alongside release artifacts, and validate
  structure in CI. `uv export --format cyclonedx1.5` needs no extra tool;
  `cyclonedx-py` and `syft` are the standalone options. PEP 770 (embedding
  SBOMs in wheels) is on the horizon. release.yml already generates, validates,
  checksums, signs, and attests the SBOM — SOTA-conformant.
  (https://bernat.tech/posts/securing-python-supply-chain/)

### 3.5 Typosquatting / dependency-confusion defenses

- uv is first-match by default (stops at the first index that has the package)
  — structurally resistant to pip-style dependency confusion; pin private
  packages with `[tool.uv.sources]` + `explicit = true` indexes if a private
  index is ever added. pip's `--extra-index-url` highest-version-wins is the
  dangerous pattern.
- Hash-pinned lockfiles (present here) defeat swap-after-resolution attacks.
- New-in-2026 class — **slopsquatting**: LLM-hallucinated package names
  pre-registered by attackers. CI gates (slopgate, ai-dependency-guard) score
  newly *introduced* dependency names against freshness/maturity/similarity
  signals. Relevant to an LLM-built repo even though uv.lock freezes the set —
  the risk lands at `uv add` time.
  (https://github.com/lirantal/pypi-security-best-practices,
  https://pypi.org/project/slopgate/)

### 3.6 API-key handling (MOONSHOT_API_KEY et al.)

- Consensus rules, all already satisfied or cheap here: env-only, never in
  source/images/committed `.env`; commit a commented `.env.example` as
  documentation (done); `.env` gitignored + `chmod 600`; OIDC instead of static
  keys in CI (done for PyPI/Sigstore; the hosted-API key stays a developer
  secret — appropriate); separate keys per environment; document revocation;
  wrap in-memory secrets so they can't leak via `print`/exception traces
  (pydantic `SecretStr` — not used; the key is a plain `str` attribute on
  `HostedK3Backend`, low risk since it's never serialized, but `SecretStr`
  would harden repr/logging); rotation on a max-age schedule.
- Env vars are readable by same-user processes (`/proc/<pid>/environ`) and
  inherited by children — the fx1 `default_runner` env scrubbing is the
  correct mitigation and worth replicating anywhere else that spawns
  subprocesses with vendor SDKs.
  (https://aquilax.ai/blog/owasp-secrets-management-environment-variables)

### 3.7 Safe deserialization — pickle vs safetensors in ML artifacts

- Pickle executes arbitrary code on load by design (`__reduce__`); a malicious
  `.pt/.pth/.bin/.pkl` ≈ running a malicious script. Hugging Face documents
  real in-the-wild pickle payloads ("BadTorch"-style). PyTorch ≥2.6 defaults
  `torch.load` to `weights_only=True`, which allowlists safe types and raises
  `UnpicklingError` naming the smuggled global — an `UnpicklingError` is the
  guard working, not a bug to work around. `weights_only=True` *narrows* but
  does not fully eliminate risk; **safetensors** (JSON header + raw tensor
  bytes, no code paths, 100 MB header cap, zero-copy, lazy loading — now a
  PyTorch Foundation project) makes RCE impossible by construction and is the
  recommended format for anything pulled or published.
- Policy shape: prefer safetensors everywhere; treat any pickle checkpoint as
  an unsigned binary from the internet; convert once in a controlled spot
  (conversion tools themselves call `torch.load` — never trust a converter to
  defuse a suspicious model); pin Hub downloads **by commit hash, never by
  name/branch**; audit every `trust_remote_code=True` (none here); prefer
  ONNX/GGUF when the runtime allows.
  (https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors,
  https://huggingface.co/blog/safetensors-joins-pytorch-foundation,
  https://drpranayjha.com/huggingface-safetensors-model-format-security/,
  https://checkmarx.com/blog/free-hugs-what-to-be-wary-of-in-hugging-face-part-2/)

### 3.8 GitHub Actions security — pwn request & secret exposure

- **Pwn request**: `pull_request_target` (or `workflow_run`/`issue_comment`) +
  checkout of fork head + any execution step = attacker code with base-repo
  secrets/token. This repo has **zero** privileged-trigger workflows — the
  strongest possible posture. `actions/checkout` v7 adds a built-in guard
  blocking the common unsafe shape unless `allow-unsafe-pr-checkout: true`.
- Hardening checklist this repo mostly satisfies: workflow-level
  `permissions: {}`, job-level least privilege (done); 40-char SHA pinning
  (done, all 16 files); `persist-credentials: false` on every checkout
  (**19 gaps**, F3); no `${{ github.event.* }}` in `run:` (compliant; the two
  dispatch-input sites are closed-set/privileged); Dependabot for
  `github-actions` (done); OIDC over stored tokens (done); secrets at step not
  job scope (only secret use is job-level `env:` in proofcore — acceptable,
  could be step-scoped); no cache restore in release/publish workflows
  (release.yml doesn't use actions/cache — correct).
- Additions worth making: **zizmor** (Actions-native SAST: template injection,
  credential persistence, excessive permissions, impostor refs; SARIF output,
  offline-capable) and **StepSecurity harden-runner** as first step of
  sensitive jobs (egress audit → later `egress-policy: block`; runtime
  detection that no static tool provides).
  (https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target,
  https://zizmor.sh/, https://github.com/step-security/harden-runner,
  https://www.wiz.io/blog/github-actions-security-guide)

---

## 4. Adoption plan

Phased; each item lists effort and an acceptance criterion so a lane can pick
it up without re-deriving. Nothing here conflicts with the honesty contract
(security gates are not research claims and must never be weakened to green a
build).

### P0 — this week (cheap, closing live gaps)

1. **`persist-credentials: false` everywhere.** Add to the 19 missing
   checkouts (`proofcore.yml` ×7, `ci.yml` ×6, `simtest.yml` ×2,
   `replay_viz.yml` ×2, `web_explorer.yml` ×2). Acceptance: grep shows
   `uses: actions/checkout` count == `persist-credentials: false` count
   (43/43) across `.github/workflows/`.
2. **Dependabot `npm` ecosystems** for `/web` and `/replay`; add
   `cooldown: {default-days: 7, semver-major: 14}` to all five ecosystems
   (uv, pip, npm ×2, github-actions). Acceptance: dependabot.yml parses;
   first npm PR observed.
3. **Pin the Kronos Hub revision in `torch_backend.load_pretrained_predictor`**
   (`revision=spec["tokenizer_rev"]` / `spec["model_rev"]`, or require local
   dirs like `models/kronos.py`). Acceptance: no `from_pretrained` call in
   first-party code resolves a moving ref; unit test asserts the revision kwarg.
4. **simtest.yml input indirection:** move `${{ inputs.* }}` into `env:` and
   reference `"$SEEDS"` etc. in the run block. Acceptance: zero `${{ }}`
   inside any `run:` block repo-wide.
5. **Workspace hygiene:** delete (or gitignore + CI-guard) `src/src/`,
   `tests/tests/`, root `NUL`, stray `probe_*.py`/`*.prof`/`run_*.ps1`.
   Acceptance: `git status --porcelain | grep '??'` shows no duplicate source
   trees; a `tests/unit/test_repo_hygiene.py` fails if `src/src` reappears.

### P1 — next two weeks (audit-gate breadth)

6. **zizmor workflow lint** as a CI job (offline, SARIF → code scanning) and
   optionally a pre-commit hook. Acceptance: zizmor clean at `regular`
   persona; findings triaged with per-rule waivers in `zizmor.toml` if needed.
7. **Second-opinion vuln source:** add `pip-audit -s osv` (or `uv audit`) to
   the existing audit job, or a weekly scheduled osv-scanner SARIF upload.
   Evaluate `UV_MALWARE_CHECK=1` on CI syncs (preview — monitor before making
   it blocking). Acceptance: two independent advisory sources gate weekly;
   malware-check documented as warn-mode until stable.
8. **JS dependency audit:** `npm audit --audit-level=high` (or osv-scanner on
   `web/package-lock.json` + `replay/package-lock.json`) in `web_explorer.yml`
   / `replay_viz.yml`; add `min-release-age=7` `.npmrc` per tree (requires npm
   ≥11.10 on the runner; node 22 is already pinned). Acceptance: PR touching
   `web/**` fails on a seeded vulnerable lock entry (regression fixture).
9. **CodeQL matrix extension:** add `javascript-typescript` (web/, replay/) and
   `actions` languages; consider `rust` when CodeQL GA supports it, else add
   `cargo audit`/`cargo deny` to the `rust-accel` job. Acceptance: codeql.yml
   lists ≥3 languages; SARIF categories appear in the Security tab.
10. **TruffleHog scheduled history sweep** (weekly, `--only-verified`,
    redacted output) next to the gitleaks job — gitleaks says *a pattern
    matched*, TruffleHog says *the key is live*. Acceptance: workflow exists,
    first clean run recorded.

### P2 — this quarter (policy: safetensors + artifact trust)

11. **Safetensors policy for first-party torch artifacts** (extends
    `docs/MODEL_CARDS.md`):
    - New fx-1/quant_fund checkpoints are saved as `.safetensors` (weights) +
      JSON sidecars (config/metadata); `torch.save` of full models is banned
      in first-party code (ruff/bandit gate: no `torch.save` outside an
      allowlist, mirroring the existing `noqa: S301` discipline).
    - `from_pretrained` calls pass `use_safetensors=True` (or load via
      `safetensors.torch.load_file`) wherever the format is available;
      pickle-format Hub artifacts get a one-time controlled conversion, never
      an in-place load on a shared host.
    - Loading any `.pt/.pth` keeps `weights_only=True` explicitly (don't rely
      on the 2.6 default surviving a downgrade), and an `UnpicklingError`
      naming an unexpected global is treated as a security event, not
      suppressed.
    - The `fx1` trusted-digest ladder becomes the template: harness
      `JoblibMixin`/`load_joblib_artifact` gain a **required-digest mode**
      (config flag, default-on for new artifacts): no sidecar ⇒ refuse, with a
      documented legacy-allowlist for existing `ranker_*.joblib` files until
      retrained. `_paper_challenger_stamp` routes through the same loader
      instead of bare `joblib.load`.
    Acceptance: a planted tampered `.joblib` fails closed in a test; new
    training runs emit `.safetensors` + digest sidecars; bandit/ruff gate
    blocks a fresh `torch.save`/bare `joblib.load` in `src/`.
12. **Signed-artifact story for model weights:** the HMAC `release.sig` ladder
    (fx1) migrates to Sigstore keyless signing for anything distributed (the
    module docstring already anticipates this); harness joblib sidecars gain
    provenance stamps (who/when/git-rev) written by training, verifiable by
    `verify-research`-style checks — aligning model artifacts with the
    receipts philosophy.
13. **harden-runner** (egress-policy: audit) on `release.yml`, `ci.yml`
    package job, and `secret-scan.yml` first; move to `block` with an
    allowlist once two weeks of audit logs are clean. Acceptance: egress
    baseline committed; any new outbound host fails the job.
14. **Dependency hygiene:** drop or wire up `python-dotenv` /
    `pydantic-settings`; verify `clarabel` is genuinely needed (cvxpy solver)
    or move it to the extra that uses it. Consider a slopsquat screen
    (slopgate-style) on any PR that changes `pyproject.toml` dependencies —
    low priority while `uv.lock` review is manual, mandatory if agents ever
    get `uv add` autonomy. Pin/self-host the MathJax CDN asset for the docs
    site.

### Explicitly not adopting (with reasons)

- **detect-secrets baseline** — gitleaks + full-history CI scanning already
  covers it; a baseline file would add drift surface.
- **Commercial SCA/secret platforms** — repo is single-maintainer research
  code; the free layered stack (gitleaks/TruffleHog/pip-audit+OSV/zizmor/
  Scorecard/CodeQL) is sufficient and auditable.
- **Blocking `uv audit` today** — preview API; adopt warn-mode first (P1.7).

---

## 5. Summary judgment

Supply-chain **provenance** (lockfile hashes, SBOM, Sigstore, SLSA
attestations, Trusted Publishing, pinned actions, least-privilege tokens,
two-layer secret scanning) is already at or above SOTA for a Python research
repo — most teams never reach the release.yml bar. The real gaps are
**deserialization trust on the harness model path** (optional sha256 sidecars,
bare `joblib.load`, no safetensors policy — F1/F2), **JS/Rust/npm blind spots
in the audit net** (F3/F4), and **single-source vulnerability data** (F5).
All are fixable in the P0–P2 sequence above without touching the honesty
contract or any research code paths.
