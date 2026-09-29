# SOTA 10 — Reproducibility & Determinism Engineering

Lane: **REPRODUCIBILITY & DETERMINISM ENGINEERING**. This note summarizes
state-of-the-art practice (2020–2026) for making research claims reproducible
from immutable, content-addressed evidence; audits the dipcatcher / fx-1
receipt-and-verification stack against that practice; and proposes a concrete
adoption plan. It modifies no code — it is the design record for the next
receipt-schema generation (`receipt.v3`).

The honesty contract this lane serves: **receipts are immutable evidence**
(`receipts/`, `verify-research`); every research claim must be reproducible
from a receipt hash.

---

## Part 1 — SOTA practice summaries

### 1.1 Deterministic training & eval

**The two sources of nondeterminism.** RepDL (Chen, Zhang, Xie; Microsoft
Research, Oct 2025) frames the problem cleanly: nondeterminism comes from
(1) random number generation and (2) floating-point computation. RNG
nondeterminism is *solved* by seeding discipline; FP nondeterminism is the
hard one, because IEEE-754 addition is non-associative —
\((a+b)+c \neq a+(b+c)\) — and GPU kernels schedule atomic reductions
dynamically, so accumulation order varies run-to-run, across batch sizes, and
across tensor-parallel (TP) configurations.

**The standard PyTorch recipe** (necessary but not sufficient):

- `torch.manual_seed(s)` plus per-worker seeds via `torch.Generator` and
  `worker_init_fn`; seed Python `random`, NumPy, and `PYTHONHASHSEED`.
- `torch.backends.cudnn.deterministic = True`,
  `torch.backends.cudnn.benchmark = False`.
- `torch.use_deterministic_algorithms(True)` — errors (not warns) on any op
  without a deterministic implementation; requires
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) from CUDA ≥ 10.2 for cuBLAS
  determinism.
- Pin threadpools: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` — BLAS reduction order changes with thread count, so a
  cross-machine sweep must record (and ideally pin) it.

**Beyond flags — bit-level reproducibility.** The 2025 wave of work closes the
FP-order gap properly:

- **RepDL** enforces *correct rounding* and *order invariance*: reductions
  (sum, matmul, conv) use a fixed reduction order while retaining parallelism;
  each op is built from the same basic ops in the same order; distinct
  computation orders get distinct APIs.
- **TBIK** (Tree-Based Invariant Kernels, arXiv:2511.17826) imposes a fixed
  full binary-tree reduction topology shared by intra-GPU MatMul and
  inter-GPU collectives, making results bit-identical *across TP sizes* —
  which eliminates the training–inference probability mismatch in RL pipelines
  where FSDP (TP=1) trains and multi-GPU TP rolls out.
- **LayerCast / batch-invariant kernels** similarly fix accumulation order
  (FP32 accumulation, TP-invariant reduction trees) so batch size and layout
  stop perturbing logits.

**Practical takeaway.** A research harness should (a) seed everything,
(b) flip the determinism flags, (c) *record the determinism configuration in
the receipt* as a verifiable fingerprint (torch/cuDNN/CUDA versions,
`CUBLAS_WORKSPACE_CONFIG`, thread counts), and (d) only claim bit-level
reproducibility when the reduction-order literature (RepDL/TBIK-class) is
actually in the stack. Claiming less than you enforce is the honest failure
mode; dipcatcher's `set_global_seed` docstring already does this correctly
("GPU determinism is not claimed").

### 1.2 Content-addressed artifacts: hashing schemes & Merkle structures

**Hash choice.**

- **SHA-256** (FIPS 180-4) remains the lingua franca: git, SLSA, in-toto,
  Sigstore, OCI, DVC (via SHA-256 mode), and MLflow all speak it. SHA-NI
  hardware acceleration makes it ~0.3–2 GB/s per core.
- **BLAKE3** (2020) is a tree hash built on BLAKE2/Bao: SIMD-parallel,
  typically 6–13+ GB/s on large blobs (roughly an order of magnitude over
  SHA-256), with verified streaming and resumable hashing. It is *not* yet a
  NIST standard, so the pragmatic pattern is: **SHA-256 for interop and
  seals; BLAKE3 (multihash-tagged, e.g. `0x1e` prefix) as an optional fast
  path for bulk object stores** — never as the sole digest in a published
  receipt.
- Self-identifying digests (**multihash**) future-proof schemas: store
  `<algo>:<hex>` rather than a bare 64-hex string.

**Merkle structures — two distinct tools for two distinct jobs.**

- **RFC 6962 (Certificate Transparency)** trees are *order-sensitive* and
  support append-only semantics: `SHA-256(0x00 || leaf)` /
  `SHA-256(0x01 || L || R)`, with inclusion proofs ("this entry is at
  position i") and consistency proofs ("the log only ever grew"). This is the
  right structure for *logs* (audit trails, receipt registers).
- **Sorted-leaf commitment trees** (order-independent) are the right
  structure for *artifact sets* — a directory, a dataset snapshot, a proof
  bundle — where you want "these exact N files, in any enumeration order"
  bound to one root. Domain separation (`leaf:`/`node:` prefixes) prevents
  second-preimage-style collisions across levels. Git, IPFS, DVC, lakeFS
  prolly trees, and Trillian all follow one of these two patterns.

**Dedup & storage.** Content-addressable stores keyed by
`objects/<algo>/<xx>/<digest>` (git, DVC, lakeFS, OCI) give deduplication,
idempotent writes, and free tamper detection: a byte flip changes the address.
The receipt then stores *only* the digest, and any reader can re-derive it.

### 1.3 Model registries & ML artifact tracking (MLflow patterns)

MLflow's durable contribution is the **run → artifacts → model version**
chain: every experiment run gets an immutable `run_id`, artifacts are written
under the run and content-addressed in modern backends (e.g. `ar-io-mlflow`
style stores commit artifacts by hash and make the run metadata itself a
hash-committed object), and the registry layers *mutable* stage labels
(staging/production) over *immutable* versions. Key patterns worth stealing
even without MLflow:

- **Model signatures** (input/output schema + example) hashed with the model —
  a receipt should bind the *interface*, not just the weights.
- **Logged params/metrics are append-only per run**; corrections are new runs,
  never edits.
- **Registry state transitions are themselves events** with actor + timestamp
  — analogous to signed tree heads over a receipt log.
- **OpenLineage ML facets** (`mlTrainingEnvironment`, `mlDeterminism`,
  `mlModelVersion`) attach environment/determinism metadata to runs in a
  standardized way — the industry's answer to "what exactly produced this
  model".

### 1.4 DVC-style data versioning

DVC's pattern: large files live in a local/remote **content-addressable
cache**; git stores tiny **`.dvc` pointer files** (`outs: [{path, size,
hash}]`) plus the pipeline DAG (`dvc.yaml`/`dvc.lock` with per-stage input
and output hashes). Reproducing = checkout git → `dvc pull` → hashes verify
every materialized byte. Links (reflink/hardlink/symlink/copy) are a
performance detail; the *contract* is: **git commits never contain data, only
digests; data movement is always hash-verified**. lakeFS generalizes this to
object stores with prolly trees (Graveler) giving git-like branches/commits
over the same content-addressed objects. The pattern maps 1:1 onto a lake
with immutable snapshot manifests: `snapshot_id = H(canonical manifest of
{logical_path, content_sha256, size} ...)`.

### 1.5 Environment locking & reproducible builds

- **`uv.lock`** is a *universal* (cross-platform) lockfile: exact resolved
  versions + hashes + markers for every platform. `--frozen` installs exactly
  the lockfile; `--locked` additionally asserts the lock is up-to-date with
  `pyproject.toml`. `.python-version` pins the interpreter. This is the
  current best practice for Python env reproducibility and dipcatcher already
  mandates `uv sync --frozen` in CI parity.
- **Reproducible builds**: `SOURCE_DATE_EPOCH` normalizes embedded timestamps
  (zip/tar mtimes, wheel metadata); `strip-nondeterminism` does the same for
  archives post-hoc; normalized umask/owner/file-order yields bit-for-bit
  wheels. The Reproducible Builds project's definition — *anyone can rebuild
  the exact same bits from the declared sources* — is the target for published
  artifacts (wheels, corpora, checkpoints).
- **SBOM**: CycloneDX 1.5+ captures the full dependency graph with hashes;
  pairing the SBOM digest into the receipt closes "which bytes of third-party
  code ran".

### 1.6 Provenance standards (C2PA / W3C PROV / OpenLineage / in-toto / SLSA / Sigstore)

- **W3C PROV-O** is the ontology: *Entity* (artifact), *Activity* (run),
  *Agent* (who/what ran it), linked by `wasGeneratedBy`, `used`,
  `wasAssociatedWith`. Every receipt schema in this repo is an implicit PROV
  graph; making that mapping explicit costs nothing and buys
  tool-interoperability.
- **OpenLineage** operationalizes PROV for data/ML pipelines: `RunEvent`s
  with typed **facets** (schema, data quality, ML training environment,
  *determinism*, source version) emitted at START/COMPLETE/FAIL. The
  `mlDeterminism` facet is precisely the "determinism configuration
  fingerprint" recommended in §1.1.
- **in-toto / SLSA v1.1**: build provenance = a signed attestation
  (DSSE envelope) stating *what* was built, *from what* sources (commit,
  parameters), *by whom* (builder identity), *under which* build type. SLSA
  levels formalize the trust ladder from "provenance exists" (L1) through
  "hosted, isolated, non-falsifiable provenance" (L3). GitHub Artifact
  Attestations (`actions/attest-build-provenance`) mint these automatically.
- **Sigstore**: keyless signing via Fulcio (short-lived OIDC-bound certs) +
  Rekor (public transparency log). The transparency-log property is key:
  signatures become *publicly auditable* without managing long-lived secrets.
- **C2PA** contributes the hard/soft binding distinction: a **hard binding**
  (cryptographic hash over bytes) proves *this exact file*; a **soft
  binding** (perceptual/watermark hash) survives transcoding. Research
  receipts need hard bindings; anything claiming "same dataset modulo
  re-export" needs an explicitly labeled soft binding.

**Synthesis.** 2026 SOTA = *determinism config recorded as a fingerprint* +
*everything content-addressed* + *a signed, transparent log committing the
whole evidence set* + *standardized provenance metadata*. No single tool
provides all four; the stack composes them.

---

## Part 2 — Audit of the current receipt / verification design

### 2.1 What exists (verified by inspection)

| Layer | Module | Mechanism |
|---|---|---|
| Research verifier | `quant_fund.research.verify` (`verify-research`) | Required provenance (`run_id`, `git_revision`, `config_sha256`, `dataset_sha256`, `git_worktree_sha256`, dataset content hash, northset inputs hash, runtime versions), hypothesis records H1–H99, benchmark-family consistency, forbidden-metric scans, immutable JSON/MD artifact pointers |
| Unified envelope | `quant_fund.research.receipt_v2` (`verify-receipt`) | `receipt.v2`: dataset/params digests, `code_files` map + `code_sha256`, environment fingerprint (Python/NumPy/Polars/SciPy, BLAS/LAPACK build, threadpoolctl rows, self-digest `fingerprint_sha256`), `live_pnl_claim: false`, verdict, canonical `receipt_sha256` seal; JSON-Schema published beside the module; v1 fallback under two digest conventions |
| Canonicalization | `quant_fund.utils.hashing` | `canonical_json_bytes` (sorted keys, compact, `allow_nan=False`, stable set/ndarray/datetime handling), streamed 1 MiB `hash_file`, order-independent frame fingerprints |
| Audit ledger | `quant_fund.audit.{ledger,merkle,signing,verify}` | **RFC 6962** append-only log (`entries.jsonl`, order-sensitive, `0x00`/`0x01` domain separation), inclusion + consistency proofs, **signed tree heads** (Ed25519 offline; Sigstore keyless when an OIDC token exists — fail-closed, no placeholder bundles) |
| PROOFCORE | `quant_fund.proofcore.{contracts,provenance}` | Pydantic contracts with strict 64-hex fields, **sorted-leaf Merkle root** with `PC:leaf:`/`PC:node:` domain separation, DuckDB append-only bundle/trial ledger |
| Proof bundles | `quant_fund.proof.{bundle,verify}` | Self-hash excluding mutable fields, sidecar parquet/metrics/config hashing, HMAC signature, **independent recomputation** of headline metrics from trade logs |
| PIT vault | `quant_fund.pit.{manifest,vault}` | Write-once bitemporal vault, streamed chunk SHA-256, `manifests/rNNNNNNN.json` hash chain, atomic pointer updates |
| Lakehouse | `quant_fund.data.lakehouse.{store,lineage,receipts}` | Content-addressed `objects/sha256/xx/<digest>` store, immutable snapshot manifests (`snapshot_id = H(canonical manifest)`, idempotent, byte-diff rejection), lineage records binding dataset → inputs → `code_hash` → params, cycle/drift verification |
| fx-1 pipeline | `fx1.train.receipts`, `fx1.data.ledger`, `fx1.serve.{signing,attestation}` | Training receipts (config/corpus/split/eval-base SHA-256s, seed, git revision + dirty flag, env fingerprint); **hash-chained corpus ledger** (source → transform-code → example, every exclusion recorded); release signing (per-artifact SHA-256 manifest + detached HMAC, cosign-shaped, serving refuses unsigned checkpoints); TEE attestation schemas (SEV-SNP/TDX/CC) |
| Supply chain | `.github/workflows/{ci,release}.yml` | `uv sync --frozen` parity, CycloneDX 1.5 SBOM, Sigstore signing, SLSA v1.1 build provenance via `actions/attest-build-provenance` |

This is a genuinely multi-tier architecture, and the tier separation is
correct: *logs* use RFC 6962 (order-sensitive), *artifact sets* use sorted
Merkle roots (order-independent), *chains* (PIT manifests, corpus ledger) use
prev-hash links. That matches SOTA usage precisely (§1.2).

### 2.2 Measured state of `receipts/` (the evidence directory itself)

`uv run --frozen dipcatcher verify-receipt receipts\<f>` over all 10
committed receipts, run for this audit:

- **PASS (3):** `capacity_eval_cd0854242ed8a9ec.json`,
  `fleet_eval_5ddf15b0dc7d3ca1.json`, `rankic_eval_9ebdad7da83e7348.json`.
- **FAIL (7):** `adaptive_mix_20asset_1d_20260922.json`,
  `adaptive_mix_band_search_20asset_1d_20260922.json`,
  `basis_pair_candidate_20asset_1d_20260922.json`,
  `basis_reversion_screen_20asset_1d_20260922.json`,
  `dip_bench_crypto_1d_20260925.json`,
  `fast_replay_p42_conformance_20260927.json`,
  `incumbent_bench_qlib.json` — all with
  `receipt_sha256_missing_or_invalid` (unsealed legacy schemas:
  `adaptive_mix_replay.v1`, `basis_pair_candidate.v1`,
  `cross_sectional_rankic.v1`-adjacent, `fx1.dip_bench/v1`, etc.).

**70 % of the committed evidence directory fails its own verifier.** The
verifier is fail-closed and correctly says so; the problem is that nothing
gates *committing* unsealed receipts into `receipts/`, and the honesty
contract ("every research claim reproducible from a receipt hash") is
therefore unenforceable for those seven files: an unsealed JSON can be edited
in place with zero detection.

### 2.3 Strengths

1. **Right cryptographic primitives, right places.** RFC 6962 with correct
   domain separation and both proof types; sorted-leaf trees for artifact
   sets; prev-hash chains for append-only manifests; signed tree heads with a
   fail-closed Sigstore path.
2. **Independent recomputation.** `proof.verify` re-derives headline metrics
   from trade logs instead of trusting the receipt's numbers — the single
   strongest anti-forgery control in the stack.
3. **Environment fingerprinting ahead of most shops.** BLAS/LAPACK build +
   loaded threadpools + self-digesting fingerprint (§1.1 asks for exactly
   this).
4. **Content-addressed lakehouse** with idempotent, byte-diff-rejecting
   snapshot commits and code-hash-bound lineage — DVC/lakeFS-grade.
5. **Honesty integration is structural**, not aspirational: forbidden-metric
   scans run inside `verify-receipt`, `live_pnl_claim` is `Literal[False]`,
   SYNTHETIC labeling is enforced.
6. **Supply chain**: `uv.lock` + `--frozen` CI parity + SBOM + SLSA v1.1 +
   Sigstore on releases.

### 2.4 Weaknesses & gaps (hash coverage, tamper detection, hygiene)

**W1 — Unsealed legacy receipts in the evidence directory (critical).**
§2.2. No CI gate re-verifies `receipts/*.json`; tampering with the seven
unsealed files is undetectable.

**W2 — Receipts themselves are unsigned (high).** `receipt.v2` seals with a
*keyless* SHA-256: anyone who can write the file can re-seal it after edits.
The audit ledger's Ed25519/Sigstore STH machinery exists but receipt files in
`receipts/` are not required to have corresponding ledger entries + inclusion
proofs. The seal detects *accidental* corruption, not *motivated* rewrite.

**W3 — No Merkle root over a receipt's artifacts (high).** `receipt.v2`
carries `dataset_hash`/`params_hash` (digests of identity JSON, not bytes)
and `code_files` (writer sources only). There is no single root committing
*all* evidence artifacts (parquets, figures, MD files, split manifests) with
per-artifact `{path, size, sha256}` rows and an order-independent Merkle root.
Artifact pointers in research notebooks are checked for immutability but
piecemeal; a receipt cannot prove "these exact 14 files" in one digest, and
inclusion proofs for a single artifact are impossible.

**W4 — Code-map coverage gaps (medium).** `code_files` hashes *writer* files
by **basename** (`path.name`): no repo-relative paths, no imported
dependencies beyond the writer, no `git_revision`-to-code guarantee when the
worktree is dirty (the v2 envelope records the revision but not
`git_worktree_sha256`, which only the notebook verifier requires). A modified
imported module leaves the receipt untouched.

**W5 — fx-1 training receipts under-seal (medium).**
`fx1.train.receipts.TrainingReceipt` binds config/corpus/split/eval-base
hashes + seed, but: (a) the receipt JSON itself has **no self-digest and no
signature** (unlike `receipt.v2`); (b) `env_fingerprint` hashes only the
*names* of env vars, not values or package versions; (c) no GPU/torch/cuDNN/
`CUBLAS_WORKSPACE_CONFIG` determinism fingerprint (§1.1) even though `seed`
is recorded — the seed is evidence without the determinism context that makes
it meaningful; (d) no weight/checkpoint hash at train time (only at serve
time via `release.manifest.json`).

**W6 — Wall-clock trust (low-medium).** Timestamps (`generated_at`,
`created_utc`, ledger `utc`) are producer-asserted. Only the Sigstore path
gives externally-checkable time (Rekor). Ed25519 STHs are self-issued.

**W7 — Duplicate source tree (hygiene, medium).** A stray `src/src/quant_fund`
mirror (~426 files, including `research/verify.py` and
`research/receipt_v2.py`) exists in the workspace. Whichever copy a tool
imports decides what "the verifier" is; code-hash receipts must bind to the
executed file, and a mirror tree invites exactly the drift W4 describes. It
should be deleted or provenance-explained (not done here — out of lane scope).

**W8 — Hash-algorithm agility (low).** All digests are bare 64-hex SHA-256
with no multihash algorithm tag; a future BLAKE3 fast path (§1.2) would
require schema surgery. Also `hash_file`'s 1 MiB chunked reads measure
~274–323 MB/s streaming on this workstation's Xeon Silver 4214 (in-memory
SHA-256 ~362 MB/s); BLAKE3 is not installed in the locked env. Not a problem
today; relevant if corpora grow to multi-GB per receipt.

**W9 — Canonicalization is repo-defined, not RFC 8785 (low).**
`canonical_json_bytes` is a *good* deterministic serializer (sorted keys,
compact, `allow_nan=False`) but it is a house convention: cross-language
verifiers (a Go or JS auditor, a Sigstore policy) cannot recompute seals
without reimplementing it. RFC 8785 (JCS) is the interop standard; the
differences are small (number serialization per ECMAScript, I-JSON
restrictions) but real for floats.

---

## Part 3 — Adoption plan

### 3.1 `receipt.v3` envelope (schema upgrade)

Extend `receipt.v2` (keep validating v2; add v3 alongside, as the repo did
v1→v2):

```jsonc
{
  "schema": "receipt.v3",
  "schema_version": 3,
  "kind": "...", "data_label": "...", "verdict": "pass|fail|blocked",
  "generated_at": "...",                       // + optional Rekor timestamp
  "git": {
    "revision": "...",
    "worktree_sha256": "...",                  // always, not just notebooks (fixes W4)
    "dirty": false
  },
  "environment": { /* receipt.v2 block, unchanged */ },
  "determinism": {                             // new; OpenLineage mlDeterminism-shaped
    "seed": 7,
    "torch": "...", "cuda": "...", "cudnn": "...",
    "use_deterministic_algorithms": false,
    "cublas_workspace_config": null,
    "threads": {"omp": 1, "openblas": 1, "mkl": 1},
    "bitwise_claim": false                     // true only with RepDL/TBIK-class kernels
  },
  "uv_lock_sha256": "...",                     // env-lock commitment (§1.5)
  "sbom_sha256": null,                         // CycloneDX digest when built in CI
  "artifacts_root": {                          // new; fixes W3
    "algorithm": "merkle-rfc6962-sorted-v1",   // explicit tree spec + multihash-ready (W8)
    "root_sha256": "...",
    "artifacts": [
      {"path": "data/...", "role": "dataset|figure|metrics|checkpoint|notebook",
       "size": 123, "sha256": "..."}
    ]
  },
  "code_files": {"quant_fund/research/fleet_eval.py": "..."},  // repo-relative paths (W4)
  "dataset_hash": "...", "params_hash": "...",
  "live_pnl_claim": false,
  "payload": { /* lane body */ },
  "receipt_sha256": "...",                     // seal (JCS-canonical bytes; W9)
  "signature": {                               // new; fixes W2
    "scheme": "ed25519|sigstore",
    "value": "...", "public_key_hex": "...",
    "ledger_index": 1234,                      // entry in audit ledger
    "inclusion_proof": ["..."]                 // RFC 6962 audit path
  }
}
```

Rules: `artifacts_root.root_sha256` = **sorted-leaf Merkle root** reusing
`proofcore.contracts.merkle_root_hex` semantics (`PC:leaf:`/`PC:node:` domain
separation) over `sha256("artifact:" + path + ":" + size + ":" + digest)` —
so the root binds names, sizes, and bytes, and per-artifact inclusion proofs
fall out for free. `verify-receipt` re-hashes every listed artifact when
present on disk (fail-closed when missing, warn-mode for remote-only stores).
Seals switch to RFC 8785 canonical bytes with the house serializer kept as a
verified-compatible fallback (its output already differs from JCS only in
number formatting; add a property test pinning equivalence for the value
domains receipts actually contain).

### 3.2 Sign-and-log every receipt (tamper detection)

1. On write: seal → append canonical receipt bytes to the existing
   `quant_fund.audit` RFC 6962 ledger → embed `ledger_index` + inclusion
   proof → sign with the existing `Ed25519Signer` (offline key) or
   `SigstoreSigner` (CI, OIDC) → write the signed receipt to `receipts/`.
   All machinery exists; this is wiring, not new crypto.
2. Periodically (CI cron): re-issue a **signed tree head** over the ledger
   and, when in CI with an OIDC token, publish it via Sigstore so Rekor
   provides external time + non-repudiation (fixes W6 progressively).
3. `verify-research` / `verify-receipt` gain a `--require-signature` flag;
   the CI gate uses it for anything committed after the cutover date.

### 3.3 Legacy migration & the receipts gate (fixes W1)

- Add a CI job: `for f in receipts/*.json: verify-receipt` — red build on any
  failure.
- The 7 unsealed legacy receipts: wrap each verbatim as the `payload` of a
  sealed `receipt.v3` envelope with `kind: "legacy_attestation"`,
  `verdict: "blocked"` or the recorded verdict, and an explicit
  `"re_sealed": true` provenance note (the re-seal attests *the file as
  committed at migration time*, not the original run — honest about what the
  hash proves). Never edit the originals' bytes; store
  `legacy_sha256` of the original file inside the envelope.
- Gate policy afterwards: no PR lands a file in `receipts/` that does not
  verify + carry a signature.

### 3.4 fx-1 lane upgrades (fixes W5)

- `TrainingReceipt` → gain `receipt_sha256` self-seal, package versions
  (torch/transformers/CUDA), the `determinism` block from §3.1, and
  `checkpoint_sha256` at train time so training receipt → release manifest →
  serving attestation forms one unbroken hash chain.
- Keep `release.sig` HMAC as tier-1 but shape it for the Sigstore swap the
  module already anticipates (detached signature over a digest manifest —
  cosign-compatible; CI releases already have Sigstore identity).
- Corpus ledger entries: add the ledger's current root to training receipts,
  so a checkpoint commits to the exact corpus-chain state it trained on.

### 3.5 Provenance metadata (standards alignment)

- Emit an **OpenLineage RunEvent** (JSON, written beside the receipt) from
  receipt writers: job = lane script, run = `run_id`, inputs/outputs =
  artifacts with `sourceVersion` = sha256, facets = `mlTrainingEnvironment` +
  `mlDeterminism` (the §3.1 block) — near-zero-cost since every field already
  exists; gives external tooling a standard view.
- Document the **W3C PROV mapping** in the schema file: receipt = Activity,
  artifacts = Entities (`wasGeneratedBy`), `git.revision` + signer identity =
  Agents (`wasAssociatedWith`).
- For published checkpoints/wheels: the SLSA v1.1 attestation from
  `release.yml` should cite `receipt_sha256` as a build parameter, chaining
  supply-chain provenance to research evidence.

### 3.6 Determinism hardening (research side)

- Add `dipcatcher determinism-report`: prints the §3.1 `determinism` block
  for the current process (torch flags, `CUBLAS_WORKSPACE_CONFIG`,
  threadpoolctl) — receipt writers call it instead of hand-assembling.
- When GPU training/eval enters scope: set
  `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  in the train entrypoint, record failures as `verdict: "blocked"` receipts
  (an op without a deterministic implementation is itself evidence).
- Only set `bitwise_claim: true` with RepDL/TBIK-class order-invariant
  kernels; until then receipts honestly claim *seeded + flags-on*
  determinism, which the environment fingerprint lets a third party test.

### 3.7 Performance & agility (W8)

- Keep SHA-256 as the seal/interop digest. If corpus hashing becomes a
  bottleneck (current streaming rate ~0.3 GB/s/core here), add BLAKE3 as an
  *additional* per-artifact field (`blake3: "..."`) in the object store
  metadata only — multihash-style `algorithm` tags in `artifacts_root` make
  this additive, never breaking.
- Delete or explain the `src/src` mirror (W7) before code-path hashing can be
  trusted end-to-end.

### 3.8 Roadmap

| Phase | Work | Gate added |
|---|---|---|
| 1 (days) | CI `verify-receipt` over `receipts/*.json`; fix/quarantine the 7 failures via `legacy_attestation` re-seals | receipts-gate job |
| 2 (1–2 wks) | `receipt.v3` schema + `artifacts_root` (sorted Merkle) + repo-relative `code_files` + `worktree_sha256` + JCS seals; writers migrate lane-by-lane | schema drift test (mirrors `test_receipt_v2`) |
| 3 (1–2 wks) | Sign-and-log: receipts into the RFC 6962 audit ledger with inclusion proofs; `--require-signature` in CI for new receipts | signed-STH cron |
| 4 | fx-1 training-receipt upgrades + corpus-root binding; OpenLineage sidecar emission; SLSA attestation cites receipt hash | fx1-gate extension |
| 5 (opportunistic) | Sigstore STH publication; BLAKE3 fast path if/when hashing throughput binds | — |

Every phase is additive: v2 receipts stay verifiable, no existing digest
moves, and the honesty contract ("reproducible from a receipt hash") becomes
*enforced* rather than *aspired* at Phase 1.

---

## Part 4 — References

**Determinism.**
1. Chen, Zhang, Xie. *RepDL: Bit-level Reproducible Deep Learning Training and Inference.* Microsoft Research / arXiv:2510.09180, Oct 2025.
2. *TBIK: Deterministic Inference across Tensor Parallel Sizes That Eliminates Training–Inference Mismatch.* arXiv:2511.17826, Nov 2025.
3. PyTorch docs: *Reproducibility* — `torch.use_deterministic_algorithms`, cuDNN flags; NVIDIA cuBLAS docs: `CUBLAS_WORKSPACE_CONFIG` determinism requirement (CUDA ≥ 10.2).
4. Pineau et al. *Improving Reproducibility in Machine Learning Research.* JMLR 22, 2021 (ML Reproducibility Checklist).
5. ACM *Artifact Review and Badging v1.1* (2022) — artifacts evaluated / results reproduced / results replicated badges.

**Hashing & Merkle structures.**
6. O'Connor, Wong. *BLAKE3: a fast, parallel, verifiable hash function.* 2020 (blake3.io); Bao tree hashing, J-P Aumasson.
7. Laurie, Langley, Kasper. *RFC 6962 — Certificate Transparency.* IETF, 2013 (MTH, inclusion/consistency proofs, `0x00`/`0x01` domain separation).
8. NIST FIPS 180-4 — Secure Hash Standard (SHA-256).
9. Trillian docs (Google) — production RFC 6962 log implementations; lakeFS/Graveler docs — prolly trees over object stores.

**Canonicalization & signing.**
10. Rundgren, Jordan, Erdtman. *RFC 8785 — JSON Canonicalization Scheme (JCS).* IETF, 2020.
11. in-toto Attestation Framework v1.0; *SLSA v1.1 — Provenance* (slsa.dev, 2023–2024); DSSE (Protocol Buffers Signing & Encryption).
12. Sigstore: Fulcio/Rekor keyless signing (sigstore.dev); *GitHub Artifact Attestations* (`actions/attest-build-provenance`).
13. C2PA Specification v2.x (2023–2025) — hard vs. soft bindings, claims/assertions.

**Provenance & lineage.**
14. Moreau et al. *W3C PROV-O: The PROV Ontology.* W3C Recommendation, 2013.
15. OpenLineage v1.x spec — RunEvent, facets incl. ML training/determinism facets (openlineage.io, 2022–2026).
16. MLflow docs — Model Registry, model signatures, artifact stores; content-addressed artifact commitments (`ar-io-mlflow` pattern).
17. CycloneDX 1.5/1.6 (OWASP/ECMA-424) — ML-BOM & SBOM.

**Data versioning & builds.**
18. DVC docs — `.dvc` pointer files, CAS cache, `dvc.lock` pipeline hashes; lakeFS docs — commit/branch semantics over objects.
19. uv docs (Astral) — universal `uv.lock`, `--frozen` / `--locked` semantics; PEP 751 (`pylock.toml`).
20. Reproducible Builds project — `SOURCE_DATE_EPOCH`, `strip-nondeterminism`, definition of a reproducible build.

**Repo-internal anchors.** `src/quant_fund/research/{verify,receipt_v2}.py`;
`src/quant_fund/audit/{ledger,merkle,signing,verify}.py`;
`src/quant_fund/proofcore/contracts.py`; `src/quant_fund/proof/{bundle,verify}.py`;
`src/quant_fund/pit/{manifest,vault}.py`;
`src/quant_fund/data/lakehouse/{store,lineage,receipts}.py`;
`src/fx1/train/receipts.py`; `src/fx1/data/ledger.py`;
`src/fx1/serve/{signing,attestation}.py`; `.github/workflows/{ci,release,fx1}.yml`.
