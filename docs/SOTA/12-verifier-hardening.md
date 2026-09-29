# SOTA 12 — Verifier System Hardening

Lane: **VERIFICATION SYSTEM HARDENING**. This note surveys state-of-the-art
practice (2013–2026) for tamper-evident logs, append-only ledgers, detached
signatures, witness/audit protocols, and artifact attestations; audits the
dipcatcher / fx-1 verification stack (`verifier/`, `receipts/`,
`src/quant_fund/audit/`, `src/quant_fund/proofcore/`, `src/quant_fund/proof/`,
`src/quant_fund/research/verify.py`, `src/quant_fund/research/receipt_v2.py`,
`src/fx1/data/ledger.py`, `src/fx1/serve/signing.py`, and the
`verify-research` / `verify-receipt` / `verify-ledger` / `audit-trace` /
`audit-record` CLIs) against that practice; and proposes a concrete adoption
plan.

It modifies no code. It is the design record for the next verifier generation.

The honesty contract this lane serves: **receipts are immutable evidence**
(`receipts/`, `verify-research`) and **every research claim should be
reproducible from a receipt hash** (`AGENTS.md`, honesty contract rule 4).
Nothing below proposes weakening `FORBIDDEN_RESEARCH_METRIC_KEYS` /
`FORBIDDEN_HEADLINE_TOKENS`, the fail-closed gates, SYNTHETIC labeling, or the
no-live-claim rule. Every adoption item only *adds* checks or shares existing
stronger checkers.

---

## 0. Verdict up front

The repo contains **five independent verification silos** that were built to
different standards, and they disagree with each other in ways that matter.

- **Strong — genuinely at the state of the art for a research codebase.**
  `src/quant_fund/audit/` is a real Certificate-Transparency-style log:
  RFC 6962 Merkle trees with correct `0x00`/`0x01` domain separation *in log
  order*, inclusion **and** consistency proofs, the iterative reference-client
  verification algorithm, length-checked audit paths, Ed25519 + Sigstore signed
  tree heads, append-only file handles with `O_APPEND` + `fsync`, cross-platform
  advisory locking, canonical-JSON byte-equality checks on re-read, and a
  witness interface (`--expect-size` / `--expect-root`) for truncation
  detection. `tests/unit/audit/test_ledger_tamper.py` has **29** distinct
  tamper scenarios including rekey-without-pinning, rechain-without-resign,
  and checkpoint-reorder. This is better than most production transparency-log
  clients.

- **Weak — the acceptance history itself is unanchored.** `verifier/vN/`
  (8 acceptance criteria documents) and `verifier/runs/` (15 run logs) are
  plain Markdown with **no hash, no chain, no Merkle root, and no signature**.
  The lab's own definition of "done" is the single most-rewarded artifact to
  forge and the only verification surface in the repo with *zero* integrity
  mechanism. `verifier/README.md` claims an "Append-only index"; append-only is
  a social convention here, not a cryptographic property.

- **Broken — the committed receipt gate is red, and nothing runs it.**
  Measured on the tracked tree: `verify-receipt` passes **3 / 10** root
  receipts; `verify-research` passes **0 / 10**; `make receipts-reverify`
  exits **1** with all ten failing. Worse, `receipts-reverify` appears in
  **no** workflow under `.github/workflows/` — CI's only receipt check is
  `dipcatcher verify-research` (default path) against a *freshly regenerated*
  `data/metadata/research/latest.json`, never against the committed `receipts/`
  directory. So the ten tracked evidence files that `AGENTS.md` calls
  "immutable evidence" have **no automated verification in CI at all**, and the
  one gate written to check them fails closed whenever anyone runs it. A red
  gate nobody runs is worse than no gate: it is a false record of assurance.

- **Silent divergence — two incompatible receipt-digest conventions.**
  `research.verify._receipt_digest` and `receipt_v2.seal_receipt` compute
  *different* hashes over the *same* sealed receipt (§2.4). The audit ledger
  commits the first; receipts advertise the second. `audit-trace` therefore
  cannot match a published seal to its own ledger entry for any
  `receipt.v2`-style receipt. This is the highest-severity correctness defect
  in the lane: it defeats honesty contract rule 4 without failing any test.

- **Wrong primitive — symmetric MACs sold as signatures.** PROOFCORE bundles
  (`proof/sign.py`) and fx-1 releases (`fx1/serve/signing.py`) are sealed with
  **HMAC-SHA256** from `PROOFCORE_SIGNING_KEY` / `FX1_SIGNING_KEY`. Anyone who
  can *verify* a bundle can *forge* one. These artifacts are distributed as
  evidence; they need asymmetric seals.

The single highest-value change is §5.2: **put `verifier/` and root
`receipts/` inside the existing audit ledger**, which is already
RFC 6962-correct, already signed, already witnessed, and already tested. The
lab built the hard part and then left its most important evidence outside it.

---

## 1. Current design summary and trust-model map

### 1.1 The five silos

| # | Silo | Entry point | Integrity primitive | Canonicalization | Signature | Order-sensitivity | External witness |
|---|---|---|---|---|---|---|---|
| 1 | **Research notebook verifier** | `research.verify.verify_research_artifact`, CLI `verify-research` | SHA-256 of whole notebook minus `artifacts.immutable_json_sha256` | `json.dumps(sort_keys, separators=(",",":"), allow_nan=True)` — **ASCII-escaped** | none | n/a (single doc) | none |
| 2 | **Unified receipt envelope** | `research.receipt_v2.verify_receipt_file`, CLI `verify-receipt` | self-referential `receipt_sha256` seal + `environment.fingerprint_sha256` + `code_sha256` | `utils.hashing.canonical_json_bytes` — **UTF-8, `ensure_ascii=False`, `allow_nan=False`**; plus a legacy `strict_json` fallback | none | n/a (single doc) | none |
| 3 | **Audit ledger** | `audit.ledger.AuditLedger`, CLI `audit-record` / `verify-ledger` / `audit-trace` | hash chain (`prev_hash`) **and** RFC 6962 Merkle root over ordered leaves | `audit.canonical.canonical_json_bytes` — **ASCII-escaped, `allow_nan=False`** | Ed25519 (mode-0600 hex key) or Sigstore keyless (Fulcio + Rekor) | **yes** — leaves stay in log order | `--expect-size` / `--expect-root` (manual witness); Sigstore Rekor (network) |
| 4 | **PROOFCORE bundles** | `proof.verify.verify_bundle`, `proofcore.ci` | `bundle_id` self-hash + `prev_bundle_hash` chain in `bundles.jsonl` + data-manifest Merkle root + sidecar re-hash + **metric recomputation at `rtol 1e-9` / `atol 1e-12`** | `proofcore.contracts.canonical_json_bytes` — **ASCII-escaped, `allow_nan=False`** | **HMAC-SHA256** (symmetric) or `none` | chain is ordered; Merkle leaves are **sorted** | `trusted_head` (optional, caller-supplied) |
| 5 | **fx-1 corpus ledger** | `fx1.data.ledger.CorpusLedger.verify_chain` | linear `prev_hash` chain over training examples + quality-gate exclusions | `json.dumps(sort_keys=True)` **inside a pydantic `model_dump_json` line** | none | yes (chain) | none |

Silos 3 and 4 are strong. Silos 1, 2, and 5 have no signature and no external
anchor. `verifier/` is silo **zero** — it has none of the above.

### 1.2 What each silo actually proves

The trust model has three separable questions, and the silos answer different
subsets:

1. **"Is this artifact self-consistent?"** — answered well by silos 2 and 4.
   `verify_bundle` is the strongest check in the repo: it recomputes headline
   numbers from raw trade-log bytes at strict float tolerances, so a receipt
   whose metrics were edited fails even with a valid hash. Nothing else in the
   repo re-derives *values*, only *digests*.
2. **"Has this artifact changed since it was produced?"** — answered only by
   silos 3 and 4, and only for artifacts they were explicitly told about.
3. **"Was this artifact ever *supposed* to exist, and when?"** — answered by
   nobody. There is no timestamp anchoring anywhere in the repo, so no
   artifact can prove it existed before some date, and no log can prove it
   was not rewritten wholesale after the fact.

### 1.3 The verifier acceptance history (`verifier/`)

`verifier/README.md` is a Markdown table of v1–v8 with a "Created (UTC)"
column and a test count. Each `verifier/vN/acceptance.md` is prose criteria
(e.g. v8: "the fx1 gate passes on a host with no plugin root", "88 manifests →
77 positive + 11 negative"). Each `verifier/runs/<UTC>_<vN>.md` is a hand-
written pass record with commands, counts, and a `Verdict: PASS against
verifier/vN/acceptance.md` line.

Concretely, from `verifier/runs/2026-09-25T12-43Z_v8.md`:

```
- pytest tests/fx1: 139 passed, 1 skipped (intentional host-gated skip) ...
Verdict: PASS against verifier/v8/acceptance.md (v1–v7 re-verified by the
same gate).
```

The claimed measurements (139 passed, 47 mypy files, 88→77+11) are **not
bound to anything**. There is no commit hash on the run, no digest of the test
report, no digest of the acceptance file the run claims to satisfy, and no
signature. Note also that the v8 runs disagree with each other:
`2026-09-25T18-20Z_v6.md` reports "91 passed" and "41 files" for v6 while
`verifier/README.md`'s v6 row reports 91 tests — consistent — but the v8 row
reports 139 tests while five separate v8 run files exist with overlapping and
unlinked content. **Append-only-ness is unenforced**, so a run file can be
edited, deleted, or back-dated with no detectable trace.

### 1.4 Root `receipts/`

Ten tracked JSON files, heterogeneous schemas: `fleet_eval.v1` (sealed,
self-consistent — verified: advertised seal `5ddf15b0…` recomputes exactly),
`fx1.dip_bench/v1`, `capacity_eval`, `rankic_eval`, plus four
`*_20260922.json` adaptive-mix / basis files, `dip_bench_crypto_1d_20260925`,
`fast_replay_p42_conformance_20260927`, and `incumbent_bench_qlib`.

Measured state:

| Check | Result | Reason |
|---|---|---|
| `verify-receipt` | **3 / 10 pass** | 7 files have no valid `receipt_sha256` → `receipt_sha256_missing_or_invalid`. `incumbent_bench_qlib.json` confirmed: errors = `["receipt_sha256_missing_or_invalid"]` |
| `verify-research` | **0 / 10 pass** | all ten fail `invalid_notebook_firm`, `invalid_notebook_product`, `invalid_notebook_claim`, `invalid_research_receipt_schema_version`, … — `verify-research` expects the full 23-family research notebook, not a lane receipt |
| `make receipts-reverify` | **exit 1, 10/10 fail** | `proofcore.ci.reverify_receipts` dispatches to `_cli_verifier`, which shells out to **`verify-research`** — the wrong verifier for this directory |
| CI coverage of `receipts/` | **none** | no workflow invokes `receipts-reverify`; `ci.yml` runs `dipcatcher verify-research` against a regenerated `latest.json` instead |

That last row is the load-bearing bug. `proofcore/ci.py:_cli_verifier` runs:

```python
proc = subprocess.run(
    [sys.executable, "-m", "quant_fund.cli.main", "verify-research", str(path)],
    capture_output=True, text=True,
)
return proc.returncode == 0
```

So the committed receipt gate measures lane receipts with the notebook
verifier and fails all of them. The `Makefile` comment — `receipts-reverify:
## Fail-closed audit; schema-specific committed receipt verifiers pending` —
acknowledges the gap. It is **not** wired into CI (no workflow references it;
only `docs/proofcore/{README,MIGRATION}.md` mention it), so it is a red manual
gate today, not an enforced one.

The test suite does not catch this — it *enshrines* it. Every green
`reverify_receipts` test injects a mock (`verify=_checksum_verifier`), and
`test_cli_verifier_respects_exit_status` monkeypatches `subprocess.run` and
then **asserts** `cmd[-2:] == ["verify-research", str(receipt)]`. The wrong
dispatch is therefore locked in as expected behaviour: a future fix that routes
lane receipts to their real verifier would fail this test until the assertion
is updated too (P1.1).

### 1.5 The one committed ledger

Exactly one `entries.jsonl` / `checkpoints.jsonl` pair is tracked:

```
research/reality/studies/reality-us-liquid-daily-2026-09-27/audit/entries.jsonl
research/reality/studies/reality-us-liquid-daily-2026-09-27/audit/checkpoints.jsonl
```

`verify-ledger` on it returns:

```json
{"checkpoints": 1, "errors": [], "fully_signed": true,
 "merkle_root": "aef713c0c2b5660670537523e22e22c54937f0f8f13af4f5518f8d73cb278e4f",
 "signed_tree_size": 1, "tree_size": 1, "trust_anchor": "bundled",
 "unsigned_suffix": 0, "valid": true}
```

Three observations. (a) **tree_size 1** — the machinery is exercised on a
single entry, so the Merkle/consistency paths that only activate at size ≥ 2
have no committed production evidence. (b) **`trust_anchor: bundled`** — the
Ed25519 public key is read *from the checkpoint itself* (`5cb6f138…`), so
without `--trust-pub` an attacker who rewrites the log re-keys it and
verification still passes. `test_rekey_is_valid_until_the_public_key_is_pinned`
documents this as intended, but nothing in CI or the Makefile pins a key.
(c) The entry's `receipt_sha256` is `52c50588cd3d…` while the sibling
`receipt.json` advertises `514ccac702d5…`. Both are correct — under different
conventions. That is §2.4.

### 1.6 Trust-model map

```
                    ┌──────────────────────────────────────────────┐
                    │  verifier/vN/acceptance.md  (8 files, prose) │  ← NO integrity
                    │  verifier/runs/*.md        (15 files, prose) │  ← NO integrity
                    └──────────────────────────────────────────────┘
                                        │ claims to verify
                                        ▼
   receipts/*.json ──────────► verify-research (notebook verifier)   0/10 pass
        (10 files)      └────► verify-receipt  (envelope verifier)   3/10 pass
              │                 └────► proofcore.ci receipts-reverify  RED
              │
              │  audit-record (records _receipt_digest, NOT the seal)
              ▼
   research/reality/.../audit/entries.jsonl ──► hash chain + RFC 6962 Merkle
                                      │            + Ed25519/Sigstore STH
                                      │            + --expect-size/--expect-root witness
                                      ▼
                          verify-ledger  valid=true, trust_anchor=BUNDLED (unpinned)
                                      │
                                      ▼
                          audit-trace (published number → ledger index → inclusion proof)
                                ✗ cannot match a receipt.v2 seal (digest divergence)

   PROOFCORE bundles ──► HMAC-SHA256 (symmetric) + chain + manifest Merkle
                          (sorted leaves) + metric recomputation @1e-9
   fx-1 corpus ledger ─► prev_hash chain, no signature, no Merkle, no witness
   fx-1 releases ──────► HMAC-SHA256 manifest over artifact digests
```

The structural problem is visible in the diagram: **the top of the pyramid —
what "accepted" even means — is the only layer with no cryptographic floor.**

---

## 2. Threat model: what is detectable today vs. not

Adversary classes, weakest-first:

- **A0 — bit rot / accidental corruption.** Disk error, bad merge, partial
  write, editor reformatting JSON, CRLF normalization, non-UTF-8 bytes.
- **A1 — careless contributor.** Regenerates a receipt, "fixes" a number,
  rewrites a run log to match a new verdict, deletes an inconvenient
  acceptance criterion, reorders files.
- **A2 — motivated insider with write access to the working tree and git
  history.** Can rewrite commits, re-run and re-date verifier logs, regenerate
  ledgers from scratch with a fresh key.
- **A3 — holder of the symmetric MAC key** (`PROOFCORE_SIGNING_KEY` /
  `FX1_SIGNING_KEY`, both plain env vars). Can forge bundles indistinguishable
  from genuine ones.
- **A4 — split-view / rollback adversary.** Serves one tree to the lab and a
  different tree to an auditor.

### 2.1 Detectable today

| Attack | Detected by | Error surfaced |
|---|---|---|
| A0: any byte flip inside `entries.jsonl` | `read_entries` re-derives `entry_hash` from the canonical preimage | `entry_hash_mismatch:N` |
| A0: whitespace / key-order / Unicode-escaping change on a ledger line | `load_jsonl` re-encodes each parsed line and compares bytes | `noncanonical_line:N` |
| A0: missing trailing newline, blank line, CR, non-UTF-8 | `load_jsonl` byte inspection | `truncated_record`, `blank_line:N`, `carriage_return`, `malformed_line:N` |
| A1: edit a ledger payload without rehashing | chain check | `entry_hash_mismatch` |
| A1: rehash one entry but not its successor's `prev_hash` | chain check | `prev_hash_mismatch:N` |
| A1: delete a middle entry | index contiguity + chain | `index_gap:N` + `prev_hash_mismatch` |
| A1: delete the *last* entry and its checkpoint | **only with `--expect-size`/`--expect-root`** | `witness_size_mismatch` / `witness_root_mismatch` |
| A1: reorder or swap ledger lines | index + prev_hash | `index_gap`, `prev_hash_mismatch` |
| A1: duplicate the last line | index contiguity | `index_gap` |
| A1: rebuild the whole chain but keep old signatures | Merkle root recompute under each checkpoint | `checkpoint_root_mismatch:N` |
| A1: shrink `tree_size` in a later checkpoint | checkpoint ordering + consistency proof | `checkpoint_order:N`, `consistency_failed:N` |
| A1: flip a signature bit / change `merkle_root` in a checkpoint | Ed25519 verify over `signed_tree_head_bytes` | `signature_invalid:N` |
| A1: edit a `receipt.v2` payload | seal recompute + pydantic + env fingerprint + code-map digest | `receipt_sha256_mismatch`, `environment_fingerprint_mismatch`, `code_sha256_mismatch`, `receipt_v2_schema:*` |
| A1: edit a PROOFCORE bundle's metrics | **metric recomputation from trade-log bytes** at `rtol 1e-9`, `atol 1e-12` | recompute mismatch reason |
| A1: swap a PROOFCORE sidecar file | sidecar re-hash | `sidecar:<kind>:hash_mismatch` |
| A1: follow a symlink to smuggle a sidecar/ledger file | symlink refusal | `unsafe_symlink` |
| A1: put a forbidden headline metric into a `research_run` payload | `FORBIDDEN_RESEARCH_METRIC_KEYS` scan at append time | `AuditError` (refuses to append) |
| A1: forge a Sigstore checkpoint bundle | `sigstore_bundle_json` hash binding + bundle verification | `sigstore_bundle_hash_mismatch`, `signature_invalid` |
| A2: re-key an Ed25519 log **when `--trust-pub` is passed** | key comparison | `trust_anchor_mismatch:N` |
| A3 (partial): replay a bundle under a different chain head | `prev_bundle_hash` + optional `trusted_head` | `chain:prev_not_head`, `chain:trusted_head_mismatch` |

Also genuinely good: the ledger **refuses to append to a corrupt log**
(`refusing to append to a corrupt ledger`), so A1 cannot heal a detected
corruption by writing more entries on top; and `verify_ledger` runs an
**inclusion self-check on every leaf**, so a broken proof implementation fails
closed rather than silently passing (`inclusion_self_check_failed:N`).
`record_research_receipt` and `record_paper_directory` both re-read the source
file after appending and raise `receipt changed during recording` /
`orders parquet changed during recording` — a TOCTOU guard most systems lack.

### 2.2 NOT detectable today

| # | Attack | Why it succeeds | Severity |
|---|---|---|---|
| **N1** | **Rewrite any `verifier/runs/*.md`** — change "91 passed" to "139 passed", flip `Verdict: FAIL` to `PASS`, back-date the timestamp | No hash, no chain, no signature, no index commitment over the directory. Git history is itself rewritable by A2. | **Critical** |
| **N2** | **Rewrite or delete any `verifier/vN/acceptance.md`** — e.g. drop v8 criterion 3 ("fails closed without a claim") so a non-compliant run retroactively passes | Same. The *definition of acceptance* is the least-protected artifact in the repo. | **Critical** |
| **N3** | **Delete a root `receipts/*.json` file** | Nothing commits the *set* of receipts. No index, no manifest, no Merkle root over `receipts/`. A deleted receipt leaves no dangling reference anywhere. | **Critical** |
| **N4** | **Roll back the entire audit ledger to an earlier valid state** (truncate `entries.jsonl` *and* `checkpoints.jsonl` together, keeping a consistent older tree) | The chain and Merkle verify perfectly at the older size. Only `--expect-size`/`--expect-root` catches it, and nothing in CI, the Makefile, or `.github/workflows/` supplies those flags. `verify.py`'s own docstring says so: "It does not by itself prove that a previously seen tip was not deleted together with its signed tree head." | **Critical** |
| **N5** | **Re-key the ledger when no `--trust-pub` is passed** | `trust_anchor` reports `bundled`; the embedded public key is trusted because it is embedded. A2 regenerates `entries.jsonl` + `checkpoints.jsonl` under a fresh key and `verify-ledger` returns `valid: true`. This is the *default* invocation. | **High** |
| **N6** | **Forge any PROOFCORE bundle or fx-1 release** | HMAC-SHA256 is symmetric: the verifier holds the key. A3 — or anyone who reads `PROOFCORE_SIGNING_KEY` from CI secrets, a shell history, or a leaked `.env` — produces bit-identical-looking authentic bundles. There is no public verification key at all. | **High** |
| **N7** | **Reconcile a published `receipt_sha256` to its ledger entry** | `_receipt_digest` hashes the *full* document including the `receipt_sha256` field; `seal_receipt` hashes the document *excluding* it. Measured on `receipts/fleet_eval_5ddf15b0dc7d3ca1.json`: seal = `5ddf15b0…`, `_receipt_digest` = `5ef209f2…`. Confirmed again on the reality study: ledger entry = `52c50588…`, receipt seal = `514ccac7…`. `audit-trace` matches on the former, so **the trail from a published number to the ledger is broken for every sealed receipt.** | **Critical** |
| **N8** | **PROOFCORE data-manifest leaf-set malleability.** `merkle_root_hex` sorts leaves and duplicates the last node at odd levels. Appending a copy of the *maximum* leaf leaves the root unchanged — verified: `m[a,b,c] == m[a,b,c,max] == d44f177398ab19ee…`. A 4-read manifest can share a root with a 3-read manifest. | Sorting discards read order (which matters for point-in-time claims); duplicate-last padding is not tree-size-committed. | **High** |
| **N9** | **Reorder PROOFCORE reads** | Same cause as N8: `m[a,b,c] == m[c,b,a]` (verified). A PIT/leakage claim depends on read order; the manifest root does not commit to it. | **High** |
| **N10** | **fx-1 corpus ledger truncation / re-derivation.** `CorpusLedger.verify_chain()` returns a bare `bool`; `audit_export()` publishes `chain_head` but nothing pins it externally, and there is no Merkle tree, no signature, and no witness. Truncate the file at any entry and the remaining chain still verifies. | No external anchor for `chain_head`. | **High** |
| **N11** | **Back-date anything.** No RFC 3161 timestamp, no OpenTimestamps anchoring, no Sigstore Rekor inclusion for receipts. `generated_at` and `recorded_at` are self-asserted local clocks; `recorded_at` is even caller-overridable (`AuditLedger.append(..., recorded_at=...)`). Nothing proves an artifact existed before a date. | No trusted time source anywhere in the repo. | **High** |
| **N12** | **Split-view (A4).** Show the lab tree size 100 and an auditor tree size 40, both internally valid. | No quorum of independent cosigners; no monitor; no published checkpoint history. The witness interface exists but is a manual CLI flag. | **Medium** |
| **N13** | **Unseal a root receipt silently.** 7 of 10 committed receipts have no valid seal at all, and `verify-research` (the gate CI actually runs) rejects all 10 for unrelated structural reasons — so "the receipt gate is red" is the *normal* state and a genuinely tampered receipt hides inside it. | Gate is red in the committed state; no per-schema dispatch. | **High** |
| **N14** | **Substitute the verifier.** Nothing binds a `verifier/runs/*.md` PASS claim to the code that produced it. There is no `git_revision` on a run log, so A1 can record a PASS for v8 measured against a worktree where the v8 criteria were deleted. | No code identity on verifier runs. | **High** |
| **N15** | **Silent acceptance-criteria drift.** `verifier/README.md`'s table is hand-maintained and already contains a **duplicate v5 row** with different content ("uniqueness deep research" and "four uniqueness moves"). No check ties the table to the `vN/acceptance.md` files. | No index integrity. | **Medium** |
| **N16** | **Digest-convention laundering.** `_receipt_digest` uses `allow_nan=True` (NaN/Infinity serialize as the non-JSON tokens `NaN`, `Infinity`), while `audit.canonical` and `receipt_v2` both use `allow_nan=False`. A receipt containing a non-finite float hashes fine under convention 1 and raises under conventions 2 and 3 — so the same document is valid in one silo and invalid in another. | Three canonicalization functions with three different float/Unicode policies. | **Medium** |

### 2.3 Coverage summary by adversary

| Adversary | Silo 1 research | Silo 2 receipt.v2 | Silo 3 audit ledger | Silo 4 PROOFCORE | Silo 5 fx-1 corpus | `verifier/` |
|---|---|---|---|---|---|---|
| A0 bit rot | ✓ (digest) | ✓ (seal) | ✓✓ (chain+Merkle+byte-equality) | ✓✓ (self-hash+sidecars) | ✓ (chain) | ✗ **none** |
| A1 careless insider | ✓ | ✓ | ✓✓ | ✓✓✓ (recompute) | ✓ | ✗ **none** |
| A2 history rewriter | ✗ | ✗ | **partial** — needs `--expect-*` + `--trust-pub`, neither wired | ✗ (HMAC key holder = A3) | ✗ | ✗ **none** |
| A3 MAC-key holder | n/a | n/a | ✓ (Ed25519 is asymmetric) | ✗ **full forge** | n/a | ✗ |
| A4 split-view | ✗ | ✗ | ✗ (manual witness only) | ✗ | ✗ | ✗ |

The right-hand column is the lane's finding: **the acceptance history is
unprotected against every adversary class including A0.**

### 2.4 The digest divergence, stated precisely

Three canonicalization functions are live:

| Function | Used by | `sort_keys` | `separators` | `ensure_ascii` | `allow_nan` | Self-exclusion |
|---|---|---|---|---|---|---|
| `research.verify._receipt_digest` | `verify-research`, **and `audit.record.record_research_receipt` via `audit.trace.receipt_digest`** | yes | `(",",":")` | **True** (default) | **True** (default) | `artifacts.immutable_json_sha256` only |
| `utils.hashing.canonical_json_bytes` | `receipt_v2.seal_receipt` / `_seal_errors` / env fingerprint / code map | yes | `(",",":")` | **False** | **False** | `receipt_sha256` |
| `audit.canonical.canonical_json_bytes` | ledger entry preimages, signed tree heads | yes | `(",",":")` | **True** | **False** | n/a |

The fatal pair is rows 1 and 2. `_receipt_digest` normalizes with
`json.loads(json.dumps(payload))` and hashes **the whole document, seal
included**; `seal_receipt` pops `receipt_sha256` before hashing. So for any
receipt that carries a seal:

```
_receipt_digest(doc)  = SHA256(canonical(doc_with_seal))          → ledger payload
seal(doc)             = SHA256(canonical(doc_without_seal))       → advertised in the file
```

These can never be equal. Verified on two independent artifacts:

| Artifact | advertised `receipt_sha256` | `_receipt_digest` |
|---|---|---|
| `receipts/fleet_eval_5ddf15b0dc7d3ca1.json` | `5ddf15b0dc7d3ca1…` | `5ef209f2bccfb987…` |
| `research/reality/studies/…-2026-09-27/receipt.json` | `514ccac702d5fa65…` | `52c50588cd3dc0e1…` (this is what the ledger entry records) |

Consequence: `audit-trace --receipt <sealed receipt> --metric <path>` searches
ledger entries for `payload.receipt_sha256 == _receipt_digest(notebook)`. A
lane that seals with `receipt_v2` and records with `audit-record` stores
`_receipt_digest`, so the trace *does* link — but the number the receipt
publishes to the world (`514ccac7…`) is not the number the ledger commits to
(`52c50588…`). An external auditor holding the receipt and the ledger cannot
connect them without re-running the lab's private hash function. Honesty
contract rule 4 — "every research claim should be reproducible from a receipt
hash" — is not met for sealed receipts.

Additionally, because `_receipt_digest` leaves `allow_nan=True`, a receipt with
a NaN metric produces a digest over the literal token `NaN`, which is not valid
JSON. That digest is stable but the preimage is unparseable by any other tool
in the repo.

---

## 3. Technique summaries and citations

### 3.1 RFC 6962 — Certificate Transparency 1.0 (Merkle trees, inclusion, consistency)

Laurie, Langley, Kasper, *Certificate Transparency*, RFC 6962, June 2013.
<https://www.rfc-editor.org/rfc/rfc6962>

The construction the repo already implements correctly:

- **Domain separation.** `MTH({d}) = SHA-256(0x00 || d)` for leaves;
  `MTH(D[0:k] || D[k:n]) = SHA-256(0x01 || MTH(left) || MTH(right))` for
  internal nodes; `MTH({}) = SHA-256("")`. The prefixes make a leaf hash
  structurally incapable of colliding with an internal-node hash — the fix for
  the classic un-prefixed Merkle second-preimage attack.
- **Log order is preserved.** The tree is built over the leaf sequence *in
  insertion order*, with the RFC's `k` = largest power of two strictly less
  than `n`. This is what makes a **consistency proof** possible: order is part
  of the commitment.
- **Inclusion proof (§2.1.1).** An audit path of ≤ ⌈log₂ n⌉ hashes lets any
  third party recompute the root from one leaf, *without* the other leaves.
- **Consistency proof (§2.1.2).** Proves tree at size *m* is a prefix of tree
  at size *n > m*. This is the append-only property, made checkable in
  O(log n) by anyone who remembers only the old root and size.
- **Signed Tree Head (STH).** `(tree_size, timestamp, sha256_root_hash)`
  signed by the log's key. The STH is the *witness artifact*: it is small,
  publishable, and unforgeable.

**Why it matters here.** Consistency proofs are the answer to N4 (rollback). A
log truncated to size 40 produces a root for which no consistency proof to the
previously-published root at size 100 exists. The repo implements
`consistency_proof` and `verify_consistency` — including the subtle
`node % 2` walk, the equal-size-requires-empty-proof rule, and the
"consistency from an empty tree is not defined" rejection — and tests them
against random leaf sets (`test_random_inclusion_and_consistency`). **The
capability exists and is unused.**

### 3.2 RFC 9162 — Certificate Transparency 2.0

Laurie et al., *Certificate Transparency Version 2.0*, RFC 9162, October 2021
(Experimental; obsoletes RFC 6962). <https://www.rfc-editor.org/rfc/rfc9162>

Changes worth borrowing even though the repo need not adopt the whole protocol:

- **`TransItem`** — a single typed, extensible structure encapsulating all CT
  artifacts (SCTs, STHs, inclusion proofs, consistency proofs), replacing
  ad-hoc field bags. RFC 9162 §1.3 lists this explicitly as a major difference
  from 1.0. *Adoption value: one envelope type for "proof object" instead of
  the current five silos each inventing its own report dict.*
- **Unified verification algorithms** — RFC 9162 adds normative algorithms for
  verifying an inclusion proof, verifying consistency between two STHs, and
  verifying a root given the complete leaf list. The repo's
  `root_from_inclusion` / `_verify_consistency` already follow this shape and
  reject proofs that are too short *or* too long (`cursor != len(proof)` →
  `"inclusion proof is too long"`), which is exactly the strictness RFC 9162
  demands.
- **`get-all-by-hash`** — one API returning the inclusion proof *and* the
  consistency proof together, so a client can atomically establish "this leaf
  is in the log, and the log grew honestly." *Adoption value: the CLI should
  return both in one call rather than requiring separate invocations.*
- **Algorithm agility** via IANA registries, and log identity by OID rather
  than by key hash — relevant to N5, because key-hash identity is what makes
  re-keying invisible.
- **Honest limitation, quoted from §1:** *"The log auditing mechanisms
  described in this document can be circumvented by a misbehaving log that
  shows different, inconsistent views of itself to different clients.
  Therefore, it is necessary to treat each log as a trusted third party."*
  RFC 9162 says the quiet part: **transparency logs alone do not defeat A4.**
  That requires witnesses (§3.6).

### 3.3 Hash chains / append-only ledgers

The pattern behind silos 3, 4, and 5: each record commits to its predecessor's
hash (`prev_hash`), so rewriting record *i* invalidates every record after it.
This is the construction in Certificate Transparency, in **blockchain**
headers, in **git** commit ancestry, and in the **THEX** / Tiger tree lineage.

Properties and limits, stated carefully:

- A hash chain detects **modification** and **reordering**, and — with an
  index field — **insertion** and **interior deletion**.
- A hash chain does **not** detect **tail truncation**. Cutting the last *k*
  entries leaves a shorter but perfectly valid chain. Detecting that requires
  an external memory of the head: a signed checkpoint, a witness, or a
  published root. This is precisely N4 and N10.
- A chain does not detect **whole-log regeneration** by an adversary who can
  also produce fresh signatures over the new log (N5) unless the verification
  key is pinned out-of-band.
- Best practice is therefore **chain + Merkle + signature + external anchor**,
  layered. The repo has the first three for silo 3 and only the first for
  silo 5.

The repo's silo-3 implementation adds two details worth preserving as house
style: entries are hashed over a **canonical preimage** that *excludes* the
`entry_hash` field itself (avoiding self-reference), and the preimage is
re-serialized on read and compared **byte-for-byte** against the stored line,
so even semantically-invisible edits (whitespace, `\u` escaping, key order)
are caught. `test_unicode_rewritten_as_utf8_is_noncanonical` and
`test_noncanonical_whitespace` pin this.

### 3.4 Detached signatures: Ed25519, Minisign/Signify, Sigstore, age

**The core asymmetry point.** A MAC (HMAC) proves "someone with the key made
this." A signature proves "the holder of a *private* key made this," and
*anyone* can check it with the public key. Evidence artifacts that get
distributed must use signatures. This is N6.

- **Ed25519** — Bernstein, Duif, Lange, Schwabe, Yang, *High-speed
  high-security signatures*, J. Cryptogr. Eng. 2(2), 2012; RFC 8032.
  Deterministic (no RNG at signing time → no nonce-reuse catastrophe), 64-byte
  signatures over 32-byte keys, ~100k sign/s. Already implemented in
  `audit/signing.py` with correct hygiene: raw private key hex at mode `0600`,
  `os.chmod` re-asserted after write, `key_id = sha256(pubkey)[:16]` (never
  the key), constant-time verify via `Ed25519PublicKey.verify`, and
  `verify_ed25519` returns `False` rather than raising on mismatch.
- **Minisign / Signify** — `minisign` (Frank Denis) and OpenBSD `signify`:
  Ed25519 with a **pre-hashed** mode (`-H`, Blake2b-512 of the file) for large
  artifacts, a **trusted comment** bound into the signature (so the key ID and
  a human label cannot be swapped), and ASCII-armored output. Relevant pattern:
  *the comment is signed*, which is how `verify-ledger --trust-pub` pinning
  should be recorded — not as a loose sidecar. OpenBSD's release-signing model
  is the canonical example of a small pinned key set guarding large artifacts.
- **Sigstore** — Torres-Arias, Kuppusamy, Curtmola, Cappos, *Sigstore:
  Software Signing for Everybody*, ACM CCS 2022. <https://www.sigstore.dev>.
  Three pieces: **Fulcio** (short-lived X.509 certs bound to an OIDC identity),
  **Rekor** (an append-only transparency log of signature events — RFC 6962
  Merkle, signed inclusion promises), and the **bundle** format.
  `sigstore/protobuf-specs` defines `Bundle` with
  `media_type = application/vnd.dev.sigstore.bundle.v0.3+json`,
  `VerificationMaterial` carrying *either* a public-key identifier, an X.509
  chain, or a single leaf certificate — and v0.3 with the Public Good Instance
  and keyless signing **MUST** use the single-certificate form — plus
  `tlog_entries` (inclusion *proof*, not just a promise) and
  `TimestampVerificationData.rfc3161_timestamps`. The proto notes RFC 3161
  timestamps "can be used when the entry has not been stored on a transparency
  log, **or in conjunction for a stronger trust model**."
  **Keyless signing's real value is N11 + N12 at once:** Rekor inclusion is a
  *public, third-party-observable* timestamp, so a signature event cannot be
  back-dated or hidden.
  `audit/signing.py` already integrates this correctly and defensively:
  `sign_with_sigstore` refuses to emit a placeholder bundle when
  `DIPCATCHER_SIGSTORE_ID_TOKEN` or the `sigstore` package is missing
  (`SignatureUnavailableError`), the bundle JSON is bound into the checkpoint
  by `signature = sha256(bundle_json)`, and `verify_sigstore_bundle` records
  `sigstore_identity_unpinned` when no identity was pinned — i.e. it refuses
  to treat "some identity signed this" as trust.
- **age** — Filippo Valsorda's file-encryption tool. **Not** a signing format;
  it is X25519/ passphrase *encryption*. Listed here only to rule it out: the
  lane needs authentication and non-repudiation, not confidentiality. Receipts
  are meant to be public.

### 3.5 Artifact attestations: in-toto v1.0, DSSE, SLSA v1.0

**in-toto Attestation Framework v1.0** — *Statement* layer
(<https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md>):

```jsonc
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [ { "name": "<NAME>", "digest": {"<ALGORITHM>": "<HEX_VALUE>"} } ],
  "predicateType": "<URI>",
  "predicate": { ... }
}
```

The spec is explicit that every `subject` element **MUST** have `digest` set,
that subjects are **assumed immutable**, and that *"subject artifacts are
matched purely by digest, regardless of content type."* The layering is
Statement (what artifact) → Predicate (what is claimed about it) → Envelope
(DSSE: how it is signed). Parsing rules require consumers to ignore
unrecognized fields and treat unset/null/empty equivalently, with the major
version in the `predicateType` URI.

*Why this is the right shape for this repo.* The current receipts conflate
subject and predicate: `receipt.v2` mixes artifact identity (`dataset_hash`,
`code_files`, `code_sha256`), environment (`environment.fingerprint_sha256`),
verdict (`verdict`), lane body (`payload`), and honesty flags
(`live_pnl_claim`) in one flat object with a single self-referential seal.
Splitting **subject** (which files, by digest) from **predicate** (what the lab
claims) from **envelope** (the signature) is what makes N3 detectable — an
index over subjects is a list of digests, which is exactly a Merkle leaf set.

**DSSE** — *Dead Simple Signing Envelope*
(<https://github.com/secure-systems-lab/dsse>). Signs
`"DSSEv1" || SP || len(type) || SP || type || SP || body` rather than the raw
body, which prevents protocol-confusion attacks where the same key signs
different message types. The repo's Ed25519 checkpoints sign
`canonical_json_bytes(payload)` directly — safe today because the payload
schema is fixed and `_CHECKPOINT_KEYS` rejects unexpected fields, but DSSE
domain separation is the correct hardening if more message types ever share a
key.

**SLSA v1.0 Provenance** — <https://slsa.dev/spec/v1.0/provenance>,
`predicateType: "https://slsa.dev/provenance/v1"`. The model: a build runs on
a platform identified by `builder.id` (the transitive closure of everything
trusted to run the build and record provenance faithfully); the process is a
parameterized template identified by `buildType`; `externalParameters` are
**untrusted and MUST be included and verified downstream**; `internalParameters`
are platform-set and trusted; `resolvedDependencies` captures everything
fetched during init/execution; `subject` names the outputs. SLSA levels gate
how much of the build must be platform-controlled and auditable.

*Mapping to this repo:* a research receipt **is** a provenance attestation and
already carries most fields — `git_revision`, `git_worktree_sha256`,
`config_sha256`, `dataset_sha256`, `dataset_content_sha256`,
`northset_inputs_sha256`, `code_sha256`, `execution_claim`, `point_in_time`.
What it lacks is (a) a **`builder.id`** equivalent — who/what ran it, which
is what makes the honesty of `execution_claim` checkable rather than asserted;
(b) `buildType` — the parameterized template, so two receipts from different
pipelines are not confusable; and (c) a **verification policy** that decides
acceptance from provenance rather than from a hand-written `Verdict: PASS`.
`verifier/runs/*.md` is a *manual* provenance record with none of these.

Also relevant: **OpenSSF Model Signing** (`model-signing`, OMS v1.0) extends
this to ML artifacts — signing checkpoints with in-toto/DSSE plus Sigstore
Rekor inclusion. `fx1/serve/signing.py`'s `release.manifest.json` +
`release.sig` over per-artifact SHA-256 digests is a hand-rolled, symmetric
version of exactly this; the migration path is direct.

### 3.6 Witness and audit patterns: C2SP `tlog-checkpoint`, `tlog-witness`, `tlog-policy`; Trillian

**C2SP `tlog-checkpoint`** — <https://c2sp.org/tlog-checkpoint>. A checkpoint
is a **signed note** whose body is precisely three lines:

1. **origin** — a unique log identity, a schema-less URL with no spaces or `+`
   (e.g. `example.com/log42`); clients MUST NOT assume it is reachable.
2. **tree size** — ASCII decimal, no leading zeroes (`0` if empty).
3. **root hash** — base64 of the RFC 6962 Merkle root at that size.

Followed by optional non-empty extension lines (their use is **NOT
RECOMMENDED** — monitors cannot audit them), a blank line, and signature lines
of the form `— <key name> <base64 sig>`. Two normative rules the repo should
copy verbatim: *"Logs MUST not sign any checkpoint which is inconsistent with
any checkpoint it previously signed"* (inconsistent = no consistency proof
constructible), and *"clients MUST ignore unknown signatures"* — which is what
enables **key rotation and witness cosigning without a format change**.
Ed25519 is the SHOULD.

The repo's `checkpoints.jsonl` record carries the same three facts
(`tree_size`, `merkle_root`, `timestamp_utc`) plus key identity — but as a
private JSON schema, hex-encoded, with a bespoke key set, so no third-party
monitor or witness can read it. Adding an **origin line** is the single
cheapest interop win: it fixes log identity independent of key, which is
exactly what defeats N5 (re-keying) and N12 (split-view).

**C2SP `tlog-witness`** — <https://c2sp.org/tlog-witness>. A synchronous HTTP
protocol for **cosignatures**. `POST /add-checkpoint` body = an `old <size>`
line, zero-or-more base64 consistency-proof lines (client MUST NOT send more
than 63), a blank line, then a checkpoint. The witness:

- verifies the checkpoint signature against keys it trusts **for that origin**,
  ignoring unknown keys (404 unknown origin, 403 no signature verifies);
- requires `old size` ≤ checkpoint size (400 otherwise);
- requires `old size` == the size of the latest checkpoint it cosigned for that
  origin, else **409 Conflict** with body = its latest size and
  `Content-Type: text/x.tlog.size`;
- verifies the RFC 6962 §2.1.2 consistency proof from old size to new size
  (422 if it fails); on equal sizes, requires identical roots (409 otherwise);
- returns 200 with one or more `—` note-signature lines from its key(s);
- **MUST persist the new checkpoint before responding**, and MUST check
  old-size-then-persist **atomically** — the spec walks through the exact race
  where interleaved requests A(size N) and B(size N+K) roll the log back by K
  leaves;
- MAY log a request whose consistency proof failed *without* cosigning it, as
  evidence of log misbehaviour.

The payoff, quoted: cosignatures make it possible *"to produce self-contained
inclusion proofs that can be verified offline."* Each witness keeps only the
latest checkpoint per origin — so witnesses are cheap, and a **quorum of
independent witnesses** is what makes rollback and split-view detectable by
someone other than the log operator. The spec also flags the open problem:
*"it must not be possible to partition clients from monitors, either by
splitting the tree or by serving a stale view to monitors."*

**`tlog-policy` / quorum.** The generalization: require *k-of-n* cosignatures
from independently operated witnesses before a checkpoint is considered
committed. This converts A4 from "trust the lab" into "the lab cannot lie to
one auditor without lying to all of them."

**Trillian** — Google's open-source Merkle log implementation
(<https://github.com/google/trillian>), the production backend for Google's CT
logs. Design lessons rather than a dependency: separation of the **log signer**
from the **log server**; a storage layer that makes it impossible to serve an
unsigned or inconsistent tree head; and **personalities** that adapt the core
log to different leaf types. The repo's `AuditLedger` is a one-process
analogue; the lesson to take is that *signing must be separable from writing*,
so that a witness can cosign without write access.

**Monitor pattern.** Independently of witnesses: a monitor fetches every new
leaf (RFC 9162 `get-all-by-hash`), checks that leaves it cares about appear,
and remembers the latest root. In this repo, the analogue is a CI job that
fetches the committed ledger, verifies it against a **pinned** public key and
a **pinned** last-known `(size, root)`, and fails closed on either mismatch.
That single job closes N4 and N5 with code that already exists.

### 3.7 Trusted timestamps: RFC 3161 vs. OpenTimestamps

**RFC 3161** — Adams, Cain, Pinkas, Zuccherato, *Internet X.509 PKI Time-Stamp
Protocol (TSP)*, September 2001. <https://www.rfc-editor.org/rfc/rfc3161>. A
client sends a `TimeStampReq` containing the **message imprint** (a hash of the
data, never the data), the TSA returns a signed `TimeStampToken` (CMS/PKCS#7)
binding `(imprint, serialNumber, genTime)` under the TSA's certificate.
Properties: precise, immediately verifiable, cryptographically strong;
costs: **trusts the TSA** not to lie or to be coerced, requires network +
certificate validation, and TSA certs expire (long-term validation needs
archive timestamps / RFC 4998 evidence records). Sigstore bundles already
carry `rfc3161_timestamps` for exactly this reason.

**OpenTimestamps** — <https://opentimestamps.org>, Todd/Poelstra. The
trust-minimized alternative: a **calendar** aggregates many documents' hashes
into a Merkle tree, commits the tree root into a **public blockchain**
(Bitcoin, ~10 min finality), and issues a compact **detached `.ots` proof**
that later expands into the Merkle path from your document to the anchored
block. Verifying needs only a Bitcoin node (or a public block explorer), no
trusted party, no certificate. Costs: coarse time resolution (block height,
not seconds), a dependency on the calendar server for *issuance* (though not
for verification, once anchored), and asynchronous finality.

**Recommendation for this repo:** they are complements, not alternatives.

- Use **Rekor inclusion (Sigstore)** as the primary timestamp for anything
  already signed keylessly — it is a public transparency log *and* a timestamp,
  and the plumbing exists.
- Use **RFC 3161** where second-level precision matters and a TSA is
  acceptable (e.g. anchoring an acceptance-criteria version at the moment it
  is declared frozen).
- Use **OpenTimestamps** for the cheap, permissionless, permanent floor: anchor
  one weekly Merkle root of `verifier/` + `receipts/` + the audit ledger. No
  ongoing trust, no expiry, verifiable by an auditor years later with nothing
  but the repo and Bitcoin. This is the only mechanism in the survey that
  survives the lab ceasing to exist — which is what "immutable evidence"
  should mean.

---

## 4. Gap analysis: repo vs. SOTA

| Capability | SOTA reference | Repo status | Gap |
|---|---|---|---|
| RFC 6962 Merkle in log order | RFC 6962 §2.1 | ✅ `audit/merkle.py`, correct domain separation, `k`-split, iterative verify | — |
| Inclusion proofs | RFC 6962 §2.1.1 | ✅ implemented, length-checked, self-checked on every verify | not exposed as a CLI artifact |
| Consistency proofs | RFC 6962 §2.1.2 | ✅ implemented + tested | **never used outside `verify_ledger`'s internal checkpoint walk** |
| Signed tree heads | RFC 6962 §3.3 | ✅ Ed25519 + Sigstore | private JSON schema, not a signed note |
| Log origin / identity | C2SP `tlog-checkpoint` | ❌ none | no origin line; identity = key hash |
| Pinned trust anchor | OpenBSD signify; CT monitors | ⚠️ `--trust-pub` exists, unused | **default invocation trusts the embedded key** (N5) |
| Witness / cosignature | C2SP `tlog-witness` | ⚠️ `--expect-size`/`--expect-root` (manual, single-valued) | no persistence, no quorum, nothing wired into CI (N4, N12) |
| External timestamp | RFC 3161 / OTS / Rekor | ⚠️ Rekor only, and only when an OIDC token exists | no anchoring for `verifier/` or `receipts/` (N11) |
| Asymmetric artifact seal | Sigstore / minisign / OMS | ⚠️ Ed25519 for ledger checkpoints | **PROOFCORE + fx-1 releases are HMAC** (N6) |
| Subject/predicate separation | in-toto Statement v1.0 | ❌ flat receipt objects | no digest-indexed subject list (N3) |
| Provenance predicate | SLSA v1.0 | ⚠️ rich fields, no `builder.id`/`buildType` | provenance asserted, not verified (N14) |
| Order-committing artifact set root | RFC 6962 | ⚠️ silo 3 ✅; silo 4 ❌ | `proofcore.merkle_root_hex` sorts + duplicates (N8, N9) |
| Receipt index / set commitment | CT log over artifacts | ❌ **none** | deleting a receipt is invisible (N3) |
| Acceptance-criteria integrity | signed policy / frozen spec | ❌ **none** | N1, N2, N15 |
| Single canonicalization | one wire format | ❌ **three** | N7, N16 |
| Gate green in committed state | CI must be green | ❌ `receipts-reverify` red 10/10 | N13 |

---

## 5. Adoption plan

Ordered by value-per-risk. Every step is additive and fail-closed; none
weakens an existing check.

### 5.1 Phase 0 — stop the bleeding (no new cryptography)

**P0.1 Fix the committed gate (N13).** Give `proofcore.ci.receipts-reverify`
per-schema dispatch instead of always calling `verify-research`:

```python
# proofcore/ci.py — replace _cli_verifier
def _cli_verifier(path: Path) -> bool:
    """Dispatch on the receipt's own schema, fail closed on unknown."""
    payload = json.loads(path.read_text())
    schema = payload.get("schema") or payload.get("schema_version")
    if schema == "receipt.v2" or schema == 2:
        cmd = ["verify-receipt"]
    elif isinstance(schema, str) and schema.endswith(".v1"):
        cmd = ["verify-receipt"]          # sealed lane receipt
    elif "notebook_version" in payload:
        cmd = ["verify-research"]         # full 23-family notebook
    else:
        return False                      # unknown schema: fail closed
    return subprocess.run([sys.executable, "-m", "quant_fund.cli.main",
                           *cmd, str(path)], capture_output=True,
                          text=True).returncode == 0
```

Then **seal the seven unsealed receipts** with `receipt_v2.seal_receipt` (a
one-time regeneration is legitimate here: they have no seal, so there is no
prior commitment to contradict). Target: `make receipts-reverify` exits 0 with
10/10 passing. Until this is green, every other gate in the repo is
credibility-discounted.

**P0.2 Unify the receipt digest (N7, N16).** This is the load-bearing fix.
Two parts:

- *Add*, do not change: `audit.trace.receipt_digest` gains a convention
  parameter and records **both** digests in the ledger payload:

```python
# audit/trace.py
def receipt_digest(notebook, *, convention: str = "legacy_full_document") -> str:
    if convention == "seal_excluding_self":
        body = {k: v for k, v in notebook.items() if k != "receipt_sha256"}
        return hash_bytes(canonical_json_bytes(body))   # == receipt_v2 seal
    return _receipt_digest(notebook)                     # existing behaviour
```

- `audit.record.record_research_receipt` writes
  `payload["receipt_sha256"]` (legacy, unchanged — old entries keep verifying)
  **and** `payload["receipt_seal_sha256"]` (the published seal, when present).
  `trace_published_number` then matches on **either** field. Existing ledgers
  stay valid; new entries are reconcilable with the number the receipt
  publishes.

Concurrently, converge canonicalization: `_receipt_digest` should move to
`allow_nan=False` (a receipt with NaN in it is not valid JSON evidence) and
`ensure_ascii=False`, matching `utils.hashing.canonical_json_bytes`. Do this
under a `digest_convention` field recorded in the ledger so old entries remain
verifiable — the repo already models this pattern in
`ReceiptVerification.digest_convention` (`canonical_json` vs `strict_json`).

**P0.3 Pin the ledger key (N5).** Add `verify-ledger --trust-pub` to CI and to
a `make audit` target, with the public key committed as a tracked file
(e.g. `receipts/TRUST_ANCHOR.pub`) and its own SHA-256 recorded in the repo.
Re-keying then fails with `trust_anchor_mismatch` instead of passing silently.

### 5.2 Phase 1 — hash-chained receipt index (N1, N2, N3, N15)

This is the deliverable the lane asks for, and it is mostly *reuse*: silo 3
already does everything needed.

**P1.1 A tracked receipt index.** New append-only JSONL, one line per
committed artifact, in the ledger's own entry format so `read_entries`,
`verify_ledger`, inclusion/consistency proofs, and signing all apply unchanged.
Leaf preimage per artifact:

```jsonc
{
  "v": 1,
  "index": 42,
  "kind": "artifact_commitment",          // extend KINDS
  "recorded_at": "2026-09-28T12:00:00Z",
  "payload": {
    "path": "receipts/fleet_eval_5ddf15b0dc7d3ca1.json",
    "sha256": "<sha256 of file bytes>",           // byte identity
    "seal_sha256": "<advertised receipt_sha256>", // published identity
    "digest_convention": "canonical_json",
    "schema": "fleet_eval.v1",
    "git_revision": "3dafeb7…",
    "live_pnl_claim": false,
    "evidence_class": "SYNTHETIC"                 // or MARKET_HISTORICAL
  },
  "prev_hash": "…",
  "entry_hash": "…"
}
```

Committing **both** the file-byte SHA-256 and the published seal closes N3
(a deleted file leaves a dangling index entry with no matching bytes) *and*
N7 (the seal is now on the record). `path` makes reordering/renaming visible.
This is the in-toto **subject** list (§3.5) expressed in the repo's own
format: `subject.digest` = `sha256`, `subject.name` = `path`.

**P1.2 A verifier-history index.** Same mechanism for `verifier/`:

```jsonc
{"kind": "acceptance_commitment",
 "payload": {"path": "verifier/v8/acceptance.md", "sha256": "…",
             "criteria_count": 6, "git_revision": "…"}}
{"kind": "verifier_run_commitment",
 "payload": {"path": "verifier/runs/2026-09-25T12-43Z_v8.md", "sha256": "…",
             "acceptance_version": "v8",
             "acceptance_sha256": "<sha256 of verifier/v8/acceptance.md>",
             "verdict": "PASS",
             "git_revision": "<commit the run measured>",
             "git_worktree_sha256": "…",
             "tests_passed": 139, "tests_skipped": 1,
             "live_pnl_claim": false}}
```

`acceptance_sha256` on the run is what defeats **N14** and **N2**: a PASS claim
is bound to the exact acceptance text it satisfied, so deleting a criterion
afterwards breaks the binding. `git_revision` + `git_worktree_sha256` reuse
`utils.reproducibility` helpers the receipts already carry. This is the SLSA
provenance model (§3.5) applied to the lab's own gate: `builder.id` → the
runner/host, `buildType` → the acceptance version, `externalParameters` → the
commands run, `subject` → the files committed.

**P1.3 Merkle root per acceptance version.** For each `vN`, compute the
RFC 6962 root over the ordered set `{acceptance.md digest} ∪ {all run files
claiming vN, in filename order}` and record it as a signed checkpoint:

```jsonc
{"kind": "acceptance_root",
 "payload": {"acceptance_version": "v8",
             "tree_size": 6,
             "merkle_root": "<hex>",
             "leaves": ["<acceptance.md sha256>", "<run1 sha256>", …],
             "live_pnl_claim": false}}
```

Use `audit.merkle.merkle_root` — **log-ordered**, not `proofcore`'s sorted
variant — so the root commits to *which runs existed and in what order*.
Because it is a checkpoint, it gets an Ed25519/Sigstore signature for free.
`verifier/README.md`'s hand-maintained table (which already contains a
duplicate v5 row, N15) becomes **generated** from the index and its digest
committed, so drift is detectable.

**P1.4 New CLI surface.**

| Command | Purpose |
|---|---|
| `dipcatcher receipt-index build --root receipts --out receipts/index.jsonl` | (Re)derive the index from disk; refuses to rewrite an existing committed entry |
| `dipcatcher receipt-index verify --root receipts --index receipts/index.jsonl --trust-pub …` | every indexed path exists and hashes correctly; every file on disk is indexed; chain + Merkle + signatures verify; **fail closed** on any missing/extra file |
| `dipcatcher verifier-index build|verify --root verifier` | same for `verifier/vN/` + `verifier/runs/` |
| `dipcatcher verifier-trace --run verifier/runs/….md` | prints the run's inclusion proof under the acceptance root, its bound `acceptance_sha256`, and `git_revision` |
| `dipcatcher audit-prove --ledger <dir> --index <i> --out proof.json` | emits a **self-contained** proof object: leaf, index, tree size, inclusion path, signed checkpoint, consistency proof from the previous checkpoint |
| `dipcatcher audit-check --proof proof.json --trust-pub …` | verifies that proof **offline**, with no access to the log |

`audit-prove` / `audit-check` implement the `tlog-witness` payoff (§3.6) —
*"self-contained inclusion proofs that can be verified offline"* — and the
RFC 9162 `get-all-by-hash` pattern of returning inclusion + consistency
together.

**P1.5 A persistent witness file (N4, N12).** The manual
`--expect-size`/`--expect-root` flags become a tracked artifact:

```
receipts/witness.jsonl        # append-only, one line per observation
{"origin":"dipcatcher.local/receipts","tree_size":42,
 "merkle_root":"<base64>","observed_at":"2026-09-28T12:00:00Z",
 "signature":"— dipcatcher.local/receipts <base64>"}
```

with `origin` per C2SP `tlog-checkpoint`. CI reads the **last** line and
passes it as `--expect-size`/`--expect-root`; a rollback then fails with
`witness_size_mismatch`. Add `--witness-quorum k` for *k*-of-*n* cosignatures
once more than one key exists. `verify_ledger` needs no change — the flags
already exist and are already tested
(`test_remove_last_entry_and_checkpoint_needs_a_witness`).

### 5.3 Phase 2 — asymmetric seals and correct Merkle (N6, N8, N9)

**P2.1 Migrate PROOFCORE and fx-1 releases off HMAC.** The `Signer` protocol in
`proof/sign.py` was explicitly designed for this ("*the protocol admits ed25519
later without a schema change*"). Add:

```python
class Ed25519Signer:                     # reuse audit.signing.Ed25519Signer
    scheme = "ed25519"
```

and extend `SignatureBlock.scheme` to
`Literal["hmac-sha256", "ed25519", "sigstore", "none"]`. Keep accepting
`hmac-sha256` for existing bundles (verify-only) while new bundles sign
asymmetrically; add a `--require-asymmetric` flag that rejects MAC-sealed
bundles so the gate can be flipped later. `fx1/serve/signing.py`'s
`release.manifest.json` + `release.sig` is the same change, and is the natural
place to adopt **OpenSSF Model Signing** / in-toto + DSSE for parity with the
ecosystem (§3.5).

Also adopt **DSSE-style domain separation** for the signed preimage
(`"DIP v1" || SP || type || SP || body`) so a ledger checkpoint signature can
never be replayed as a bundle signature under a shared key.

**P2.2 Fix `proofcore.merkle_root_hex` (N8, N9).** Two defects, one cause:
it is designed as an *order-independent set commitment*, but the data-manifest
root is used as an *ordered log* commitment (reads happen in sequence, and
point-in-time claims depend on that order). Fixes, additive:

- Commit to the **leaf count**: hash `b"PC:tree:" + str(n).encode()` into the
  root, or prefix each leaf with its index (`b"PC:leaf:" + i.to_bytes(8,"big")
  + digest`). Either kills the duplicate-of-max-leaf equivalence
  (`m[a,b,c] == m[a,b,c,max]`, verified `d44f177398ab19ee…`).
- Offer `merkle_root_hex(leaf_hashes, *, ordered: bool)`: `ordered=True` uses
  the RFC 6962 construction from `audit.merkle` (no sorting), which also kills
  N9 (`m[a,b,c] == m[c,b,a]`, verified). Default new bundles to
  `ordered=True`; record the mode in the manifest so old bundles still verify.
- Add **inclusion proofs per read**, so a single dataset read can be proven
  against the manifest root without exposing the other reads — the property
  that makes the corpus auditable without being downloadable
  (`fx1/data/ledger.py`'s stated goal).

**P2.3 Give the fx-1 corpus ledger a signed head (N10).** `CorpusLedger`
already computes `chain_head` in `audit_export()`. Add periodic signed
checkpoints (reuse `audit.signing`) and emit `chain_head` into the receipt
index, so truncation is detectable without external state. Long-term, route
corpus entries through `AuditLedger` and retire the parallel chain — two
append-only log implementations is one too many.

### 5.4 Phase 3 — timestamps, witnesses, attestations (N11, N12)

**P3.1 Anchor weekly.** Compute the Merkle root over
`{receipt index root} ∪ {verifier index root} ∪ {audit ledger root}` and:

1. sign it as a checkpoint (Ed25519 always; Sigstore when a token exists —
   Rekor inclusion is then a public timestamp, §3.7);
2. optionally request an **RFC 3161** token for second-level precision;
3. create an **OpenTimestamps** `.ots` detached proof and commit it.

Step 3 is the permanent floor: verifiable years later with only the repo and
Bitcoin, no TSA, no OIDC provider, no lab. Keep `.ots` files small and
committed beside `receipts/witness.jsonl`.

**P3.2 Adopt the signed-note checkpoint format.** Emit, alongside (not instead
of) `checkpoints.jsonl`, a C2SP `tlog-checkpoint` signed note:

```
dipcatcher.local/receipts
42
qvcawN9VJFQXK4VqZ6hY7sQ3vLk7dM1eF0gH2iJ3kL4=

— dipcatcher.local/receipts Az3grlgtzPICa5OS8npVmf1Myq/5IZniMp+ZJurmRDeOoRDe4URYN7u5/Zhcyv2q1gGzGku9nTo+zyWE+xeMcTOAYQ8=
```

Three lines + signatures, base64 root, decimal size. This costs ~20 lines of
code and buys: a stable **origin** (log identity independent of key → defeats
N5), interoperability with any `tlog-witness`-compliant witness (→ N12), and
free key rotation and cosigning because *"clients MUST ignore unknown
signatures."*

**P3.3 Implement a minimal `tlog-witness` client + local witness.**
`POST /add-checkpoint` with `old <size>` + consistency proof lines (≤ 63) +
checkpoint; enforce the spec's error codes (404 unknown origin, 403 no
signature verifies, 400 bad old size, **409 Conflict** when old size ≠ the
witness's latest, 422 bad consistency proof); **persist before responding**
and make check-then-persist atomic (the spec's A/B race). Run the witness as a
separate process with no write access to the log — Trillian's
signer/server separation lesson (§3.6). Start with one witness key in CI;
extend to *k*-of-*n* via `--witness-quorum`.

**P3.4 Emit in-toto Statements for receipts.** Wrap each receipt as

```jsonc
{"_type": "https://in-toto.io/Statement/v1",
 "subject": [{"name": "receipts/fleet_eval_….json",
              "digest": {"sha256": "<file sha256>"}}],
 "predicateType": "https://dipcatcher.local/research-receipt/v1",
 "predicate": { /* the existing receipt body, verbatim */ }}
```

signed under DSSE. Because *"subject artifacts are matched purely by digest"*,
this makes the receipt index a first-class, tool-readable subject list and
lets third-party in-toto/SLSA policy engines evaluate the lab's evidence.
The honesty flags (`live_pnl_claim: false`, `evidence_class`, SYNTHETIC
labels) stay exactly where they are — moving them would weaken the contract.

**P3.5 SLSA-shaped verifier provenance.** Add `builder.id` (runner/host
identity), `buildType` (`verifier/vN`), `externalParameters` (commands run),
`resolvedDependencies` (`uv.lock` digest), and `subject` (files produced) to
each `verifier_run_commitment` entry. Then acceptance becomes a **policy
evaluation over provenance** rather than a hand-typed `Verdict: PASS` — the
difference between "the lab says it passed" and "the recorded build satisfies
the recorded criteria."

### 5.5 Test suite extensions

Existing coverage to preserve: `tests/unit/audit/test_ledger_tamper.py`
(29 tamper scenarios), `test_merkle.py` (5, incl. RFC 6962 proof-length and
random inclusion/consistency), `test_signing.py`, `test_trace.py`,
`test_record_paper.py`, `tests/property/test_adversarial_receipts.py`,
`tests/regression/test_phase1_sealed_receipts.py`.

New tests, grouped:

**Digest unification (§5.1 P0.2)**
- `test_seal_and_trace_digest_agree_on_sealed_receipt` — for a `receipt.v2`
  document, `receipt_digest(doc, convention="seal_excluding_self")` **equals**
  the advertised `receipt_sha256`. This is the regression test for N7; it
  fails today.
- `test_legacy_ledger_entries_still_verify_after_convention_split` — an entry
  written with only `receipt_sha256` still traces.
- `test_nonfinite_float_is_rejected_under_canonical_convention` — `NaN` in a
  receipt raises rather than producing a digest over the token `NaN` (N16).
- `test_digest_convention_field_is_recorded_and_reported` —
  `digest_convention` appears in the ledger payload and in the trace result.

**Receipt index (§5.2)**
- `test_index_detects_deleted_receipt` (N3) — remove a file, `receipt-index
  verify` fails closed with `indexed_file_missing`.
- `test_index_detects_unindexed_receipt` — add a file, fails with
  `unindexed_file`.
- `test_index_detects_byte_edit_of_indexed_receipt` — fails with
  `sha256_mismatch`.
- `test_index_detects_rename` — fails on path/sha256 disagreement.
- `test_index_detects_reorder` — log-ordered Merkle root changes.
- `test_index_refuses_to_rewrite_committed_entry` — rebuilding is append-only.
- `test_index_roundtrips_through_offline_proof` — `audit-prove` → delete the
  index → `audit-check --proof` still verifies (self-contained proof).

**Verifier history (§5.2)**
- `test_verifier_index_detects_edited_run_log` (N1) — flip `FAIL`→`PASS`,
  sha256 mismatch.
- `test_verifier_index_detects_deleted_acceptance_criterion` (N2) — edit
  `vN/acceptance.md`, all runs bound to its old `acceptance_sha256` fail.
- `test_verifier_index_detects_backdated_run` — `recorded_at` older than the
  bound witness observation; fails against the anchor.
- `test_run_bound_to_wrong_git_revision_fails` (N14).
- `test_acceptance_root_is_log_ordered` — permuting run files changes the
  root (proves `audit.merkle` is used, not `proofcore`'s sorted variant).
- `test_readme_table_drift_is_detected` (N15) — regenerate the table, compare
  digest; catches the existing duplicate-v5-row condition.
- `test_acceptance_root_matches_signed_checkpoint` — root under a checkpoint
  signature verifies; an edited root fails `checkpoint_root_mismatch`.

**Witness and rollback (§5.2 P1.5, §5.4 P3.3)**
- `test_rollback_with_matching_truncation_detected_by_witness` (N4) —
  truncate entries **and** checkpoints to a consistent older tree; without
  `--expect-*` it passes (documenting the limitation), with the witness file
  it fails `witness_size_mismatch`.
- `test_rekey_detected_when_key_pinned` (N5) — extends the existing
  `test_rekey_is_valid_until_the_public_key_is_pinned` to assert CI *does*
  pin.
- `test_witness_quorum_requires_k_of_n` — *k*-1 cosignatures fail, *k* pass.
- `test_witness_409_on_stale_old_size` / `test_witness_422_on_bad_consistency`
  / `test_witness_404_unknown_origin` / `test_witness_403_no_valid_signature`
  — the C2SP error contract.
- `test_witness_persists_before_responding` — the atomicity race from
  `tlog-witness`: interleave size *N* and *N+K*, assert no rollback.
- `test_unknown_signature_lines_are_ignored` — extra `—` lines from an
  untrusted key neither validate nor invalidate the checkpoint.

**PROOFCORE Merkle and seals (§5.3)**
- `test_proofcore_root_rejects_duplicate_of_max_leaf` (N8) — assert
  `merkle_root_hex([a,b,c]) != merkle_root_hex([a,b,c,max])` after the fix.
  Today this **fails** (`d44f177398ab19ee…` for both).
- `test_proofcore_ordered_root_rejects_permutation` (N9) — assert
  `root([a,b,c]) != root([c,b,a])` under `ordered=True`. Today this fails.
- `test_proofcore_root_commits_leaf_count` — `root([a]) != root([a,a])`
  (already true; pin it) and the new count commitment.
- `test_per_read_inclusion_proof_verifies_against_manifest_root`.
- `test_ed25519_sealed_bundle_verifies_without_secret` (N6) — verify with only
  a public key; assert no env secret is read.
- `test_hmac_bundle_rejected_under_require_asymmetric`.
- `test_signature_cannot_be_replayed_across_message_types` — DSSE domain
  separation.

**Timestamps (§5.4)**
- `test_opentimestamps_proof_verifies_offline_against_mock_chain` — anchor,
  then verify the `.ots` path with a stubbed block header.
- `test_rfc3161_token_binds_message_imprint` — imprint must equal the artifact
  digest; a token for a different imprint is rejected.
- `test_rekor_inclusion_is_recorded_when_token_present` and
  `test_missing_token_never_emits_placeholder_bundle` (already the behaviour in
  `sign_with_sigstore`; pin it).

**Gate hygiene**
- `test_receipts_reverify_is_green_on_committed_tree` — the meta-test: the CI
  gate must pass as committed (N13). This is the test that would have caught
  the current red state.
- `test_every_committed_receipt_has_a_seal` — 10/10.
- `test_every_sealed_receipt_is_in_the_index` and
  `test_every_indexed_receipt_is_in_a_ledger`.

### 5.6 Sequencing and acceptance

| Phase | Contents | Closes | Depends on |
|---|---|---|---|
| **0** | P0.1 gate dispatch + reseal 7 receipts; P0.2 digest unification; P0.3 pin key | N7, N13, N16, N5 | — |
| **1** | P1.1 receipt index; P1.2 verifier index; P1.3 acceptance Merkle roots; P1.4 CLI; P1.5 witness file | N1, N2, N3, N4, N14, N15 | Phase 0 (needs one digest convention) |
| **2** | P2.1 Ed25519/DSSE seals; P2.2 proofcore Merkle fix; P2.3 corpus ledger head | N6, N8, N9, N10 | independent of Phase 1 |
| **3** | P3.1 anchoring; P3.2 signed-note checkpoints; P3.3 witness protocol; P3.4 in-toto; P3.5 SLSA provenance | N11, N12, plus interop | Phases 1–2 |

**Phase 1 acceptance criteria** (proposed `verifier/v9`), all fail-closed:

1. `make receipts-reverify` exits 0 on the committed tree, 10/10 receipts.
2. Every committed receipt carries a `receipt_sha256` that recomputes under a
   single named convention, and that convention is recorded on the artifact.
3. `receipts/index.jsonl` exists, is hash-chained, is signed, and
   `receipt-index verify --trust-pub` returns `valid: true`,
   `fully_signed: true`.
4. Deleting, editing, renaming, reordering, or adding any file under
   `receipts/` or `verifier/` causes the corresponding index verify to fail
   closed. (Tested, not asserted.)
5. Every `verifier/runs/*.md` is bound to the SHA-256 of the
   `verifier/vN/acceptance.md` it claims to satisfy and to a `git_revision`.
6. Each `verifier/vN/` has a signed acceptance Merkle root over its criteria
   document and all runs claiming that version, in log order.
7. CI verifies both indices against a **committed pinned public key** and a
   **committed witness checkpoint**; truncating either index together with its
   checkpoints fails the build.
8. `audit-trace` links a published `receipt_sha256` from a sealed receipt to
   its ledger entry and produces an inclusion proof that rebuilds the signed
   root — offline, with `audit-check`.
9. No existing honesty check is weakened: `FORBIDDEN_RESEARCH_METRIC_KEYS` /
   `FORBIDDEN_HEADLINE_TOKENS` unchanged, `live_pnl_claim: false` still stamped
   on every new record type, SYNTHETIC labeling intact, and
   `tests/fx1/test_honesty_inheritance.py` still passing.
10. New tests from §5.5 all pass, including the two that must fail on today's
    tree (`test_seal_and_trace_digest_agree_on_sealed_receipt`,
    `test_receipts_reverify_is_green_on_committed_tree`) — landing them red
    first is the point.

### 5.7 Explicitly out of scope

- **Public transparency log.** Publishing a Rekor-style public log of research
  receipts is a policy decision with disclosure consequences, not a hardening
  step. Sigstore/Rekor *inclusion* for signatures is in scope (P3.1) because it
  is opt-in per artifact and already implemented.
- **Blockchain anchoring as a trust root.** OpenTimestamps is recommended only
  as a *timestamp floor*; no consensus mechanism becomes part of the
  verification path, and no verification requires a full node.
- **zkML / TEE attestation of the verifier itself.** `fx1/serve/attestation.py`
  already models an attestation ladder for inference; extending it to the
  verifier is a separate lane.
- **Anything that weakens the honesty contract.** No proposal here relaxes
  forbidden-metric scanning, SYNTHETIC labeling, fail-closed behaviour, or the
  no-live-claim rule. Per `AGENTS.md`: strengthen, never weaken.
- **Live trading / broker connectivity.** Unchanged and out of scope; see
  `docs/INSTITUTIONAL_READINESS.md`.

---

## 6. Citations

**Standards — transparency logs and Merkle trees**

1. B. Laurie, A. Langley, E. Kasper. *Certificate Transparency.* RFC 6962,
   IETF, June 2013. <https://www.rfc-editor.org/rfc/rfc6962> — Merkle Tree Hash
   with `0x00`/`0x01` domain separation (§2.1), inclusion proofs (§2.1.1),
   consistency proofs (§2.1.2), Signed Tree Heads (§3.3).
2. B. Laurie, A. Langley, E. Kasper, R. Stradling, A. H. Kitching. *Certificate
   Transparency Version 2.0.* RFC 9162, IETF, October 2021 (Experimental;
   obsoletes RFC 6962). <https://www.rfc-editor.org/rfc/rfc9162> — `TransItem`,
   algorithm agility, OID log identity, `get-all-by-hash`, normative
   verification algorithms, and the §1 caveat that inconsistent views require
   treating the log as a trusted third party.
3. R. Merkle. *A Digital Signature Based on a Conventional Encryption
   Function.* CRYPTO '87, LNCS 293. — original Merkle tree construction.
4. M. Crosby, D. Wallach. *Efficient Data Structures for Tamper-Evidence
   Logging.* USENIX Security 2009. — history trees; the academic basis for
   auditable append-only logs.
5. *Merkle Tree Hash Structure* / Tiger tree (THEX). — hash-tree lineage and
   the second-preimage motivation for leaf/node domain separation.

**Standards — checkpoints, witnesses, quorum**

6. C2SP. *Transparency Log Checkpoints (`tlog-checkpoint`).*
   <https://c2sp.org/tlog-checkpoint> — signed-note body: origin, tree size
   (ASCII decimal), base64 RFC 6962 root; extension lines NOT RECOMMENDED;
   "Logs MUST not sign any checkpoint which is inconsistent with any checkpoint
   it previously signed"; "clients MUST ignore unknown signatures"; Ed25519
   SHOULD.
7. C2SP. *Transparency Log Witness Protocol (`tlog-witness`).*
   <https://c2sp.org/tlog-witness> — `POST /add-checkpoint` with `old <size>` +
   ≤ 63 base64 consistency-proof lines + checkpoint; 404/403/400/409/422 error
   contract; `Content-Type: text/x.tlog.size`; persist-before-respond and the
   atomic check-then-persist requirement (the N/N+K rollback race);
   self-contained offline inclusion proofs; the open monitor-partition problem.
8. C2SP. *Transparent Log Policy (`tlog-policy`).* <https://c2sp.org/tlog-policy>
   — quorum / *k*-of-*n* cosignature policy for commit.
9. C2SP. *Signed Notes.* <https://c2sp.org/signed-note> — the note signature
   line format (`— <key name> <base64>`).
10. Google. *Trillian.* <https://github.com/google/trillian> — production
    Merkle-log backend; signer/server separation and personalities.
11. Certificate Transparency monitors — e.g.
    <https://github.com/google/certificate-transparency-go> (`ctclient`,
    `loglist`), and Cloudflare's *nimbus* / Google's *ct-go* monitor designs.

**Standards — signatures and envelopes**

12. S. Josefsson, I. Liusvaara. *Edwards-Curves Digital Signature Algorithm
    (EdDSA).* RFC 8032, IETF, January 2017.
    <https://www.rfc-editor.org/rfc/rfc8032>
13. D. J. Bernstein, N. Duif, T. Lange, P. Schwabe, B.-Y. Yang. *High-speed
    high-security signatures.* J. Cryptographic Engineering 2(2), 2012.
14. *minisign* — F. Denis. <https://jedisct1.github.io/minisign/> — pre-hashed
    (`-H`) Ed25519 signing with a **signed trusted comment**; OpenBSD
    `signify`(1) is the same lineage.
15. OpenBSD. *signify(1)* manual page — release-artifact signing with a small
    pinned key set.
16. *Sigstore* — <https://www.sigstore.dev>; S. Torres-Arias, P. Kuppusamy,
    J. Curtmola, J. Cappos. *Sigstore: Software Signing for Everybody.*
    ACM CCS 2022. — Fulcio (OIDC-bound short-lived certs), Rekor (append-only
    transparency log of signature events).
17. sigstore/protobuf-specs. *Bundle* —
    <https://github.com/sigstore/protobuf-specs/blob/main/protos/sigstore_bundle.proto>.
    `media_type = application/vnd.dev.sigstore.bundle.v0.3+json`;
    `VerificationMaterial` oneof (public-key identifier / X.509 chain / single
    certificate — v0.3 + PGI + keyless **MUST** use the single-certificate
    form); `tlog_entries` carrying inclusion **proofs**;
    `TimestampVerificationData.rfc3161_timestamps`, usable "in conjunction for
    a stronger trust model."
18. *Dead Simple Signing Envelope (DSSE).*
    <https://github.com/secure-systems-lab/dsse> — signs
    `"DSSEv1" || SP || len(type) || SP || type || SP || body` to prevent
    cross-protocol signature reuse.
19. *age* — F. Valsorda. <https://age-encryption.org> — **encryption**, not
    signing; noted here only to rule it out for evidence artifacts.

**Standards — attestations and provenance**

20. in-toto. *Attestation Framework v1.0 — Statement layer.*
    <https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md> —
    `_type: https://in-toto.io/Statement/v1`, `subject[]` (each **MUST** have
    `digest`; subjects assumed immutable; matched purely by digest),
    `predicateType`, `predicate`; parsing rules (ignore unrecognized fields;
    unset/null/empty equivalent; major version in the type URI).
21. in-toto. *Predicate / ResourceDescriptor* specs, same repository.
22. SLSA. *Provenance v1.0.* <https://slsa.dev/spec/v1.0/provenance> —
    `predicateType: https://slsa.dev/provenance/v1`; `builder.id` as the
    transitive closure of the trusted platform; `buildType` as the
    parameterized template; `externalParameters` untrusted and MUST be verified
    downstream; `internalParameters` platform-trusted; `resolvedDependencies`;
    `subject`.
23. SLSA. *SLSA v1.0 specification and levels.* <https://slsa.dev/spec/v1.0/>.
24. OpenSSF. *Model Signing (`model-signing`).*
    <https://github.com/ossf/model-transparency> — in-toto/DSSE + Sigstore
    signing for ML checkpoints; the direct migration target for
    `fx1/serve/signing.py`.
25. J. Cappos et al. *in-toto: Providing farm-to-table guarantees for bits and
    bytes.* USENIX Security 2019. — the layout/step/inspection model.
26. *The Update Framework (TUF).* <https://theupdateframework.io> — threshold
    signatures, key rotation, and the freeze/rollback/out-of-sync attack
    taxonomy that this lane's N4/N5 map onto.

**Standards — trusted time**

27. C. Adams, P. Cain, D. Pinkas, R. Zuccherato. *Internet X.509 Public Key
    Infrastructure Time-Stamp Protocol (TSP).* RFC 3161, IETF, September 2001.
    <https://www.rfc-editor.org/rfc/rfc3161> — `TimeStampReq` message imprint,
    CMS `TimeStampToken`, TSA trust model.
28. D. Pinkas, N. Pope, J. Todd. *Linked Time-Stamp Services: Concepts,
    Definitions and Mechanisms.* RFC 4998, IETF, August 2007. — evidence
    records for long-term archive timestamps.
29. *OpenTimestamps.* <https://opentimestamps.org>; P. Todd et al. — calendar
    Merkle aggregation anchored into Bitcoin, detached `.ots` proofs,
    verification with no trusted third party.
30. S. Haber, W. S. Stornetta. *How to Time-Stamp a Digital Document.*
    CRYPTO 1990, LNCS 537. — the original hash-chain timestamping paper and the
    theoretical basis for both RFC 3161 and OpenTimestamps.

**Repo surface audited** (all paths relative to `D:/dipcatcher`)

`AGENTS.md`; `verifier/README.md`; `verifier/v1..v8/acceptance.md`;
`verifier/runs/*.md` (15 files); `receipts/*.json` (10 files);
`research/reality/studies/reality-us-liquid-daily-2026-09-27/{receipt.json,audit/entries.jsonl,audit/checkpoints.jsonl}`;
`src/quant_fund/audit/{canonical.py,ledger.py,merkle.py,record.py,signing.py,trace.py,verify.py}`;
`src/quant_fund/cli/{audit_cmds.py,report_cmds.py,research_cmds.py}`;
`src/quant_fund/research/{verify.py,receipt_v2.py,phase1_verify.py,catalog.py}`;
`src/quant_fund/proofcore/{contracts.py,ci.py,__init__.py}`;
`src/quant_fund/proof/{bundle.py,sign.py,verify.py,recorder.py}`;
`src/quant_fund/research/reality_sweep.py`;
`src/quant_fund/utils/hashing.py`;
`src/fx1/data/ledger.py`; `src/fx1/serve/{signing.py,attestation.py}`;
`tests/unit/audit/{test_ledger_tamper.py,test_merkle.py,test_signing.py,test_trace.py,test_record_paper.py,test_cli.py}`;
`tests/property/test_adversarial_receipts.py`;
`tests/regression/test_phase1_sealed_receipts.py`; `Makefile`; `docs/evidence/index.md`.

---

## 7. Reproducing every measurement in this document

All commands run from the repo root on the tracked tree at `3dafeb7`.
No file was modified by this lane.

```powershell
# §1.4 / N13 — the committed receipt gate is red
uv run dipcatcher verify-receipt receipts/fleet_eval_5ddf15b0dc7d3ca1.json   # pass
uv run dipcatcher verify-receipt receipts/incumbent_bench_qlib.json          # fail:
                                                                             #   receipt_sha256_missing_or_invalid
uv run dipcatcher verify-research receipts/fleet_eval_5ddf15b0dc7d3ca1.json  # fail:
                                                                             #   invalid_notebook_firm, ...
uv run python -m quant_fund.proofcore.ci receipts-reverify receipts          # exit 1, 10/10 fail

# §1.5 — the one committed ledger
uv run dipcatcher verify-ledger research/reality/studies/reality-us-liquid-daily-2026-09-27/audit
#   -> valid=true, fully_signed=true, tree_size=1, trust_anchor="bundled"

# §2.4 / N7 — the digest divergence, on two independent artifacts
$env:PYTHONPATH="src"; uv run python -c @"
import json
from quant_fund.research.verify import _receipt_digest
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
for p in ('receipts/fleet_eval_5ddf15b0dc7d3ca1.json',
          'research/reality/studies/reality-us-liquid-daily-2026-09-27/receipt.json'):
    d = json.load(open(p)); body = {k: v for k, v in d.items() if k != 'receipt_sha256'}
    print(p)
    print('  seal      ', d.get('receipt_sha256') or hash_bytes(canonical_json_bytes(body)))
    print('  _receipt_dig', _receipt_digest(d))
"@
#   fleet_eval : seal 5ddf15b0dc7d3ca1...  _receipt_digest 5ef209f2bccfb987...
#   reality    : seal 514ccac702d5fa65...  _receipt_digest 52c50588cd3dc0e1...
#                                          (== the value in entries.jsonl)

# N8 / N9 — proofcore Merkle is order-independent and duplicate-of-max malleable
$env:PYTHONPATH="src"; uv run python -c @"
import hashlib
from quant_fund.proofcore.contracts import merkle_root_hex as m
h = lambda s: hashlib.sha256(s.encode()).hexdigest(); a, b, c = h('a'), h('b'), h('c')
print('order-independent      :', m([a,b,c]) == m([c,b,a]))      # True
print('duplicate-of-max malle.:', m([a,b,c]) == m([a,b,c,max(a,b,c)]))  # True
print('root                   :', m([a,b,c])[:16])                # d44f177398ab19ee
"@

# N6 — symmetric seals
#   src/quant_fund/proof/sign.py       scheme "hmac-sha256", key from PROOFCORE_SIGNING_KEY
#   src/fx1/serve/signing.py           hmac.new(_key(), manifest_bytes, sha256), FX1_SIGNING_KEY

# N1 / N2 / N3 — no integrity mechanism over verifier/ or the receipts/ set
git ls-files verifier          # 8 acceptance.md + 15 runs/*.md + README.md; no digests
git ls-files receipts          # 10 .json; no index, no manifest, no Merkle root
git log --show-signature -1    # commits are GPG-signed (RSA B5690EEEBB952194) but the
                               # public key is not in the repo, so the signature cannot
                               # be checked locally: "Can't check signature: No public key"
```

**Honesty framing.** This is a research-infrastructure and correctness-
engineering note. It contains no market result, no performance claim, no
promotion, and no live-trading authorization. `live_pnl_claim` remains `false`
on every artifact discussed here, SYNTHETIC results remain correctness tests
and never market evidence, and nothing proposed weakens
`FORBIDDEN_RESEARCH_METRIC_KEYS` / `FORBIDDEN_HEADLINE_TOKENS` or their fx-1
mirror. Every measurement above is reproducible from the tracked tree.
