# Evidence integrity

How dipcatcher keeps the evidence corpus tamper-evident — the layers, what each
detects, and the boundary where detection stops.

## The layers

| Layer | Artifact | Command | Detects |
|---|---|---|---|
| Seal | `receipt_sha256` on each receipt body | `dipcatcher verify-receipt` | Content mutation of a single receipt |
| Contract | per-kind deep checks | `verify-receipt` dispatch | Forged aggregates, renamed kinds, inconsistent claims |
| Lattice | claim groups across receipts | `dipcatcher lattice --strict` | Two receipts contradicting each other on the same claim |
| Provenance DAG | `receipt_graph` | — | Orphaned/fabricated citation edges |
| Epoch chain | `corpus_epoch_*.json` per (dir, pattern) | `dipcatcher corpus-epoch --check` | Receipt deletion, reorder, unstamped arrivals, dishonest deltas |
| Head pins | `quality/epoch_heads.json` | `--heads-pin` on check | Rewind attack: deleting the newest epoch to hide later tamper |
| Crown jewels | `quality/crown_jewels.json` | `dipcatcher crown-jewels --check` | Silent edits to the files that *define* the gates (Makefile, pyproject, uv.lock, hooks, gitleaks, conftests, mkdocs, AGENTS.md) **and the verifier's own source** — a `return ok` patch to corpus_epoch/receipt_v2/lane_contracts/etc. fails `jewel_mutated` |
| Pin signature | `gate_pins.sig` + `quality/gate_signing.pub` | `verify-repo` (`pin_signatures` gate) | Forged pins: an editor can re-pin after rewriting a gate file, but cannot re-sign — Ed25519 covers both pin files' bytes |
| Timestamp anchor | `quality/timestamps/*.tsr` + `anchors.json` | `verify-repo` (`timestamp_anchors` gate) | Retroactive history fabrication: an RFC 3161 TSA signature proves the pinned state existed by wall-clock time T — the chain can't be minted after the fact |
| Inclusion proof | `corpus_proof_*.json` (RFC 6962 merkle path) | `dipcatcher corpus-proof --check` | "Was this file in epoch N?" in O(log n) — the path recomputes the epoch's Merkle root and binds to the named chained receipt |
| Consistency proof | `epoch_consistency.v1` (hop digest list) | `dipcatcher corpus-consistency --check [--held]` | History rewrite: proves the live chain *extends* an old head digest a verifier already holds (anchored pin, earlier clone) — a rewritten chain can't reach the held bytes without keeping every real hop verbatim |
| Verifier coverage | derived jewel set | inside `crown-jewels --check` | A new integrity-critical module (`*_epoch*`, `*merkle*`, `*anchor*`, …) that dodges `DEFAULT_JEWELS` fails `verifier_unpinned` — the pin can't be dodged by omission |
| Checkpoint | `quality/checkpoint.json` (signed, anchored) | `dipcatcher verify-checkpoint` | Portable signed tree-head: an auditor needs only checkpoint + pubkey + its TSA token — signature verifies offline, pinned digests bind a clone. `current` flags staleness without invalidating authenticity |
| Admission | `receipt_admission.v1` | `dipcatcher admit-batch --strict` | A new receipt that breaks lattice/FDR on entry |
| Capstone | `repo_integrity.v1` | `dipcatcher verify-repo` | One sealed verdict over all of the above — contract-checked so the attestation can't claim `ok` while a gate lists errors, hide a gate entirely, or disagree with its own digest pins |

## Corpus discipline

Five corpora are epoch-chained under `quality/epoch_heads.json`:

- `receipts/*.json` — append-only, strict: any member digest change between
  epochs is `member_mutated` (tamper evidence). `legacy-unsealed/` is
  quarantined with byte-pins + recorded verify-errors (`quality/legacy_quarantine.json`).
- `verifier/*.md` — append-only acceptance records, strict.
- `quality/*.json` — mutable bookkeeping: `--allow-member-updates` gives ordered
  history attestation; a post-stamp edit still fails as `head_member_digest_drift`
  until `make stamp-epochs` re-stamps.
- `.github/workflows/*.yml` — mutable *and* security-critical:
  `--require-stamped` makes every unstamped `.yml` arrival a CI error plus
  `--allow-member-updates` for edits.
- `configs/*` — mutable: post-hoc config edits would silently rewrite what a
  sealed bench measured, so each change needs a stamp like the quality corpus.

After touching a covered file: `make stamp-epochs` (advances all chains +
pin) and `dipcatcher crown-jewels --write` if a jewel changed. Both must land
in the same commit as the change. If `GATE_SIGNING_KEY` is provisioned,
finish with `make sign-pins` — the signature must cover the final pin bytes
(sign last, since stamping updates `epoch_heads.json`).

Timestamp anchors (`dipcatcher anchor-timestamp`) go **stale by design** on
every stamp — the `.tsr` binds the pin bytes at request time. Re-anchor at
quiet points: after merges land, at release. A stale anchor stays valid
proof of the earlier state; `fresh: false` just says the pin has moved on.
The FreeTSA certs in `quality/timestamps/` make chain verification offline;
only stamping needs the network.

## Trust boundary

Receipt seals are sha256, not signatures — but the *pins* are: when
`gate_pins.sig` + `quality/gate_signing.pub` are committed, rewriting a gate
file, re-pinning, and re-stamping still cannot produce a valid signature
without the private key (held outside the repo). For unsigned trees the
boundary is **git review**: pins and epochs are small, reviewable diffs in
every PR, and `verify-repo` receipts pin the pin files' digests into the
receipts corpus so their history is itself chained. Either way, silent
tamper is impossible — any rewrite leaves a committed, hash-linked trail or
a broken signature.

## Operator quick reference

```bash
make evidence-audit        # all gates (CI runs this)
make stamp-epochs          # re-stamp chains + heads pin after touching a corpus
dipcatcher verify-repo     # compose every gate into one sealed verdict
dipcatcher verify-repo --out quality/repo_integrity.json   # seal the attestation
dipcatcher admit-batch receipts/new.json --strict          # gate a new receipt
```
