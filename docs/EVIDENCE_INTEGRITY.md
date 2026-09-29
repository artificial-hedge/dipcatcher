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
| Crown jewels | `quality/crown_jewels.json` | `dipcatcher crown-jewels --check` | Silent edits to the files that *define* the gates (Makefile, pyproject, uv.lock, hooks, gitleaks, conftests, mkdocs, AGENTS.md) |
| Admission | `receipt_admission.v1` | `dipcatcher admit-batch --strict` | A new receipt that breaks lattice/FDR on entry |
| Capstone | `repo_integrity.v1` | `dipcatcher verify-repo` | One sealed verdict over all of the above |

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
in the same commit as the change.

## Trust boundary

Seals are sha256, not signatures — a determined editor could rewrite file,
receipt, pin, and chain together and re-stamp. Detection therefore bottoms out
at **git review**: pins and epochs are small, reviewable diffs in every PR, and
`verify-repo` receipts pin the pin files' digests into the receipts corpus so
their history is itself chained. The system makes silent tamper impossible —
any rewrite leaves a committed, hash-linked trail.

## Operator quick reference

```bash
make evidence-audit        # all gates (CI runs this)
make stamp-epochs          # re-stamp chains + heads pin after touching a corpus
dipcatcher verify-repo     # compose every gate into one sealed verdict
dipcatcher verify-repo --out quality/repo_integrity.json   # seal the attestation
dipcatcher admit-batch receipts/new.json --strict          # gate a new receipt
```
