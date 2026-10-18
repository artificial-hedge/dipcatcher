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
| Bitcoin anchor | `quality/timestamps/ots/*.ots` + `*.hdr` + `*.blk` + `ots_anchors.json` | `dipcatcher ots-stamp` / `ots-upgrade [--full]` / `verify-repo` (`timestamp_anchors` gate) | A second, independent trust root: `ots-stamp` POSTs the epoch-heads sha256 (never its contents) to the public OpenTimestamps calendar pool and commits the merged detached proof — `pending:<uri>` proves calendar submission immediately. `ots-upgrade` polls each anchor's own calendars for the confirmed timestamp and commits the claimed 80-byte block header, after which verify runs a pure sha256d<nBits PoW check (`bitcoin:<height>:pow_verified`). `--full` additionally commits the block's txid list + raw coinbase (`*.blk`), and verify then reaches `fully_verified`: coinbase txid = txids[0], the recomputed block merkle root equals `header[36:68]`, and the attestation's committed digest sits in a coinbase OP_RETURN push — end-to-end leaf→block with zero explorer trust. `scripts/verify_ots_auditor.py` is a dependency-free independent reimplementation (differentially pinned against the library in tests); a proof stale to the current target reports `fresh=false`/`[stale]` honestly — a timestamp of superseded bytes is still valid evidence of *those* bytes |
| Inclusion proof | `corpus_proof_*.json` (RFC 6962 merkle path) | `dipcatcher corpus-proof --check` | "Was this file in epoch N?" in O(log n) — the path recomputes the epoch's Merkle root and binds to the named chained receipt. `member_tree_root` is sealed into the epoch payload and mirrored into the heads pin, so `--check --pin epoch_heads.json` verifies offline — the quorum-signed, OTS-anchored pin is the whole trust root. `leaf_index` is shape-bound: the side sequence is a pure function of (index, n_members), so a forged index cannot recompute the pinned root |
| Absence proof | `corpus_absence.v1` sealed receipts (bounding neighbors) | `dipcatcher corpus-absence --member X`; `--check` / `--check --pin` | "Was this name NOT committed at epoch N?" — the two sorted neighbors bracketing the gap carry shape-bound inclusion paths whose leaf_indexes are provably adjacent (`hi == lo + 1`), so no member could sit between them. Verified offline against the heads pin or live via `verify_epoch_absence`; standalone oracle: `scripts/verify_corpus_proof.py --absence` |
| Member-name authenticity | enforced inside `corpus_epoch` / `epoch_merkle` | automatic on every stamp + verify | Filesystem-semantic attacks: NFD spellings alias NFC members on APFS (`member_name_not_nfc`), distinct names colliding under casefold can't coexist on macOS/Windows checkouts (`member_name_alias`), symlink members hash their *target's* bytes (`member_is_symlink`), and control/format/line-separator codepoints can inject fake verdict lines or bidi-rename members (`member_name_not_portable`). Stamp refuses all four; the live chain flags them. Every diagnostic label that interpolates a filename carries its ASCII repr, so even an unstamped hostile name can't inject output |
| Authenticated epoch loading | `_load_epoch_receipt` seal-recompute + `corpus_epoch_<sha16(seal)>` name binding | inside `verify_epoch_proof` / `verify_epoch_absence` | Forged or renamed epoch receipts presented to live proof verification: the receipt body must re-seal (`epoch_receipt_tampered`) and its filename must be the one its seal dictates (`epoch_receipt_name_mismatch`), v2 envelopes unwrapped. `check_epoch_chain` additionally contract-verifies every committed epoch (`epoch_receipt_invalid:<name>`) |
| Consistency proof | `epoch_consistency.v1` (hop digest list) | `dipcatcher corpus-consistency --check [--held]` | History rewrite: proves the live chain *extends* an old head digest a verifier already holds (anchored pin, earlier clone) — a rewritten chain can't reach the held bytes without keeping every real hop verbatim |
| Verifier coverage | derived jewel set | inside `crown-jewels --check` | A new integrity-critical module (`*_epoch*`, `*merkle*`, `*anchor*`, …) that dodges `DEFAULT_JEWELS` fails `verifier_unpinned` — the pin can't be dodged by omission |
| Checkpoint | `quality/checkpoint.json` (signed, anchored) | `dipcatcher verify-checkpoint` | Portable signed tree-head: an auditor needs only checkpoint + pubkey + its TSA token — signature verifies offline, pinned digests bind a clone. `current` flags staleness without invalidating authenticity |
| Checkpoint spine | `quality/checkpoints/*.json` — every historical checkpoint, each carrying `prev_sha256` | `dipcatcher checkpoint-chain` | Pin-state history as a *chain*, not a point: deleting an archive record → `dangling_prev`, injecting a side-chain → `spine_fork`+`spine_orphan`, a re-signed record → `signature_invalid`, Rekor-witnessed digests that vanished → `witnessed_absent`. Terminus resolvable only in Rekor (pre-archive era) is a valid `witness` genesis |
| Transparency log | `quality/witness/*.json` (Rekor hashedrekord entry) | `dipcatcher verify-witness` (offline), `verify-witness --online` (live consistency: committed tree is a prefix of Rekor's *current* signed head) | External non-repudiation: the checkpoint digest + our ECDSA witness sig sit in sigstore's public append-only log — a state *we* can't rewrite either. Proof self-verifies: RFC 6962 inclusion walk to Rekor's signed root + signed-entry-timestamp + checkpoint-note sig under the pinned Rekor pubkey. Proofs minted before the archive era bound `witnessed_absent` vs `grandfathered_witness` (pre-retention history) |
| Auditor bundle | `auditor_bundle.v1` (one JSON doc) | `dipcatcher witness-bundle` / `verify-bundle <file>` | Zero-trust third-party audit: bundle carries checkpoint + pins + pubkeys + **every archived checkpoint and every committed witness proof** — the auditor replays the whole spine (links, signatures, forks, orphans, Rekor order), not just the head. The witness key is authenticated by the **log itself** — Rekor's entry body records the signer pubkey, which must equal the bundled `witness_signing.pub` byte-for-byte. `scripts/verify_auditor_bundle.py` is an independent second implementation (same verdict contract; members outside the pinned prefixes fail `unexpected_member`). No repo access, no trusted inputs beyond Rekor's own key. The bundle also carries `auditor_self_sha256` — the sha256 of the standalone script at build time — so an auditor can diff their local copy before trusting its verdict; a mutated script warns `auditor_self_mismatch` |
| Tamper drill | `tamper_drill.v1` | `dipcatcher tamper-drill` | Self-adversarial: clones the real tree, lands every probe mutation (epoch/merkle/anchor/sig/checkpoint/spine/key classes), and requires `verify-repo` to flag **every** one — a verifier that went blind reports `missed:` instead of `ok` |
| Fuzz drill | `fuzz_drill.v1` | `dipcatcher fuzz-drill --seed N` | Metamorphic two-sided fuzzing of the substrate itself: seeded byte flips/truncations/key-renames across corpus members, jewels, pins, spine records, witness proofs, and checkpoints — plus benign probes (`docs/` write) that must NOT alarm. Grades `escaped`/`false_positive`; a `baseline_dirty` verdict refuses to drill on an already-drifted tree. First campaign: 4 real escapes found and closed (spine extent, checkpoint-absent fail-open, unstamped injection, witness-proof delete/byte-drift) — 6 seeds × 17 mutations (incl. crafted pin-rollback / forged-epoch / removal-laundering / foreign-extension-drop policy attacks) now fully `calibrated` |
| Receipt fuzz | `receipt_fuzz.v1` | `dipcatcher fuzz-receipts --seed N` | Forge-and-reseal: mutates one claim leaf inside each committed receipt, then re-seals *honestly* (sha256 seals are integrity, not authenticity — a self-consistent forgery is free to mint). Maps the verification boundary: escapes = claim fields that are seal-bound but not contract re-derived (mostly `*_real_drill` result fields whose source streams aren't committed — they are *attested*, not *verified*). Deliberately a measurement lane, not a gate — the `replay_proof` lane closes the gap by pinning inputs and re-executing |
| Key rotation | `rotation_*.json` (`key_rotation.v1`) | `dipcatcher verify-rotations` | Key substitution without authorization: each link is dual-signed — the outgoing key proves *authorization*, the incoming proves *possession* — and the first link's `old_pubkey` must re-verify a real spine signature (`unanchored_genesis` otherwise). The spine walks a **keyring** — live key + every authorized retired key — so records verify under their era's key (`signature_key_unknown` for a never-authorized signer). Live `gate_signing.pub` must equal the chain terminus |
| Custody | `custody_proof.v1` (one JSON doc) | `dipcatcher custody --member X --out bundle.json`; `custody --check bundle.json --member-file F`; `custody --member X --timeline` | Per-member provenance with **zero repo access**: the bundle embeds the member's earliest-chain pinning epoch (proof of first committed state), the epoch→head hop files raw (digests re-derived on verify), inclusion merkle path, signed pins, checkpoint, both witness pubkeys, every committed Rekor proof, and the RFC 3161 timestamp dir — materialized into a synthetic root so the existing file verifiers run unmodified. Eight layers: member→inclusion→chain→pins→signature→checkpoint→witness→timestamps. `--timeline` enumerates a member's committed digests across epochs (byte-evolution lineage for mutable corpora) |
| Quarantine registry | `quality/legacy_quarantine.json` (`legacy_quarantine.v2`) | `suite-health --strict --quarantine …` | Retained receipts that cannot verify (pre-seal era, detached preregistration seals, contracts that postdate the bytes, unpinned tapes, absent honesty declarations). Each is pinned by path, bytes, exact `verify-receipt` error set, category and reason. One category, `admission_honesty_gap`, holds receipts that verify but entered the corpus without passing admission: its error set is admission's honesty codes, re-derived from the bytes on every check, and the entry fails closed (`quarantine_entry_admissible`) once nothing is missing. Base rule (`dipcatcher quarantine-gate`, run by `make admission-gate` on every pull request and every push to `main`, against the PR base or the pre-push tip): an entry the base registry did not hold must pin bytes the base already had (stamped by its head, or present under the same name), so a change can never add a receipt and register it in the same diff; an `admission_honesty_gap` entry must name an unstamped base file with identical bytes, so a stamped receipt cannot be selectively demoted. Such receipts are reported as `quarantined`. `suite-health` never counts them as verified or harvests them into pooled evidence, and admission never admits them. Known gap, pre-existing: `receipt_lattice` and `corpus_inference` still read claims from top-level quarantined receipts. An unregistered failure, registry drift, a stale entry, an entry that now verifies, or an unregistered file under `legacy-unsealed/` fails strict. Production gates and tests read the same registry. `schema_drift_acks` pins every member of a tag group whose two claim shapes are sealed history, so `schema_drift` reports that group as acknowledged drift instead of accepting a retag-and-reseal. Both the registry and the ledger below are crown jewels: changing either is a gate change |
| Epoch acknowledgments | `quality/epoch_allowed_removals.json` (`epoch_acknowledgments.v1`) | `--epoch-acks` on `corpus-epoch --check`, `suite-health`, `admit`/`admit-batch`; loaded by `verify-repo` and the standalone auditors | An append-only chain can neither forget nor rewrite a recorded removal or mutation. Each entry explains exactly one transition already recorded in a sealed epoch: corpus key, member, `removed`/`mutated`, from/to digests, epoch and reason. Acknowledged incidents are printed on every check. An entry that matches nothing is `epoch_ack_unmatched`. Live drift (`head_member_missing_live`, `head_member_digest_drift`) is never acknowledgeable: stamp first, then explain. On append-only chains a name removed and later re-added with different bytes, under any case or Unicode-normalized spelling, is `member_resurrected` (a mutation spread over two epochs; acknowledgeable only as the exact `mutated` transition). Before it is stamped, the same re-add is `member_resurrected_live`, which is never acknowledgeable: new evidence takes a new name. Base rule (`dipcatcher epoch-ack-gate`, run by `make admission-gate` on every pull request and every push to `main`, against the PR base or the pre-push tip): every entry must name a chain in the corpus policy table (`epoch_ack_unknown_corpus` otherwise), and an entry the base ledger did not hold, for an append-only chain, must name an epoch of the base chain, authenticated by `chain_ancestry` against the base pin's `chain_root`. A base ledger that is absent or pre-v1 reads as empty |
| Admission | `receipt_admission.v1` | `dipcatcher admit-batch --strict --quarantine … --heads-pin … --epoch-acks …` | A new receipt that breaks lattice/FDR on entry. The epoch check binds to the real merged corpus under the heads pin and ledger, not to the intermediate shadow. Registry-matched quarantined candidates are listed as retained and never admitted. Candidates whose bytes the *base* branch's pinned head epoch already stamped (`--base-heads-pin`, e.g. a receipt restored to its stamped bytes) are corpus members, so they are listed as already-stamped rather than re-admitted. Only the base pin can grant this, and it must authenticate its head epoch. With the base pin, every acknowledgment (removal or mutation) must name an epoch of the base chain, authenticated against the base pin's `chain_root` (`epoch_ack_postdates_base` otherwise), so one change can never both record a removal or rewrite of an append-only receipt and excuse it. When no candidate is gated, the merged corpus's chain is still checked once (`chain_errors`) |
| Retraction | `receipt_tombstone.v1` (`tombstone_<sha16>.json`) | `dipcatcher tombstone <receipt> --reason ...` | The corpus is append-only — a wrong/superseded receipt can't be deleted (epoch flags removal, lattice flags drift), so it is retracted: a sealed record pinned to the target's bytes excludes its claims from the lattice (`retracted:` block), honors partial `scope` per claim path, and fails closed if the target drifts (`tombstone_target_drift`) or the tombstone is unsealed |
| Release attestation | `release_attestation.v1` (one JSON doc) | `dipcatcher attest-release --artifact W [--witness]` / `verify-release A --artifact W` | Extends the proof to bytes that live *outside* the repo (wheels, sdists, exported bundles): an Ed25519-signed binding of each artifact's sha256+size to the live pin-state digests (`attestation_stale` when they've drifted — staleness ≠ forgery). With `--witness` the attestation is anchored in Rekor and the self-verifying proof travels **inside** the payload — both pubkeys embedded, the witness key authenticated by the logged entry itself — so the attestation alone convinces an auditor with no repo access |
| Capstone | `repo_integrity.v1` | `dipcatcher verify-repo` | One sealed verdict over all of the above — contract-checked so the attestation can't claim `ok` while a gate lists errors, hide a gate entirely, or disagree with its own digest pins. `--evidence-only` verifies an exported evidence bundle (the five evidence dirs + `gate_pins.sig`, no `src/`): crown_jewels reports a skipped marker and the attestation records `mode=evidence_only`, so a partial verdict can never masquerade as a full-tree one. Coverage closure: every file under a corpus dir must match the declared member glob or a per-corpus exemption (`uncovered_member:<rel>` otherwise) — a `.yaml` workflow dropped beside `*.yml` members would still run on GitHub, and now surfaces |

## Corpus discipline

Twenty-eight corpora are epoch-chained under `quality/epoch_heads.json` —
the *entire tracked tree*: after the eight evidence corpora, every code and
doc directory is closed-world chained too (`src`, `tests`, `scripts`,
`docs`, `research`, `replay`, `reports`, `notebooks`, `examples`, `clients`,
`typings`, `spec`, `docker`, `deploy`, `third_party`, `rust`, `web`,
`.box-soft-verify`, `.cursor`, `.github` beyond `workflows/`). A tamper
anywhere in the repo — source, test fixture, vendored code, deploy surface —
drifts a head. Two pins keep the world closed: every tracked top-level dir
must be a corpus or an explicit tool-state opt-out, and the root-level file
set is pinned (root files sit above every corpus dir).

- `receipts/*.json` — append-only, strict: any member digest change between
  epochs is `member_mutated` (tamper evidence). `legacy-unsealed/` is
  quarantined with byte-pins + recorded verify-errors (`quality/legacy_quarantine.json`),
  as are retained root-level receipts that cannot verify. A wrong receipt is
  quarantined or tombstoned, never re-sealed or edited in place. A recorded
  historical removal or mutation is explained in the epoch acknowledgment
  ledger, never rewritten.
- `verifier/*.md` — append-only acceptance records, strict.
- `quality/*.json` — mutable bookkeeping: `--allow-member-updates` gives ordered
  history attestation; a post-stamp edit still fails as `head_member_digest_drift`
  until `make stamp-epochs` re-stamps.
- `.github/workflows/*.yml` — mutable *and* security-critical:
  `--require-stamped` makes every unstamped `.yml` arrival a CI error plus
  `--allow-member-updates` for edits. The rest of `.github` (templates,
  plans) is its own `*` corpus.
- `configs/*` — mutable: post-hoc config edits would silently rewrite what a
  sealed bench measured, so each change needs a stamp like the quality corpus.
- `artifacts/*` — committed claim artifacts (champion selects, dev grids):
  regenerated between runs → mutable, stamped arrivals only.
- `.dsh-24x7/*` — committed fleet evidence (the SOTA input stream): a live
  dir, so arrivals land between stamps, but a stamped member is immutable —
  an `.npz` mutating post-stamp is tamper.
- The twenty closed-world corpora — mutable (`--allow-member-updates`) with
  stamped arrivals (`--require-stamped`): any member byte drifting between
  epochs is recorded tamper evidence, and a committed file no check covers is
  a coverage hole (`uncovered_member`). Exemptions: `__pycache__/` dirs are
  never members anywhere (machine-local bytecode), and `.github`'s corpus
  exempts `workflows/` (its own chain). Membership exemptions are corpus-scoped
  by basename — a `witness/` dir under `receipts` is an ordinary member.
- `data/metadata/*` — committed dataset manifests + validation outputs: the
  inputs `data_manifest` receipts pin; write-once dated dirs, stamped
  arrivals only.

After touching a covered file: `make stamp-epochs` (advances all chains +
pin) and `dipcatcher crown-jewels --write` if a jewel changed. Both must land
in the same commit as the change. If `GATE_SIGNING_KEY` is provisioned,
finish with `make sign-pins` — the signature must cover the final pin bytes
(sign last, since stamping updates `epoch_heads.json`), then
`make checkpoint` + `dipcatcher witness-checkpoint` to extend the spine and
anchor the new pin-state in Rekor.

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
every commit (reviewed before merge on a pull request, but only after
landing on a direct push to `main`; see the residual boundaries below), and
`verify-repo` receipts pin the pin files' digests into the
receipts corpus so their history is itself chained. Either way, silent
tamper is impossible — any rewrite leaves a committed, hash-linked trail or
a broken signature.

Standalone-verifier boundary, stated plainly: `verify_epoch_proof` /
`verify_corpus_proof.py --proof` accept a *properly-named, self-sealed,
contract-valid* epoch receipt they are shown — a wholly fabricated chain
link verifies standalone because nothing in the file itself is dishonest.
Provenance is owned one layer up: the heads pin (`epoch_heads.json`) names
the exact chain terminus a proof must bind to, `check_epoch_chain` rejects
orphan/fork/multiple-genesis chains, and the pin is quorum-signed +
OTS-anchored. An auditor should therefore always pass `--pin` (or
`--checkpoint` + `--pubkey`, which additionally binds the pin file's bytes
inside the Ed25519-signed checkpoint) — bare `--proof` mode authenticates
shape, not custody.

Residual boundaries, as of 2026-10-14 (measured on this repair merged
with `main` at b0e48dd60, after stamping the receipts and quality chains;
not assumed):

- **`make evidence-audit` is red on inherited drift, and stops early.**
  After the receipts and quality chains, it checks `.github/workflows`,
  `configs`, `artifacts`, `data/metadata` and the code corpora (`src`,
  `tests`, `scripts`, `docs`, ...). Those chains carry post-stamp drift
  that predates this repair, and this repair's own edits to files in them
  add to it until they are reviewed and restamped. Make stops at the
  first failing command, so `crown-jewels --check`, `verify-witness`,
  `verify-repo` and the drills after it do not run. The CI
  `evidence-audit` job runs its later steps (bundle round-trip,
  chain-consistency proofs, lattice, `admission-gate` with
  `epoch-ack-gate` and `quarantine-gate`, replay sweep) with `!cancelled()`
  after a successful sync, so those report on every pull request and every
  push to `main` even while this step is red. The base-comparison steps
  compare against the PR's base branch or, for a direct push, the tip
  `main` had before the push (`github.event.before`). The replay sweep is
  red: at the 120 s default (`make replay-sweep`) 16 carriers give
  pass=0 fail=2 skipped=14; with `--timeout 600` they give pass=1 fail=1
  skipped=14. The `mcs_vol` carrier passes only with the longer timeout
  (139.7 s, `independent_confirmation.elapsed_s` in the migration
  record). The corpus carrier
  (`data/manifests/replay/corpus.json`) fails on both: its receipt
  reproduces byte-for-byte, but the artifact it pins embeds
  `generated_at_commit`, so the regenerated bytes can match only at the
  commit the pin was made at. On `main` (b0e48dd60) that carrier is
  skipped as `inputs_not_verified`, because the receipts it replays did
  not verify there; this repair restored them, so the carrier now runs
  and fails. Making it reproducible needs a producer change and a new
  manifest pin, which rewrites pinned evidence and is left to the lane
  owner.
- **Crown-jewel pins are current; the pin signature is not.** The admission
  and auditor hardening changes four jewels (`Makefile`,
  `scripts/verify_epoch_chain.py`, `src/quant_fund/research/admission.py`,
  `src/quant_fund/research/corpus_epoch.py`), re-pinned with
  `dipcatcher crown-jewels --write` in the same commit, so
  `crown-jewels --check` reports no drift on this tree. The signature over
  the pins (`gate_pins.sig`) is still stale: it was last signed on
  2026-09-30, and `verify-repo`'s `pin_signatures` gate reports `pin_drift`
  on `crown_jewels.json` and `epoch_heads.json`. This was already true on
  `main` before this change. It cannot be refreshed without
  `GATE_SIGNING_KEY`; until the key holder runs `make sign-pins` (then
  `make checkpoint` and the witness/anchor steps), the pins are checked only
  when `crown-jewels --check` or `verify-repo` is run directly, and
  `evidence-audit`, `tamper-drill` and `fuzz-drill` stay red on
  `baseline_dirty`.
- **Admission's honesty bar covers arrivals, not stamped history.** The
  receipts head (`corpus_epoch_588a35edec9d3f5b`) stamps 382 members that
  verify and are not in the registry: 44 epoch receipts and 338 claim
  receipts. Run through `honesty_stamp_errors`
  (`quant_fund.research.legacy_unsealed`, the check admission uses), 170
  of the 338 lack at least one stamp admission now requires, and 143 of
  those lack only `live_pnl_claim: false`. They predate that bar and stay
  stamped members.
  `p0_evidence_inventory_v1.json` is quarantined not because its content
  is worse but because it was never stamped or admitted.
- **`data/metadata` epoch receipts are tracked via `!data/metadata/corpus_epoch_*.json`.**
  An earlier pin named a gitignored head (`corpus_epoch_7dc5e67df8042167.json`)
  and the only tracked ancestors were orphaned (`prev` never committed), so
  `evidence-audit` could not pass. 2026-10-15 reset that corpus to a fresh
  committed genesis of the live members and un-ignored `corpus_epoch_*.json`
  so `make stamp-epochs` heads stay reviewable. `epoch-consistency` still
  skips a chain only when the base tree does not commit the base pin's head;
  deleting a head the base does commit still fails the proof. A base whose
  heads pin is missing or unreadable fails the gate.
- **A bad direct push to `main` is reported, not prevented.** Direct
  commits to `main` are this repo's normal path, so `epoch-consistency`,
  `epoch-ack-gate`, `quarantine-gate` and the admission diff also run on
  `push` events, against the pre-push tip. They turn the run red after
  the push has landed; nothing rejects the push unless branch protection
  requires the check. If the pre-push tip cannot be fetched (for example
  after a force-push), the base step fails and the run is red. Only a
  push that creates `main` has no pre-push tip; the step says so and the
  base steps are skipped. Scheduled and manual runs have no change to
  compare and do not run them.
- **`schema_drift` needs full history.** Its phantom-revision check
  resolves `git_revision` fields through `refs/tags/attic/receipt-provenance/*`.
  It runs on the real corpus in the CI `test` job, which checks out with
  `fetch-depth: 0` and so fetches those tags. A shallow clone (such as the
  `evidence-audit` job's default checkout) would report them as phantoms.

Key lineage: when `GATE_SIGNING_KEY` rotates, `dipcatcher rotate-key`
records a dual-signed `key_rotation.v1` link (old key authorizes, new key
proves possession), then the operator installs the new pubkey in
`gate_signing.pub`, re-signs (`make sign-pins`), and checkpoints + witnesses
the transition. The rotation chain's genesis is anchored to the spine: the
outgoing key must have signed a real checkpoint. A substituted key whose
holder never signed — i.e. never held authority — fails
`unanchored_genesis`; a rotation recorded but never installed fails
`live_key_not_terminus`.

## Operator quick reference

```bash
make evidence-audit        # all gates (CI runs this): suite-health --strict with the
                           # quarantine registry, heads pin + epoch ack ledger, every chain
make stamp-epochs          # re-stamp chains + heads pin after touching a corpus
dipcatcher verify-repo     # compose every gate into one sealed verdict
dipcatcher verify-repo --out quality/repo_integrity.json   # seal the attestation
dipcatcher admit-batch receipts/<new>.json --strict          # gate a new receipt
```
