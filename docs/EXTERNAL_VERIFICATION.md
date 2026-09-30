# External receipt verification (referee path)

The lab's own receipts (`docs/RECEIPT_VERIFICATION.md`, `docs/RECEIPT_V2.md`)
audit work produced inside this repository. This path audits somebody else's:
a frozen external data snapshot a third party claims to have preserved, and
what that snapshot says about revisions between vintages. The verifier is
`quant_fund.research.external_receipt.verify_receipt`; the CLI is
`scripts/verify_external_receipt.py`.

Provenance only. A passing verification is not a benchmark, not a SOTA claim,
not a proof, and never a live-performance claim. There is no broker
connectivity anywhere in this path.

## What an external receipt must contain

One JSON file plus the CSV snapshots it references.

```json
{
  "series": "VIXCLS",
  "snapshots": [
    {
      "vintage_date": "2026-01-05",
      "path": "snapshots/synthetic_vixcls_20260105.csv",
      "sha256": "<sha256 of the snapshot file bytes>",
      "pit_status": "candidate_only",
      "proof_status": "not_proof"
    },
    {
      "vintage_date": "2026-02-02",
      "path": "snapshots/synthetic_vixcls_20260202.csv",
      "sha256": "<sha256>",
      "pit_status": "candidate_only",
      "proof_status": "not_proof"
    }
  ]
}
```

Each snapshot CSV is one vintage of one series: exactly two columns,
`observation_date` plus a value column named `VIXCLS_<vintage>` with the
`vintage_date` separators removed (`2026-01-05` → `VIXCLS_20260105`). Dates
are ISO-8601 and unique within the file. `""` and `"."` mean *not published in
this vintage* and are recorded as a null observation. Any other value must
parse as a finite decimal — `nan`, `inf`, and `-inf` are rejected, not coerced.

Honesty labels are part of the receipt, not the verifier's opinion:

- `pit_status` must be `candidate_only`. A snapshot cannot arrive
  pre-promoted to point-in-time evidence.
- `proof_status` must be `not_proof`. A submitter cannot declare its own
  snapshot proven.

`series` is currently pinned to `VIXCLS`; the value-column name check encodes
that. Extending to another series means extending the module and its tests,
not loosening the check.

## How to submit

1. Freeze each vintage as its own immutable CSV. Never edit a vintage in
   place; a correction is a new vintage file.
2. Compute the SHA-256 of each frozen file's bytes and record it in the
   receipt next to its `vintage_date`.
3. Store paths relative to the submission root, or relative to the receipt
   file, so an independent clone can relocate the whole tree together.
4. Send the receipt and the snapshot directory. Nothing else is needed; this
   path performs no network access and resolves no external identifiers.

## How `verify_receipt` adjudicates

```python
from quant_fund.research.external_receipt import verify_receipt

result = verify_receipt("path/to/external_receipt.json", base_dir="path/to")
```

Signature: `verify_receipt(receipt_path, *, base_dir=None) -> dict[str, Any]`.

Snapshot paths resolve in order: `base_dir` when supplied, then the current
working directory, then the receipt's own directory. An absolute `path` is
used as given.

Checks, in order, each fail-closed:

1. `series == "VIXCLS"`; `snapshots` is a non-empty list of objects.
2. Every snapshot has non-empty `vintage_date`, `path`, and `sha256`, plus the
   `candidate_only` / `not_proof` labels above.
3. Vintage dates are unique across the receipt.
4. Each referenced file exists — a missing snapshot is a failure, never a skip.
5. Each file's actual SHA-256 equals the recorded one. One changed byte fails.
6. Each file parses: correct header, exactly one value column, the column name
   matches its vintage, ISO dates, no duplicate or blank date, no non-numeric
   or non-finite value.
7. Snapshots are sorted by vintage and each adjacent pair is diffed over their
   common observation dates, splitting the changes into restated values
   (`value_changes`) and newly published dates (`availability_changes`).

On success the result is provenance, not a verdict on the underlying research:

```json
{
  "ok": true,
  "proof_status": "not_proof",
  "snapshot_count": 2,
  "hashes_verified": 2,
  "revision_pairs": [
    {
      "older_vintage": "2026-01-05",
      "newer_vintage": "2026-02-02",
      "common_observations": 3,
      "changed_observations": 2,
      "value_changes": 1,
      "availability_changes": 1,
      "revisions": [
        {"observation_date": "2025-12-30", "older_value": "14.62",
         "newer_value": "14.71"},
        {"observation_date": "2025-12-31", "older_value": null,
         "newer_value": "15.03"}
      ]
    }
  ]
}
```

`ok: true` means integrity and receipt-schema checks passed. It is not a proof
status; `proof_status` stays `not_proof` on every success path. The result
carries no performance metric of any kind — no Sharpe, Sortino, Calmar, P&L,
or NAV key appears in it, and
`tests/unit/research/test_external_receipt.py` pins that.

## Failures are published

Any failed check raises `ValueError` naming the offending field, path, or
vintage, and the CLI exits nonzero with that message on stderr. This is the
point of a referee position: a submission that does not verify leaves a
readable failure record, not silence and not a quietly downgraded `ok`.

This differs from the notebook verifier on purpose. `verify_research_artifact`
collects errors into `{"valid": false, "errors": [...]}` because a research
notebook can fail many independent ways at once. An external receipt fails at
the first integrity breach, because every later check is meaningless once one
hash or one label does not hold. Publish the exception text verbatim.

Never "fix" a submission by editing a frozen snapshot to match its recorded
hash, by relaxing a label check, or by dropping a vintage that does not
resolve. Re-request the frozen bytes from the submitter, or record the
failure.

## CLI

```bash
uv run python scripts/verify_external_receipt.py path/to/external_receipt.json
uv run python scripts/verify_external_receipt.py path/to/external_receipt.json \
  --base-dir path/to/submission_root
```

Prints the result dict as sorted-key JSON on success; exits nonzero on the
first failed check. Use `--base-dir` whenever the receipt and its snapshots
were submitted as a relocatable tree.

## Limitations

- Hash agreement proves the bytes a submitter froze are the bytes being
  checked. It does not prove the bytes match the upstream vendor's release for
  that vintage. Source entitlement, vendor restatement policy, and historical
  availability reconstruction remain disclosed limitations.
- Revision pairs are reported over vintages present in one receipt. A vintage
  the submitter withheld is invisible here; absence of revisions is not
  evidence that none occurred.
- This is not a digital signature and not an independent attestation of past
  contents. Passing verification never authorizes live trading.
