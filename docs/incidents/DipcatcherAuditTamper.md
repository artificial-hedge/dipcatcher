# DipcatcherAuditTamper

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_audit_tamper is 1 for 1 minute. Startup found a paper journal or audit log that was truncated, reordered, or edited. The hash chain covers lines written by this process. Older lines without a hash are only a prefix.

## Impact

The process does not submit. Readiness stays closed. The chain does not use a signing key. This is not a price or a P&L figure.

## First 5 minutes

1. Read the acceptance code: audit_truncated, audit_reordered, or audit_edited.
2. Compare `journal/events.jsonl`, `audit/audit.jsonl`, and `log_tip.json` with the last backup.
3. Do not delete the files to make startup pass, and do not rewrite the hashes.
4. Restore a known-good copy or leave the book halted.
5. Confirm a clean start clears the gauge and does not replay an order that was already committed.

## Rollback and kill switch

The chain is the evidence. Rolling the image back does not repair a changed line. The operator kill file remains an independent latch. Keep it closed until the log is a copy you trust.
