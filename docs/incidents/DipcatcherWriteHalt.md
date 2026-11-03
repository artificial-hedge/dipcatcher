# DipcatcherWriteHalt

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_write_halt is 1 for 1 minute. The process could not append or replace a byte on the ledger, journal, rate-store, or audit path.

## Impact

New orders are rejected. A full disk or a read-only path does not drop the write and report the order as accepted. Readiness stays closed until a later probe lands. This is not a price or a P&L figure.

## First 5 minutes

1. Read the halt code: enospc, read_only, or write_failed.
2. Confirm which of the four paths refused the byte.
3. Free space or remount the filesystem read-write. Do not delete the ledger to make the probe pass.
4. Confirm the next probe clears the latch and the previous digest is still the one from before the failure.
5. Leave the operator kill file latched if the book should stay closed after the disk recovers.

## Rollback and kill switch

The halt is the disk. Rolling back the image does not create free blocks. The operator kill file remains an independent latch. Clear it only after the probe write succeeds and the book is meant to accept orders again.
