# DipcatcherFilesystemReadOnly

Research and simulated paper. This is not a live-trading incident.

## Trigger

node_filesystem_readonly is 1 for 1 minute on a non-temporary filesystem. DipcatcherDiskPressure does not cover this case: that rule requires the mount to be writable.

## Impact

New orders are rejected when the ledger, journal, rate store, or audit path sits on that mount. The order is not reported as accepted. DipcatcherWriteHalt is the write-path page. This page is the mount itself. This is not a price or a P&L figure.

## First 5 minutes

1. Name the mount from the node-exporter labels.
2. Confirm whether the remount was an operator action or a kernel fault.
3. Do not delete the ledger to clear the page.
4. Remount the filesystem read-write only when the block device is healthy.
5. Confirm the next probe clears the write-halt latch and this page goes quiet.

## Rollback and kill switch

Rolling the image back does not remount a filesystem. The operator kill file stays an independent latch. Clear it only after the mount is writable and the book is meant to accept orders again.
