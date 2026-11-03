# DipcatcherSplitBrain

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_split_brain is 1 for 1 minute. A paper process failed to take the order lease, or its fencing token no longer matches the lock file.

## Impact

That process submits nothing. The process that still holds the lease may keep the simulated book. A second writer does not journal an order. This is not a price or a P&L figure.

## First 5 minutes

1. Read the acceptance code `split_brain` on the process that paged.
2. Confirm two processes were pointed at the same state directory.
3. Leave the lock file in place. A dead pid is not a reason to delete it.
4. Stop the extra process. The survivor, or a new process after the holder has exited, takes the flock and increments the token.
5. Confirm the new token is one greater and that the order id was accepted once.

## Rollback and kill switch

The lease is the fence. Rolling the image back does not make two writers safe. The operator kill file remains an independent latch. Clear it only when one process should submit again.
