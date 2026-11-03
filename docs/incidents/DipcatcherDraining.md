# DipcatcherDraining

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_draining == 1 for 1 minute.

## Impact

A replica is in the SIGTERM drain. The proxy removes it from the pool while the other replica is healthy. A drain that lasts means the process did not finish stopping.

## First 5 minutes

1. Confirm which replica publishes dipcatcher_draining 1.
2. See whether a deploy, restart, or rollback is in progress.
3. Read that replica's log for a stuck request or a lock.
4. If the other replica is ready, leave traffic there.
5. If both replicas are draining, stop the rollout and roll back.

## Rollback and kill switch

make -f deploy/prod.mk prod-rollback recreates the previous image. Do not clear the operator kill file just to end a drain.
