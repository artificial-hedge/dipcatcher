# DipcatcherTargetDown

Research and simulated paper. This is not a live-trading incident.

## Trigger

up is 0 for a lab-api or research-api target for 1 minute.

## Impact

Prometheus cannot scrape that replica. The replica may be stopped, failing its healthcheck, or refusing the scrape.

## First 5 minutes

1. Match the down target to lab-a, lab-b, research-a, or research-b.
2. Read that container's health and the last log lines.
3. Confirm the scrape token file is still mounted. Do not print the token.
4. If the sibling replica is up, leave traffic there.
5. If the pair is down, roll back or restart that pair.

## Rollback and kill switch

make -f deploy/prod.mk prod-rollback recreates the previous image. A canary that already aborted should have done this when auto-rollback is on.
