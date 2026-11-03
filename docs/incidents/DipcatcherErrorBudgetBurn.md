# DipcatcherErrorBudgetBurn

Research and simulated paper. This is not a live-trading incident.

## Trigger

The fast burn is above 14.4 times the 0.1 percent budget on both the 5 minute and 1 hour windows, or the slow burn is above 6 times on both the 30 minute and 6 hour windows, for 2 minutes.

## Impact

HTTP 5xx responses are consuming the 30 day error budget quickly. Availability is a process objective, not a research score.

## First 5 minutes

1. Split the burn by the service label, lab or research.
2. See whether a deploy started the burn.
3. Check DipcatcherTargetDown and disk pressure before scaling anything.
4. If one replica is the source, keep it out of the pool.
5. If the burn continues, roll back the image that started it.

## Rollback and kill switch

make -f deploy/prod.mk prod-rollback restores the previous image. Trip the kill file when the burning path is still taking new orders.
