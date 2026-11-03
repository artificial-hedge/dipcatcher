# DipcatcherErrorBudgetExhausted

Research and simulated paper. This is not a live-trading incident.

## Trigger

The 30 day HTTP error budget remaining is below zero for 5 minutes.

## Impact

The 99.9 percent availability objective for this window is already spent. Further 5xx responses keep it negative.

## First 5 minutes

1. Read dipcatcher:slo:error_budget_remaining_30d.
2. Identify the hours that spent it from the error ratio rules.
3. Stop a rollout that is still adding 5xx.
4. Keep the previous image if it was the last quiet one.
5. Write down that the budget is an HTTP objective, not a market result.

## Rollback and kill switch

make -f deploy/prod.mk prod-rollback is the image step. The operator kill file blocks new orders while the budget is exhausted and the cause is still in the path.
