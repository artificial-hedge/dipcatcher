# DipcatcherErrorRate

Research and simulated paper. This is not a live-trading incident.

## Trigger

Instrumented stages are recording any errors for 5 minutes. otel_export is excluded.

## Impact

The rate is above zero. The page-severity ratio alert is separate. A warning means errors exist, not that the budget is gone.

## First 5 minutes

1. List the stages in dipcatcher_errors_total.
2. Decide whether the errors are the same as DipcatcherErrorRatio.
3. Read a single log line for the stage.
4. Note the deploy that was running when the rate left zero.
5. Watch for five more minutes if the count is not climbing.

## Rollback and kill switch

A warning does not by itself require a rollback. Roll back when the errors track a new image. The kill file stays a separate decision.
