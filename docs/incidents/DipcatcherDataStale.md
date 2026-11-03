# DipcatcherDataStale

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_data_freshness_seconds is at least 86400 for 5 minutes.

## Impact

Market data used by the lab is older than one day. Research output that depends on that feed is stale. This is not a price or a P&L figure.

## First 5 minutes

1. Confirm the freshness gauge and which job should have updated it.
2. Check the collector log for a failed pull.
3. See whether the host clock jumped.
4. Do not fill the gap with a hand-written timestamp.
5. Leave orders blocked if the book would trade on that feed.

## Rollback and kill switch

Roll back only when a new image broke the collector. Otherwise fix the pull. The operator kill file blocks new orders until the feed is fresh again.
