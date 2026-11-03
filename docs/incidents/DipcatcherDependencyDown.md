# DipcatcherDependencyDown

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_dependency_down is 1 for 1 minute. The paper dependency probe failed closed before an order was journaled. The code is dns, tls, broker_down, or broker_slow.

## Impact

No new intent line is written. A later healthy probe can accept the order once. A failed probe does not leave a second submit behind. This is not a price or a P&L figure.

## First 5 minutes

1. Read the acceptance code: dependency_dns, dependency_tls, dependency_broker_down, or dependency_broker_slow.
2. Confirm `DIPCATCHER_PAPER_BROKER_URL` and the timeout. Unset means the probe is off.
3. Restore DNS, present a certificate the process can verify, or bring the broker port back inside the timeout.
4. Confirm the gauge clears on the next successful probe and the journal has one intent for the order.
5. Leave the operator kill file latched if the book should stay closed after the dependency recovers.

## Rollback and kill switch

The probe is the gate. Rolling the image back does not repair DNS, a certificate, or the broker. The operator kill file remains an independent latch.
