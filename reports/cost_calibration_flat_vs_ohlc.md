# Cost calibration — flat half-spread vs OHLC estimators

SYNTHETIC execution diagnostic only. Not market evidence. `live_pnl_claim=false`. Research-headline return ratios and equity curves are absent by construction.

- schema: `cost_calibration.v1`
- seed: `7`
- half-spread floor (bps): `1.0`
- planted relative full spread: `0.002`
- Corwin–Schultz closed-form relative: `0.0008081198205963833`
- Corwin–Schultz effective half-spread bps (floored): `4.040599102981917`

## Trial results

```
      spread_estimator |  half_spread_bps_floor |             commission |                 spread |             total_cost |                n_fills
-----------------------+------------------------+------------------------+------------------------+------------------------+-----------------------
                  flat |                   1.00 |               625.5742 |               625.5742 |              1251.1483 |                     38
        corwin_schultz |                   1.00 |               625.1519 |              2437.6576 |              3062.8095 |                     38
          abdi_ranaldo |                   1.00 |               625.5742 |               625.5742 |              1251.1483 |                     38
                  roll |                   1.00 |               625.5663 |               739.1585 |              1364.7247 |                     37
```

## Notes

- Default path is flat `half_spread_bps`; calibrated estimators are opt-in.
- Calibrated cost is `max(floor, estimator_half_spread_bps)`.
- `run_backtest_fast` refuses any non-flat `spread_estimator`.
- Calibrated estimators: corwin_schultz, abdi_ranaldo, roll.
