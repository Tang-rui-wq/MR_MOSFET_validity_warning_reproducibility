# NASA MOSFET HI / RUL labels

This file explains `mosfet_hi_rul_labels.csv`.

## Label definition
- `RUL_norm` is kept as a linear time-to-failure reference.
- `HI_label` is rebuilt as a nonlinear health indicator.
- `Damage_label` is rebuilt as a nonlinear damage indicator.
- The nonlinear labels are derived from `Rds_on_proxy` and `DamageProxy`.
- A convex damage law is used so health declines slowly at first and faster near failure.

## Summary
- Total rows: 52
- Number of Tests: 8
- HI range: [0.0000, 0.9976]
- Damage range: [0.0024, 1.0000]
- RUL(end) hours range: [0.0000, 131.3355]
