# NASA MOSFET run-level sample table

This file explains the run-level training table generated from:
- cleaned NASA MOSFET trajectories
- run-level HI/RUL labels
- Simulink-generated physical proxy features

## Files
- `mosfet_run_samples_all.csv`: all run-level samples with `split` column
- `mosfet_train_samples.csv`: train subset
- `mosfet_val_samples.csv`: validation subset
- `mosfet_test_samples.csv`: test subset

## Split plan
- Test_8 -> train
- Test_9 -> train
- Test_10 -> train
- Test_11 -> train
- Test_12 -> train
- Test_13 -> val
- Test_14 -> test
- Test_27 -> test

## Feature groups
- Raw electrical/thermal features: supplyVoltage, packageTemperature, drainSourceVoltage, drainCurrent, flangeTemperature, switchingFrequency, dutyCycle
- Simulink proxy features: P_proxy, Rds_on_proxy, DeltaT_pf, x1, x2, Tth_proxy, ThermalExposure, DegradationSlope, DamageProxy, HI_proxy
- Labels: HI_label, Damage_label, RUL_true_hours_*, RUL_norm, life_stage

## Table summary
- Total run samples: 52
- Train samples: 35
- Val samples: 5
- Test samples: 12
