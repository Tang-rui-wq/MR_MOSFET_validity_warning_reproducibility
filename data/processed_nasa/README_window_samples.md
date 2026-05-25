# NASA MOSFET sliding-window sample table

Window length: 256 samples

Stride: 128 samples

## Files
- `mosfet_window_samples_all.csv`: all window-level samples with `split` column
- `mosfet_window_train_samples.csv`: train subset
- `mosfet_window_val_samples.csv`: validation subset
- `mosfet_window_test_samples.csv`: test subset

## Split plan
- Test_1 -> train
- Test_2 -> train
- Test_3 -> train
- Test_4 -> train
- Test_5 -> train
- Test_6 -> val
- Test_7 -> train
- Test_8 -> train
- Test_9 -> train
- Test_10 -> train
- Test_11 -> train
- Test_12 -> train
- Test_13 -> val
- Test_14 -> test
- Test_15 -> train
- Test_16 -> train
- Test_17 -> train
- Test_18 -> train
- Test_19 -> train
- Test_20 -> val
- Test_21 -> train
- Test_22 -> train
- Test_23 -> train
- Test_24 -> train
- Test_25 -> val
- Test_26 -> train
- Test_27 -> test
- Test_28 -> train
- Test_29 -> train
- Test_30 -> train
- Test_31 -> val
- Test_32 -> train
- Test_33 -> train
- Test_34 -> test
- Test_35 -> train
- Test_36 -> train
- Test_37 -> val
- Test_38 -> train
- Test_39 -> train
- Test_40 -> train
- Test_41 -> train
- Test_42 -> test

## Summary
- Total windows: 14819
- Train windows: 9747
- Val windows: 2444
- Test windows: 2628

## Label note
- `RUL_norm` is kept as a linear time-to-fail reference.
- `HI_label` and `Damage_label` are nonlinear labels built from robust on-state `Rds_on_cond_median`, temperature compensation, and `DamageProxy`.
