# NASA MOSFET cleaned dataset (detailed)

原始目录：`D:\NASA\MOSFET_Thermal_Overstress_Aging\data_raw\MOSFET_Thermal_Overstress_Aging_v0`

## 为什么要清洗
- 去掉异常文件，避免错误样本直接污染寿命标签和特征统计。
- 确认同一个 `Test` 内 `run` 顺序和时间单调，保证后续 `RUL_true = t_fail - t_k` 有物理意义。
- 统一读取字段，只保留包含 `supplyVoltage/packageTemperature/drainSourceVoltage/drainCurrent/flangeTemperature` 的可用样本。
- 提前筛出适合第一版论文训练的完整退化轨迹，减少后续实验反复返工。

## 当前输出文件
- `mosfet_manifest_detailed.csv`：逐文件清洗明细
- `mosfet_test_quality_summary.csv`：逐 Test 质量汇总
- `mosfet_selected_tests.csv`：第一版建议训练 Test

## 关键规则
1. `test_rampup.mat` 只作参考，不进主训练集。
2. `Test_13_run_1 1 (error setting up this run).mat` 和 `Test_25_run_4.mat` 直接排除。
3. 文件小于 1 MB 的 `.mat` 视为异常候选，默认排除。
4. 若 `steadyState` 缺失、关键字段缺失、单个 run 内时间不单调，也排除。
5. 第一版训练只优先选：run 数较多、run_id 连续、跨 run 时间单调的 Test。

## 推荐进入第一版训练的 Test
- Test_1：kept_runs=2，run_ids=1,2
- Test_2：kept_runs=3，run_ids=1,2,3
- Test_3：kept_runs=1，run_ids=1
- Test_4：kept_runs=2，run_ids=1,2
- Test_5：kept_runs=2，run_ids=1,2
- Test_6：kept_runs=2，run_ids=1,2
- Test_7：kept_runs=1，run_ids=1
- Test_8：kept_runs=7，run_ids=1,2,3,4,5,6,7
- Test_9：kept_runs=7，run_ids=1,2,3,4,5,6,7
- Test_10：kept_runs=7，run_ids=1,2,3,4,5,6,7
- Test_11：kept_runs=7，run_ids=1,2,3,4,5,6,7
- Test_12：kept_runs=7，run_ids=1,2,3,4,5,6,7
- Test_13：kept_runs=5，run_ids=1,2,3,4,5
- Test_14：kept_runs=7，run_ids=1,2,3,4,5,6,7
- Test_15：kept_runs=1，run_ids=1
- Test_16：kept_runs=1，run_ids=1
- Test_17：kept_runs=1，run_ids=1
- Test_18：kept_runs=2，run_ids=1,2
- Test_19：kept_runs=2，run_ids=1,2
- Test_20：kept_runs=2，run_ids=1,2
- Test_21：kept_runs=1，run_ids=1
- Test_22：kept_runs=1，run_ids=1
- Test_23：kept_runs=2，run_ids=1,2
- Test_24：kept_runs=2，run_ids=1,2
- Test_25：kept_runs=3，run_ids=1,2,3
- Test_26：kept_runs=1，run_ids=1
- Test_27：kept_runs=5，run_ids=1,2,3,4,5
- Test_28：kept_runs=2，run_ids=1,2
- Test_29：kept_runs=1，run_ids=1
- Test_30：kept_runs=1，run_ids=1
- Test_31：kept_runs=1，run_ids=1
- Test_32：kept_runs=1，run_ids=1
- Test_33：kept_runs=1，run_ids=1
- Test_34：kept_runs=3，run_ids=1,2,3
- Test_35：kept_runs=1，run_ids=1
- Test_36：kept_runs=1，run_ids=1
- Test_37：kept_runs=1，run_ids=1
- Test_38：kept_runs=1，run_ids=1
- Test_39：kept_runs=1，run_ids=1
- Test_40：kept_runs=2，run_ids=1,2
- Test_41：kept_runs=1，run_ids=1
- Test_42：kept_runs=1，run_ids=1

## 汇总
- 总 Test 数：42
- 推荐 Test 数：42
- 不推荐原因主要包括：too_few_runs、run_id_gap、time_not_monotonic_across_runs、no_valid_runs
