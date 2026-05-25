# NASA MOSFET cleaned index

数据原始目录：
D:\NASA\MOSFET_Thermal_Overstress_Aging\data_raw\MOSFET_Thermal_Overstress_Aging_v0

本目录用途：
- `mosfet_manifest.csv`：每个 `.mat` 文件的 test/run 索引
- `mosfet_test_summary.csv`：每个 Test 的文件数量和总体积汇总

当前清理规则：
- 文件名按 `Test_x_run_y.mat` 解析
- `test_rampup.mat` 单独标记为 `rampup_reference`
- 带 `error setting up this run` 的文件标记为 `setup_error_file`
- 小于 1 MB 的 `.mat` 标记为 `very_small_check`，后续加载时优先人工复核

建议下一步：
1. 先从 `test_id` 连续、文件数较完整、无异常标记的测试组入手
2. 用 MATLAB 或 Python 批量读取变量名，生成二级索引
3. 再定义 HI / RUL 标签和训练窗口
