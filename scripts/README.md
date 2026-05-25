# NASA MOSFET Python Training

当前只保留一条主线：PINN-HI GRU 健康指标预测。

## 主流程

```text
清洗后的 NASA 窗口样本
-> 有效导通区 Rds(on) 清洗
-> 温度补偿后的电参数退化型 HI_label
-> PINN-HI GRU 训练
-> 导出 .mat 权重
-> Simulink 调用 HI_pred_gru
```

## 运行

在 VSCode 中运行任务：

```text
HI: full train export diagnose
```

或者在终端运行：

```powershell
cd .\python_training
.\venv\Scripts\python.exe train_nasa_mosfet_hi_pinn_gru.py
.\venv\Scripts\python.exe export_hi_gru_to_mat.py
.\venv\Scripts\python.exe run_nasa_mosfet_experiments.py
.\venv\Scripts\python.exe run_nasa_mosfet_early_rul_experiment.py
```

## 当前主线输出

- `results/hi_pinn_gru_comparison_metrics.csv`
- `results/nasa_mosfet_hi_pinn_gru_predictions.csv`
- `results/latest_figures`
- `results/test/summary_true_vs_pred.png`
- `results/train/per_test`、`results/val/per_test`、`results/test/per_test`
- `weights/nasa_mosfet_hi_pinn_gru_weights.mat`

## 说明

`HI_label` 不再混入 `RUL_norm`。当前论文主线采用电参数退化型 HI：

```text
D_Rds = max(Rds_tc / Rds_ref - 1, 0)
Damage_Rds = clip(D_Rds / Delta_Rds_EOL, 0, 1)
HI_label = 1 - Damage_Rds
```

其中 `Rds_tc` 是经过有效导通区筛选、假尖峰抑制和温度补偿后的导通电阻退化量。`Delta_Rds_EOL` 由训练集有效 Test 的 Rds 退化分布校准得到，主线值为 `0.209199`，并已完成 `0.15/0.20/0.209199/0.25/0.30` 敏感性验证。GRU 输入不包含 `HI_proxy_end` 或 `DamageProxy_end`，避免直接公式复现或标签泄漏。

不满足 Rds(on) 可观测退化条件的 Test 会被过滤并写入 `results/nasa_mosfet_hi_pinn_gru_metrics.json` 的 `quality_filter.excluded_tests`。

Simulink 相关文件已经整理到同级目录：

```text
../matlab_simulink
```

Simulink 推理函数使用与 Python 训练一致的滑动窗口统计口径，不再直接用瞬时采样量替代训练特征。

## 早期 RUL 专用特征

早期 RUL 实验位于：

```text
results/experiments/early_prediction
```

脚本 `run_nasa_mosfet_early_rul_experiment.py` 会同时跑两套输入：

- `basic_higru`：主 HI-GRU 的 10 个输入特征。
- `early_rul_expanded`：额外加入温度波动、功率峰值、Rds 波动、DeltaT、热累积量、早期 Rds 累计最大值和早期 Rds 趋势斜率。

当前测试集结果显示，扩展特征改善了 30% 和 70% 前缀 RUL，但 50% 前缀变差，因此 early RUL 只能作为补充实验，不应作为论文主打创新点。
