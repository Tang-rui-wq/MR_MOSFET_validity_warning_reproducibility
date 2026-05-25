from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_data import load_all_samples
from run_nasa_mosfet_experiments import (
    ExperimentConfig,
    aggregate_eol,
    build_loaders,
    eol_rul_metrics,
    hi_metrics,
    predict_model,
    train_model,
)
from train_nasa_mosfet_hi_pinn_gru import (
    DEFAULT_RDS_FAILURE_DELTA,
    RESULTS_DIR,
    RUL_FAILURE_THRESHOLD,
    assign_paper_test_level_splits,
    apply_rds_failure_delta,
    calibrate_rds_failure_delta,
    drop_invalid_on_state_tests,
    prepare_hi_features,
    truncate_post_eol_windows,
)


OUT_DIR = RESULTS_DIR / "experiments" / "rds_delta_sensitivity"
CANDIDATE_DELTAS = [0.15, 0.20, DEFAULT_RDS_FAILURE_DELTA, 0.25, 0.30]


def run_one_delta(df_valid: pd.DataFrame, delta: float, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_delta = apply_rds_failure_delta(df_valid, delta)
    df_model, _ = truncate_post_eol_windows(df_delta)
    _, loaders, mu, sigma = build_loaders(df_model)
    config = ExperimentConfig(
        name=f"PINN_HI_GRU_{scenario}",
        family="rds_delta_sensitivity",
        cell="gru",
        residual_prior=True,
        physics_prior_loss=True,
        monotonic_loss=True,
        damage_ode_loss=True,
        rds_dynamic_weight=True,
        physics_prior_weight=0.240,
    )
    model, history = train_model(config, loaders)
    pred = predict_model(model, df_model, mu, sigma, config.name)
    hi_m = hi_metrics(pred)
    by_test, _ = eol_rul_metrics(pred)
    rul_m = aggregate_eol(by_test)
    for table in (hi_m, by_test, rul_m):
        table.insert(0, "scenario", scenario)
        table.insert(1, "rds_failure_delta", delta)
    pd.DataFrame(history).to_csv(OUT_DIR / f"{scenario}_history.csv", index=False)
    pred.to_csv(OUT_DIR / f"{scenario}_predictions.csv", index=False)
    return hi_m, rul_m, by_test


def plot_sensitivity(summary: pd.DataFrame) -> None:
    test = summary[summary["scope"] == "test"].sort_values("rds_failure_delta")
    fig, ax1 = plt.subplots(figsize=(8.0, 4.4))
    ax1.plot(test["rds_failure_delta"], test["hi_rmse"], marker="o", linewidth=2.0, label="HI RMSE")
    ax1.set_xlabel("Delta_Rds_EOL")
    ax1.set_ylabel("HI RMSE")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        test["rds_failure_delta"],
        test["eol_mae_hours"],
        marker="s",
        linewidth=2.0,
        color="tab:orange",
        label="EOL MAE (h)",
    )
    ax2.set_ylabel("EOL MAE (h)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title("Rds(on) EOL Delta Sensitivity on Test Split")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rds_delta_sensitivity.png", dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame, calibration: dict[str, object], excluded_tests: list[dict[str, float]]) -> None:
    test = summary[summary["scope"] == "test"].sort_values("hi_rmse")
    candidate_ids = [int(c["test_id"]) for c in calibration.get("candidates", [])]
    train_ids = [int(x) for x in calibration.get("calibration_train_tests", [])]
    lines = [
        "# Delta_Rds_EOL 训练集校准与敏感性验证",
        "",
        "## 校准方法",
        "",
        "仅使用论文主线训练集进行校准。先计算每个训练 Test 的在线 Rds(on) 相对退化量 95 分位值，然后取这些值的低四分位数作为失效归一化阈值。",
        "",
        "这样做的原因：NASA 部分样本会在失效后继续采集，末端 Rds 退化幅度差异很大；取低四分位数比取最大值或均值更保守，也避免被后失效长尾主导。验证集和测试集不参与 Delta_Rds_EOL 校准。",
        "",
        "```text",
        f"calibrated Delta_Rds_EOL = {float(calibration['rds_failure_delta']):.6f}",
        f"raw lower-quartile delta = {float(calibration['raw_delta']):.6f}",
        f"paper train tests = {len(train_ids)}",
        f"number of train candidates = {int(calibration['n_candidates'])}",
        f"candidate Test IDs = {candidate_ids}",
        "```",
        "",
        "审计结论：候选 Test 全部来自 paper train split，validation/test 中没有任何 Test 被用于阈值校准。",
        "",
        "## Test 集敏感性结果",
        "",
        test.to_markdown(index=False),
        "",
        "## 结论",
        "",
        "如果校准值附近的误差与 0.20/0.25 接近，说明模型对阈值不敏感；如果某个阈值明显更差，论文里需要说明该阈值不适合当前 NASA 数据。",
        "",
        "排除的 Rds(on) 不可信 Test 数量：",
        "",
        f"```text\n{len(excluded_tests)}\n```",
    ]
    (OUT_DIR / "RDS_DELTA_SENSITIVITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = prepare_hi_features(load_all_samples(), DEFAULT_RDS_FAILURE_DELTA)
    df_valid, excluded_tests = drop_invalid_on_state_tests(df_raw)
    df_valid = assign_paper_test_level_splits(df_valid)
    calibration = calibrate_rds_failure_delta(df_valid)
    train_ids = sorted(int(x) for x in df_valid.loc[df_valid["split"] == "train", "test_id"].unique())
    val_test_ids = sorted(int(x) for x in df_valid.loc[df_valid["split"].isin(["val", "test"]), "test_id"].unique())
    candidate_ids = sorted(int(c["test_id"]) for c in calibration.get("candidates", []))
    leakage = sorted(set(candidate_ids) & set(val_test_ids))
    if leakage:
        raise RuntimeError(f"Delta_Rds_EOL calibration leakage: candidates overlap val/test IDs {leakage}")
    calibration.update(
        {
            "method": "paper_train_split_lower_quartile_of_per_test_rds_resid_p95",
            "calibration_train_tests": train_ids,
            "validation_test_tests_excluded_from_calibration": val_test_ids,
            "audit": "PASS: all calibration candidates are in the paper train split.",
        }
    )
    pd.DataFrame(calibration.get("candidates", [])).assign(split="train").to_csv(
        OUT_DIR / "rds_delta_calibration_candidates.csv", index=False
    )
    calibrated_delta = float(calibration["rds_failure_delta"])

    deltas = CANDIDATE_DELTAS.copy()
    if not any(abs(d - calibrated_delta) < 1.0e-8 for d in deltas):
        deltas.append(calibrated_delta)
    deltas = sorted(deltas)

    all_hi = []
    all_rul = []
    all_by_test = []
    for delta in deltas:
        scenario = "calibrated" if abs(delta - calibrated_delta) < 1.0e-8 else f"delta_{delta:.2f}"
        print(f"Running sensitivity scenario {scenario}: Delta_Rds_EOL={delta:.6f}")
        hi_m, rul_m, by_test = run_one_delta(df_valid, delta, scenario)
        all_hi.append(hi_m)
        all_rul.append(rul_m)
        all_by_test.append(by_test)

    hi_all = pd.concat(all_hi, ignore_index=True)
    rul_all = pd.concat(all_rul, ignore_index=True)
    by_test_all = pd.concat(all_by_test, ignore_index=True)
    hi_all.to_csv(OUT_DIR / "hi_metrics_by_delta.csv", index=False)
    rul_all.to_csv(OUT_DIR / "rul_metrics_by_delta.csv", index=False)
    by_test_all.to_csv(OUT_DIR / "rul_by_test_by_delta.csv", index=False)

    hi_test = hi_all[hi_all["scope"].isin(["train", "val", "test"])].rename(
        columns={"mae": "hi_mae", "rmse": "hi_rmse"}
    )
    rul_test = rul_all[["scenario", "rds_failure_delta", "scope", "eol_mae_hours", "eol_rmse_hours", "rul_mae_hours"]]
    summary = hi_test.merge(rul_test, on=["scenario", "rds_failure_delta", "scope"], how="left")
    summary.to_csv(OUT_DIR / "rds_delta_sensitivity_summary.csv", index=False)
    plot_sensitivity(summary)
    write_report(summary, calibration, excluded_tests)

    print("Rds delta sensitivity finished.")
    print(summary[summary["scope"] == "test"].sort_values("hi_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
