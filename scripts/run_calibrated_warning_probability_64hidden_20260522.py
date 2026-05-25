from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_DIR = Path(r"D:\NASA\simulink\python_training\results\experiments\pg_rgru_warning_probability_64hidden_20260522")
OUT_DIR = Path(r"D:\NASA\simulink\python_training\results\experiments\pg_rgru_warning_probability_calibration_64hidden_20260522")
MR_DIR = Path(r"C:\Users\Tangrui\Desktop\MR_submission_latex_warning_probability_20260522")

MODEL_PREFIX = "MultiplierEventPGRGRU"
CANONICAL_MODEL = "MultiplierEventPGRGRU_seed42"
SELECTED_RULE = {
    "probability_threshold": 0.78,
    "persistence_windows": 10,
    "hi_guard": 0.80,
}


def first_time(t: np.ndarray, flag: np.ndarray) -> float:
    idx = np.flatnonzero(flag)
    return float(t[idx[0]]) if idx.size else float("nan")


def apply_decision(prob: np.ndarray, hi_pred: np.ndarray, threshold: float, persistence: int, hi_guard: float) -> np.ndarray:
    raw = (prob >= threshold) & (hi_pred <= hi_guard)
    confirmed = np.zeros_like(raw, dtype=bool)
    count = 0
    for i, value in enumerate(raw):
        count = count + 1 if value else 0
        if count >= persistence:
            confirmed[i] = True
    return confirmed


def decision_metrics(df: pd.DataFrame, threshold: float, persistence: int, hi_guard: float, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    out_parts = []
    for (model, split, test_id), sub in df.groupby(["model", "split", "test_id"], sort=True):
        one = sub.sort_values("window_id").copy()
        p = one["warning_prob"].to_numpy(dtype=np.float64)
        hi = one["HI_pred"].to_numpy(dtype=np.float64)
        y = one["warning_target"].to_numpy(dtype=np.float64) >= 0.5
        t = one["time_hours"].to_numpy(dtype=np.float64)
        decision = apply_decision(p, hi, threshold, persistence, hi_guard)
        target_time = first_time(t, y)
        trigger_time = first_time(t, decision)
        one[f"{label}_decision"] = decision.astype(int)
        one[f"{label}_raw"] = ((p >= threshold) & (hi <= hi_guard)).astype(int)
        one[f"{label}_target"] = y.astype(int)
        out_parts.append(one)
        rows.append(
            {
                "model": model,
                "seed": int(one["seed"].iloc[0]),
                "split": split,
                "test_id": int(test_id),
                "target_warning_present": bool(y.any()),
                "triggered": bool(decision.any()),
                "covered": bool((decision & y).any()) if y.any() else False,
                "missed": bool(y.any() and not (decision & y).any()),
                "false_before_target": bool(y.any() and np.any(decision & (t < target_time))),
                "target_warning_start_h": target_time,
                "decision_trigger_h": trigger_time,
                "lead_time_error_h": float(target_time - trigger_time)
                if np.isfinite(target_time) and np.isfinite(trigger_time)
                else np.nan,
                "false_warning_windows": int(np.count_nonzero(decision & ~y)),
                "non_target_windows": int(np.count_nonzero(~y)),
                "false_warning_window_rate": float(np.count_nonzero(decision & ~y) / max(1, np.count_nonzero(~y))),
                "raw_positive_rate": float(np.mean((p >= threshold) & (hi <= hi_guard))),
            }
        )
    return pd.DataFrame(rows), pd.concat(out_parts, ignore_index=True)


def aggregate_metrics(by_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split), sub in by_test.groupby(["model", "split"], sort=True):
        present = sub[sub["target_warning_present"]].copy()
        finite = present["lead_time_error_h"].dropna().to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": model,
                "seed": int(sub["seed"].iloc[0]),
                "split": split,
                "warning_covered_tests": int(present["covered"].sum()),
                "missed_warning_tests": int(present["missed"].sum()),
                "false_before_target_tests": int(present["false_before_target"].sum()),
                "false_warning_windows": int(present["false_warning_windows"].sum()) if not present.empty else 0,
                "non_target_windows": int(present["non_target_windows"].sum()) if not present.empty else 0,
                "false_warning_window_rate_overall": float(
                    present["false_warning_windows"].sum() / max(1, present["non_target_windows"].sum())
                )
                if not present.empty
                else np.nan,
                "false_warning_window_rate": float(present["false_warning_window_rate"].mean()) if not present.empty else np.nan,
                "lead_time_error_mae_h": float(np.mean(np.abs(finite))) if finite.size else np.nan,
                "lead_time_error_mean_h": float(np.mean(finite)) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def test_seed_summary(metrics: pd.DataFrame, label: str) -> pd.DataFrame:
    test = metrics[metrics["split"] == "test"].copy()
    rows = []
    for metric in [
        "warning_covered_tests",
        "missed_warning_tests",
        "false_before_target_tests",
        "false_warning_window_rate_overall",
        "false_warning_window_rate",
        "lead_time_error_mae_h",
        "lead_time_error_mean_h",
    ]:
        vals = test[metric].to_numpy(dtype=np.float64)
        rows.append(
            {
                "decision": label,
                "metric": metric,
                "n_seeds": int(vals.size),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "min": float(np.min(vals)) if vals.size else np.nan,
                "max": float(np.max(vals)) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def raw_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for (model, split, test_id), sub in df.groupby(["model", "split", "test_id"], sort=True):
        one = sub.sort_values("window_id").copy()
        one["raw_050_decision"] = (one["warning_prob"].to_numpy(dtype=np.float64) >= 0.5).astype(int)
        one["raw_050_target"] = (one["warning_target"].to_numpy(dtype=np.float64) >= 0.5).astype(int)
        parts.append(one)
    tmp = pd.concat(parts, ignore_index=True)
    return decision_metrics_raw(tmp, "raw_050_decision", "raw_050")


def decision_metrics_raw(df: pd.DataFrame, decision_col: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, split, test_id), sub in df.groupby(["model", "split", "test_id"], sort=True):
        one = sub.sort_values("window_id").copy()
        decision = one[decision_col].to_numpy(dtype=np.int64).astype(bool)
        y = one["warning_target"].to_numpy(dtype=np.float64) >= 0.5
        t = one["time_hours"].to_numpy(dtype=np.float64)
        target_time = first_time(t, y)
        trigger_time = first_time(t, decision)
        rows.append(
            {
                "model": model,
                "seed": int(one["seed"].iloc[0]),
                "split": split,
                "test_id": int(test_id),
                "target_warning_present": bool(y.any()),
                "triggered": bool(decision.any()),
                "covered": bool((decision & y).any()) if y.any() else False,
                "missed": bool(y.any() and not (decision & y).any()),
                "false_before_target": bool(y.any() and np.any(decision & (t < target_time))),
                "target_warning_start_h": target_time,
                "decision_trigger_h": trigger_time,
                "lead_time_error_h": float(target_time - trigger_time)
                if np.isfinite(target_time) and np.isfinite(trigger_time)
                else np.nan,
                "false_warning_windows": int(np.count_nonzero(decision & ~y)),
                "non_target_windows": int(np.count_nonzero(~y)),
                "false_warning_window_rate": float(np.count_nonzero(decision & ~y) / max(1, np.count_nonzero(~y))),
                "decision": label,
            }
        )
    return pd.DataFrame(rows), df


def make_figure(decisions: pd.DataFrame, raw_by_test: pd.DataFrame, calibrated_by_test: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    fig = plt.figure(figsize=(7.05, 4.05))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], width_ratios=[1.38, 0.92], hspace=0.38, wspace=0.36)
    ax_hi = fig.add_subplot(gs[0, 0])
    ax_prob = fig.add_subplot(gs[1, 0], sharex=ax_hi)
    ax_sum = fig.add_subplot(gs[:, 1])

    d34 = decisions[(decisions["model"] == CANONICAL_MODEL) & (decisions["test_id"] == 34)].sort_values("window_id").copy()
    focus = (d34["time_hours"] >= 41.55) & (d34["time_hours"] <= 42.02)
    d34f = d34[focus]
    target = d34f["warning_target"].to_numpy(dtype=np.float64) >= 0.5
    decision = d34f["calibrated_decision"].to_numpy(dtype=np.int64).astype(bool)
    t = d34f["time_hours"].to_numpy(dtype=np.float64)
    target_start = first_time(t, target)
    decision_start = first_time(t, decision)

    for ax in (ax_hi, ax_prob):
        if np.isfinite(target_start):
            ax.axvspan(target_start, t[-1], color="#EAF1F8", zorder=0)
        if np.isfinite(decision_start):
            ax.axvline(decision_start, color="#1F77B4", linewidth=0.9, linestyle=(0, (2.5, 2.5)), zorder=2)
        ax.grid(axis="y", color="#D9E0E8", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_hi.plot(t, d34f["HI_true"], color="#2B2B2B", linewidth=1.45, label="Causal reference HI")
    ax_hi.plot(t, d34f["HI_pred"], color="#1F77B4", linewidth=1.35, label="PG-RGRU HI")
    ax_hi.axhline(0.20, color="#687385", linewidth=0.8, linestyle=(0, (1.2, 1.7)))
    ax_hi.text(t[1], 0.225, "HI = 0.2", color="#56606E", fontsize=7)
    ax_hi.set_ylim(-0.03, 0.92)
    ax_hi.set_ylabel("HI")
    ax_hi.set_title("a  Test 34 causal HI near warning", loc="left", fontsize=8.2, fontweight="bold")
    ax_hi.legend(loc="upper right", frameon=False, fontsize=6.8)

    ax_prob.plot(t, d34f["warning_prob"], color="#D55E00", linewidth=1.45, label="Warning probability")
    ax_prob.axhline(SELECTED_RULE["probability_threshold"], color="#8C4A17", linewidth=0.8, linestyle=(0, (3, 2)))
    ax_prob.fill_between(t, 0, decision.astype(float), step="post", color="#1F77B4", alpha=0.20, label="Confirmed trigger")
    ax_prob.set_ylim(-0.04, 1.05)
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xlabel("Elapsed aging time (h)")
    ax_prob.set_title("b  Calibrated probability decision", loc="left", fontsize=8.2, fontweight="bold")
    ax_prob.legend(loc="lower right", frameon=False, fontsize=6.8)

    raw_summary = test_seed_summary(aggregate_metrics(raw_by_test), "Raw p>=0.5")
    cal_summary = test_seed_summary(aggregate_metrics(calibrated_by_test), "Calibrated")
    summary = pd.concat([raw_summary, cal_summary], ignore_index=True)
    false_vals = summary[summary["metric"] == "false_warning_window_rate_overall"].set_index("decision")
    cover_vals = summary[summary["metric"] == "warning_covered_tests"].set_index("decision")

    false_rates = [100 * false_vals.loc["Raw p>=0.5", "mean"], 100 * false_vals.loc["Calibrated", "mean"]]
    false_err = [100 * false_vals.loc["Raw p>=0.5", "std"], 100 * false_vals.loc["Calibrated", "std"]]
    cover_counts = [cover_vals.loc["Raw p>=0.5", "mean"], cover_vals.loc["Calibrated", "mean"]]
    colors = ["#8C98A5", "#1F77B4"]
    ax_sum.hlines(0, false_rates[1], false_rates[0], color="#C7D0DA", linewidth=3.0, zorder=1)
    ax_sum.errorbar(
        false_rates,
        [0, 0],
        xerr=false_err,
        fmt="o",
        color="#1F2933",
        ecolor="#1F2933",
        markersize=0,
        elinewidth=0.9,
        capsize=2,
        zorder=2,
    )
    ax_sum.scatter(false_rates, [0, 0], s=[42, 52], color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax_sum.annotate(
        "",
        xy=(false_rates[1] + 0.25, 0),
        xytext=(false_rates[0] - 0.25, 0),
        arrowprops=dict(arrowstyle="-|>", color="#1F77B4", lw=1.0, shrinkA=0, shrinkB=0),
        zorder=2,
    )
    ax_sum.text(false_rates[0], 0.11, f"Raw {false_rates[0]:.1f}%", ha="center", va="bottom", fontsize=7.0, color="#4C5561")
    ax_sum.text(false_rates[1], -0.11, f"Calibrated {false_rates[1]:.1f}%", ha="center", va="top", fontsize=7.0, color="#1F77B4")
    ax_sum.set_xlim(0, max(false_rates) + 1.2)
    ax_sum.set_ylim(-0.42, 0.52)
    ax_sum.set_yticks([])
    ax_sum.set_xlabel("False warning windows (%)")
    ax_sum.grid(axis="x", color="#D9E0E8", linewidth=0.45)
    ax_sum.spines["top"].set_visible(False)
    ax_sum.spines["right"].set_visible(False)
    ax_sum.spines["left"].set_visible(False)
    ax_sum.set_title("c  Test-set warning decision", loc="left", fontsize=8.2, fontweight="bold")
    ax_sum.text(
        0.02,
        0.74,
        f"coverage: {cover_counts[1]:.0f}/7\nmissed warnings: 0/7",
        transform=ax_sum.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#3E4652",
    )
    ax_sum.text(
        0.02,
        0.98,
        f"p >= {SELECTED_RULE['probability_threshold']:.2f}; "
        f"{SELECTED_RULE['persistence_windows']} windows; "
        f"HI <= {SELECTED_RULE['hi_guard']:.2f}",
        transform=ax_sum.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color="#3E4652",
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.12)
    base = OUT_DIR / "Figure_calibrated_warning_decision"
    for ext in ("pdf", "svg", "png", "tiff"):
        kwargs = {"dpi": 600} if ext in {"png", "tiff"} else {}
        fig.savefig(base.with_suffix(f".{ext}"), **kwargs)
        fig.savefig(MR_DIR / f"Figure_candidate_calibrated_warning_decision.{ext}", **kwargs)
    fig.savefig(base.with_name(base.name + "_review").with_suffix(".png"), dpi=220)
    fig.savefig(MR_DIR / "Figure_candidate_calibrated_warning_decision_review.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(SOURCE_DIR / "predictions_all.csv")
    pred = pred[pred["model"].astype(str).str.startswith(MODEL_PREFIX)].copy()

    raw_by_test, raw_predictions = raw_metrics(pred)
    calibrated_by_test, decisions = decision_metrics(
        pred,
        SELECTED_RULE["probability_threshold"],
        SELECTED_RULE["persistence_windows"],
        SELECTED_RULE["hi_guard"],
        "calibrated",
    )
    raw_metrics_agg = aggregate_metrics(raw_by_test)
    calibrated_metrics_agg = aggregate_metrics(calibrated_by_test)
    summary = pd.concat(
        [
            test_seed_summary(raw_metrics_agg, "Raw p>=0.5"),
            test_seed_summary(calibrated_metrics_agg, "Calibrated"),
        ],
        ignore_index=True,
    )

    raw_by_test.to_csv(OUT_DIR / "raw_warning_by_test.csv", index=False)
    calibrated_by_test.to_csv(OUT_DIR / "calibrated_warning_by_test.csv", index=False)
    raw_metrics_agg.to_csv(OUT_DIR / "raw_warning_metrics_by_seed.csv", index=False)
    calibrated_metrics_agg.to_csv(OUT_DIR / "calibrated_warning_metrics_by_seed.csv", index=False)
    summary.to_csv(OUT_DIR / "calibrated_warning_summary.csv", index=False)
    decisions.to_csv(OUT_DIR / "calibrated_warning_predictions_all.csv", index=False)
    decisions[decisions["model"] == CANONICAL_MODEL].to_csv(MR_DIR / "Figure_candidate_calibrated_warning_decision_source.csv", index=False)
    (OUT_DIR / "selected_rule.json").write_text(json.dumps(SELECTED_RULE, indent=2), encoding="utf-8")

    make_figure(decisions, raw_by_test, calibrated_by_test)

    print("Saved to:", OUT_DIR)
    print("\nSelected rule")
    print(json.dumps(SELECTED_RULE, indent=2))
    print("\nTest seed summary")
    print(summary.to_string(index=False))
    print("\nCalibrated test by seed")
    print(calibrated_metrics_agg[calibrated_metrics_agg["split"] == "test"].to_string(index=False))
    print("\nCandidate figure copied to:", MR_DIR / "Figure_candidate_calibrated_warning_decision.pdf")


if __name__ == "__main__":
    main()
