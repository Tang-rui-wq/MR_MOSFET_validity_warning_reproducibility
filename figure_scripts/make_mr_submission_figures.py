from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT = Path(r"C:\Users\Tangrui\Desktop\MR_submission_latex")
HARDWARE = Path(r"C:\Users\Tangrui\Desktop\数据采集_Id零偏修正")
NASA_RESULTS = Path(r"D:\NASA\simulink\python_training\results")

RDS_HI_CSV = NASA_RESULTS / r"experiments\rds_delta_sensitivity\hi_metrics_by_delta.csv"
RDS_RUL_CSV = NASA_RESULTS / r"experiments\rds_delta_sensitivity\rul_metrics_by_delta.csv"
SIM_TEST14_CSV = NASA_RESULTS / r"simulink\Test_14_simulink_outputs.csv"
TEST34_PRIOR_CSV = NASA_RESULTS / r"latest_figures\origin_source_data\fig05_test34_prior.csv"

BASELINE_FILES = {
    "GRU": NASA_RESULTS / r"experiments\baseline\GRU_predictions.csv",
    "LSTM": NASA_RESULTS / r"experiments\baseline\LSTM_predictions.csv",
    "GRU+": NASA_RESULTS / r"experiments\baseline_credibility\GRU_128_180_predictions.csv",
    "LSTM+": NASA_RESULTS / r"experiments\baseline_credibility\LSTM_128_180_predictions.csv",
    "PG-RGRU": NASA_RESULTS / r"experiments\baseline\PINN_HI_GRU_predictions.csv",
}

SUMMARY_CSV = HARDWARE / "Id_zero_offset_correction_summary.csv"


mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.0,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


COLORS = {
    "black": "#222222",
    "blue": "#2E6F9E",
    "orange": "#C7782A",
    "green": "#3D7A57",
    "red": "#B85C5C",
    "gray": "#7A8490",
    "grid": "#DDE3EA",
}


def save_all(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def read_idcorr_csv(path: Path) -> tuple[dict[str, str], pd.DataFrame]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("TIME_s,"))
    metadata: dict[str, str] = {}
    for line in lines[:header_idx]:
        if "," in line:
            key, value = line.split(",", 1)
            metadata[key.strip()] = value.strip()
    df = pd.read_csv(path, skiprows=header_idx)
    return metadata, df


def shade_effective_windows(ax: plt.Axes, time_us: np.ndarray, mask: np.ndarray) -> None:
    if len(time_us) == 0:
        return
    mask = mask.astype(bool)
    starts = np.where(mask & np.r_[True, ~mask[:-1]])[0]
    ends = np.where(mask & np.r_[~mask[1:], True])[0]
    for s, e in zip(starts, ends):
        ax.axvspan(time_us[s], time_us[e], color="#E9ECEF", zorder=0)


def make_figure1() -> None:
    photo = Image.open(PROJECT / "Figure_1a.jpeg").convert("RGB")
    circuit = Image.open(PROJECT / "Figure_1b.png").convert("RGB")

    cropped_photo = photo.crop((90, 120, 1085, 780))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), gridspec_kw={"wspace": 0.05})
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(cropped_photo)
    axes[0].add_patch(Rectangle((300, 235), 415, 365, fill=False, lw=1.15, ec="#111111"))
    axes[1].imshow(circuit)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save_all(fig, PROJECT / "Figure_1", dpi=600)
    plt.close(fig)


def make_figure2() -> None:
    panels = [
        ("50% 0.5_IdCorr.csv", "Low-current record", "$I_{\\mathrm{d,corr}}$ median = 0.02 A"),
        ("50% 5.0_IdCorr.csv", "Measurable-current record", "$I_{\\mathrm{d,corr}}$ median = 0.62 A"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 4.15), sharex="col", gridspec_kw={"hspace": 0.16, "wspace": 0.22})
    series = [
        ("Vgs_V", "$V_{gs}$ (V)", COLORS["blue"]),
        ("Vds_V", "$V_{ds}$ (V)", COLORS["black"]),
        ("Id_corr_A", "$I_{\\mathrm{d,corr}}$ (A)", COLORS["green"]),
    ]
    for col, (fname, title, subtitle) in enumerate(panels):
        _, df = read_idcorr_csv(HARDWARE / fname)
        t = (df["TIME_s"].to_numpy() - df["TIME_s"].iloc[0]) * 1e6
        keep = (t >= 0) & (t <= 320)
        t = t[keep]
        sub = df.loc[keep].reset_index(drop=True)
        eff = sub["EffectiveOnState"].fillna(0).to_numpy() > 0.5
        for row, (name, ylabel, color) in enumerate(series):
            ax = axes[row, col]
            shade_effective_windows(ax, t, eff)
            ax.plot(t, sub[name].to_numpy(), lw=0.95, color=color)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
            if row == 0:
                ax.set_title(f"{title}\n{subtitle}", fontsize=8.4, fontweight="bold")
            if row < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time ($\\mu$s)")
            if row == 0:
                ax.set_ylim(-6, 22)
            if row == 2 and col == 0:
                ax.set_ylim(-0.08, 0.16)
            if row == 2 and col == 1:
                ax.set_ylim(-0.12, 0.82)
        panel_label(axes[0, col], "a" if col == 0 else "b")
    save_all(fig, PROJECT / "Figure_2", dpi=600)
    plt.close(fig)


def make_figure3() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    bus = summary[summary["corrected_file"].str.startswith("50%")].copy()
    rows = []
    for _, row in bus.iterrows():
        if not np.isfinite(row["id_corr_on_med_A"]):
            continue
        meta, df = read_idcorr_csv(HARDWARE / row["corrected_file"])
        thr = float(meta.get("Vgs_threshold_V", 7.5))
        gate_high = df["Vgs_V"].to_numpy() > thr
        denom = np.abs(df["Id_corr_A"].to_numpy())
        valid = gate_high & (denom > 1e-3)
        r_raw = np.abs(df.loc[valid, "Vds_V"].to_numpy()) / denom[valid]
        r_raw = r_raw[np.isfinite(r_raw)]
        r_raw = r_raw[(r_raw >= 0) & (r_raw <= 50)]
        if len(r_raw) == 0:
            continue
        rows.append(
            {
                "id_corr_A": float(row["id_corr_on_med_A"]),
                "median_ohm": float(np.median(r_raw)),
                "spread90_ohm": float(np.percentile(r_raw, 95) - np.percentile(r_raw, 5)),
            }
        )
    data = pd.DataFrame(rows).sort_values("id_corr_A")
    data.to_csv(PROJECT / "Figure_3_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.28})
    axes[0].plot(data["id_corr_A"], data["median_ohm"], marker="o", ms=3.5, lw=1.25, color=COLORS["blue"])
    axes[0].set_ylabel("Median raw $|V_{ds}|/|I_{\\mathrm{d,corr}}|$ ($\\Omega$)")
    axes[1].plot(data["id_corr_A"], data["spread90_ohm"], marker="^", ms=3.5, lw=1.25, color=COLORS["orange"])
    axes[1].set_ylabel("Raw 90% spread ($\\Omega$)")
    for i, ax in enumerate(axes):
        ax.axvline(0.10, ls="--", lw=0.9, color=COLORS["gray"])
        ax.text(0.105, 0.94, "$I_{\\min}=0.10$ A", transform=ax.get_xaxis_transform(), fontsize=7, va="top", color=COLORS["gray"])
        ax.set_xlabel("$I_{\\mathrm{d,corr}}$ (A)")
        ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
        panel_label(ax, "a" if i == 0 else "b")
    axes[0].set_title("Raw ratio median", fontsize=8.4, fontweight="bold")
    axes[1].set_title("Raw ratio dispersion", fontsize=8.4, fontweight="bold")
    save_all(fig, PROJECT / "Figure_3", dpi=600)
    plt.close(fig)


def make_figure4() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    bus = summary[summary["corrected_file"].str.startswith("50%") & summary["id_corr_on_med_A"].notna()].copy()
    duty = summary[summary["corrected_file"].str.startswith("4V") & summary["id_corr_on_med_A"].notna()].copy()
    bus["valid_cycle_ratio"] = bus["valid_cycles"] / bus["pwm_cycles"]
    duty["valid_cycle_ratio"] = duty["valid_cycles"] / duty["pwm_cycles"]

    source = pd.concat(
        [
            bus.assign(scan_group="50% bus"),
            duty.assign(scan_group="4 V duty"),
        ],
        ignore_index=True,
    )
    source.to_csv(PROJECT / "Figure_4_source.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    x = np.arange(len(summary))
    ax.scatter(x, summary["id_offset_A"], s=13, color=COLORS["red"], label="Raw off-state median")
    ax.scatter(x, np.zeros_like(x), s=13, color=COLORS["blue"], label="After zero-offset correction")
    ax.axhline(0, lw=0.7, color=COLORS["gray"])
    ax.set_xlabel("Record index")
    ax.set_ylabel("$I_d$ offset (A)")
    ax.set_title("Zero-offset correction", fontsize=8.4, fontweight="bold")
    ax.legend(loc="lower left", fontsize=6.3, handletextpad=0.5, borderaxespad=0.2)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
    panel_label(ax, "a")

    ax = axes[1]
    ax.plot(bus["id_corr_on_med_A"], bus["rds_cycle_cv"], marker="o", ms=3.5, lw=1.25, color=COLORS["blue"])
    ax.axvline(0.10, ls="--", lw=0.9, color=COLORS["gray"])
    ax.text(0.105, 0.95, "$I_{\\min}=0.10$ A", transform=ax.get_xaxis_transform(), fontsize=7, va="top", color=COLORS["gray"])
    ax.set_xlabel("$I_{\\mathrm{d,corr}}$ (A)")
    ax.set_ylabel("Filtered CV")
    ax.set_title("Repeatability boundary", fontsize=8.4, fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
    panel_label(ax, "b")

    ax = axes[2]
    ax.plot(bus["id_corr_on_med_A"], bus["valid_cycle_ratio"], marker="s", ms=3.3, lw=1.2, color=COLORS["green"], label="Bus scan")
    ax.plot(duty["id_corr_on_med_A"], duty["valid_cycle_ratio"], marker="^", ms=3.3, lw=1.2, color=COLORS["orange"], label="Duty scan")
    ax.axvline(0.10, ls="--", lw=0.9, color=COLORS["gray"])
    ax.set_xlabel("$I_{\\mathrm{d,corr}}$ (A)")
    ax.set_ylabel("Retained-cycle ratio")
    ax.set_ylim(0, 1.08)
    ax.set_title("Retained effective cycles", fontsize=8.4, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.7)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
    panel_label(ax, "c")

    save_all(fig, PROJECT / "Figure_4", dpi=600)
    plt.close(fig)


def first_crossing(x: np.ndarray, y: np.ndarray, threshold: float = 0.2) -> float | None:
    idx = np.where(y <= threshold)[0]
    if len(idx) == 0:
        return None
    return float(x[idx[0]])


def make_figure5() -> None:
    df = pd.read_csv(SIM_TEST14_CSV)
    t = df["elapsed_hours"].to_numpy()
    true = df["HI_label_true_out"].to_numpy()
    prior = df["HI_physics_prior_out"].to_numpy()
    py = df["HI_pred_pinn_out"].to_numpy()
    sim = df["HI_pred_feature_gru"].to_numpy()
    err = np.abs(py - sim)
    eol = first_crossing(t, py)
    source = pd.DataFrame(
        {
            "elapsed_hours": t,
            "HI_true": true,
            "HI_physics_prior": prior,
            "HI_python_PG_RGRU": py,
            "HI_simulink_PG_RGRU": sim,
            "abs_python_simulink_error": err,
        }
    )
    source.to_csv(PROJECT / "Figure_5_source.csv", index=False)

    step = max(1, len(df) // 4500)
    sl = slice(None, None, step)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.15), sharex=False, gridspec_kw={"hspace": 0.34})

    ax = axes[0]
    ax.plot(t[sl], true[sl], color=COLORS["black"], lw=1.15, label="Reference HI")
    ax.plot(t[sl], prior[sl], color=COLORS["gray"], lw=1.0, label="Physics prior")
    ax.plot(t[sl], py[sl], color=COLORS["orange"], lw=1.05, label="Python PG-RGRU")
    ax.plot(t[sl], sim[sl], color=COLORS["green"], lw=1.0, ls="--", label="Simulink PG-RGRU")
    ax.axhline(0.2, color=COLORS["red"], lw=1.0, ls="--", label="$HI=0.2$")
    if eol is not None:
        ax.axvline(eol, color=COLORS["gray"], lw=0.9, ls="--")
    ax.set_ylabel("HI")
    ax.set_title("Full replay", fontsize=8.4, fontweight="bold")
    ax.legend(ncol=3, fontsize=6.6, loc="lower left")
    panel_label(ax, "a")

    ax = axes[1]
    zoom = (t >= 18.0) & (t <= 22.5)
    ax.plot(t[zoom], true[zoom], color=COLORS["black"], lw=1.15)
    ax.plot(t[zoom], py[zoom], color=COLORS["orange"], lw=1.05)
    ax.plot(t[zoom], sim[zoom], color=COLORS["green"], lw=1.0, ls="--")
    ax.axhline(0.2, color=COLORS["red"], lw=1.0, ls="--")
    if eol is not None:
        ax.axvline(eol, color=COLORS["gray"], lw=0.9, ls="--")
    ax.set_ylabel("HI")
    ax.set_title("Warning-region zoom", fontsize=8.4, fontweight="bold")
    panel_label(ax, "b")

    ax = axes[2]
    ax.plot(t[sl], err[sl], color="#2F4B7C", lw=0.95)
    if eol is not None:
        ax.axvline(eol, color=COLORS["gray"], lw=0.9, ls="--")
    ax.set_ylabel("|Python - Simulink| HI")
    ax.set_xlabel("Elapsed aging time (h)")
    ax.set_title("Full implementation difference", fontsize=8.4, fontweight="bold")
    panel_label(ax, "c")

    for ax in axes:
        ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
        ax.set_ylim(bottom=-0.02)
    axes[0].set_ylim(-0.02, 1.06)
    axes[1].set_ylim(0.05, 0.42)
    axes[2].set_ylim(0, max(0.12, float(np.nanmax(err)) * 1.05))
    save_all(fig, PROJECT / "Figure_5", dpi=600)
    plt.close(fig)


def make_figure6() -> None:
    df = pd.read_csv(TEST34_PRIOR_CSV)
    df.columns = ["elapsed_hours", "HI_true", "HI_physics_prior", "HI_PG_RGRU"]
    df.to_csv(PROJECT / "Figure_6_source.csv", index=False)
    t = df["elapsed_hours"].to_numpy()
    pred = df["HI_PG_RGRU"].to_numpy()
    eol = first_crossing(t, pred)

    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.plot(t, df["HI_true"], lw=1.25, color=COLORS["black"], label="Reference HI")
    ax.plot(t, df["HI_physics_prior"], lw=1.05, color=COLORS["gray"], label="Physics prior")
    ax.plot(t, pred, lw=1.2, color=COLORS["green"], label="PG-RGRU")
    ax.axhline(0.2, color=COLORS["red"], lw=1.0, ls="--", label="$HI=0.2$")
    if eol is not None:
        ax.axvline(eol, color=COLORS["gray"], lw=0.9, ls="--")
    ax.set_xlabel("Elapsed aging time (h)")
    ax.set_ylabel("HI")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
    ax.legend(ncol=4, loc="upper center", fontsize=7)
    save_all(fig, PROJECT / "Figure_6", dpi=600)
    plt.close(fig)


def make_figure7() -> None:
    rows = []
    for model, path in BASELINE_FILES.items():
        df = pd.read_csv(path)
        test = df[df["split"].eq("test")].copy()
        for test_id, group in test.groupby("test_id"):
            rmse = float(np.sqrt(np.mean((group["HI_pred"] - group["HI_true"]) ** 2)))
            rows.append({"model": model, "test_id": int(test_id), "hi_rmse": rmse})
    data = pd.DataFrame(rows)
    data.to_csv(PROJECT / "Figure_7_source.csv", index=False)

    order = ["GRU", "LSTM", "GRU+", "LSTM+", "PG-RGRU"]
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    rng = np.random.default_rng(4)
    palette = {
        "GRU": "#9A9A9A",
        "LSTM": "#7A8490",
        "GRU+": "#6D8FB3",
        "LSTM+": "#88A7C5",
        "PG-RGRU": COLORS["green"],
    }
    for i, model in enumerate(order):
        vals = data.loc[data["model"].eq(model), "hi_rmse"].to_numpy()
        ax.boxplot(
            vals,
            positions=[i],
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "lw": 1.0},
            boxprops={"facecolor": "#F4F6F8", "edgecolor": palette[model], "lw": 0.9},
            whiskerprops={"color": palette[model], "lw": 0.9},
            capprops={"color": palette[model], "lw": 0.9},
        )
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, s=18, color=palette[model], alpha=0.92, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Per-test HI RMSE")
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
    save_all(fig, PROJECT / "Figure_7", dpi=600)
    plt.close(fig)


def make_figure8() -> None:
    hi = pd.read_csv(RDS_HI_CSV)
    rul = pd.read_csv(RDS_RUL_CSV)
    hi_test = hi[hi["scope"].eq("test")][["scenario", "rds_failure_delta", "rmse"]].copy()
    rul_test = rul[rul["scope"].eq("test")][
        ["scenario", "rds_failure_delta", "eol_mae_hours", "rul_mae_hours", "pred_crossed_tests", "n_tests"]
    ].copy()
    endpoint = hi_test.merge(rul_test, on=["scenario", "rds_failure_delta"]).sort_values("rds_failure_delta")

    current = pd.DataFrame(
        {
            "id_corr_A": [0.020, 0.100, 0.300, 0.620],
            "filtered_cv": [0.259, 0.039, 0.022, 0.016],
            "valid_ratio": [0.039, 0.800, 0.800, 0.800],
        }
    )
    endpoint.to_csv(PROJECT / "Figure_8a_endpoint_sensitivity_source.csv", index=False)
    current.to_csv(PROJECT / "Figure_8b_current_boundary_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    x = endpoint["rds_failure_delta"].to_numpy()
    ax.plot(x, endpoint["rmse"], marker="o", ms=4, linewidth=1.45, color=COLORS["blue"], label="HI RMSE")
    ax.set_xlabel("$\\Delta R_{ds,EOL}$")
    ax.set_ylabel("Test HI RMSE", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax.set_ylim(0.025, 0.055)
    ax.axvline(0.20919872437861223, color=COLORS["gray"], linestyle="--", linewidth=0.9)
    ax.text(0.211, 0.052, "calibrated", fontsize=7, rotation=90, va="top", ha="left", color=COLORS["gray"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, endpoint["eol_mae_hours"], marker="s", ms=4, linewidth=1.35, color=COLORS["orange"], label="EOL MAE")
    ax2.set_yscale("log")
    ax2.set_ylabel("EOL MAE (h, log)", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax2.set_ylim(0.015, 2.0)
    panel_label(ax, "a")
    ax.set_title("Endpoint sensitivity", fontsize=8.4, fontweight="bold")
    lines = [ax.lines[0], ax2.lines[0]]
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=6.8)

    ax = axes[1]
    x2 = current["id_corr_A"].to_numpy()
    ax.plot(x2, current["filtered_cv"], marker="o", ms=4, linewidth=1.45, color=COLORS["blue"], label="Filtered CV")
    ax.axvline(0.10, color=COLORS["gray"], linestyle="--", linewidth=0.9)
    ax.text(0.105, 0.277, "$I_{\\min}=0.10$ A", fontsize=7, va="top", color=COLORS["gray"])
    ax.set_xlabel("$I_{\\mathrm{d,corr}}$ (A)")
    ax.set_ylabel("Filtered CV", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax.set_ylim(0, 0.30)
    ax.set_xlim(0, 0.66)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)

    ax3 = ax.twinx()
    ax3.spines["right"].set_visible(True)
    ax3.plot(x2, current["valid_ratio"], marker="s", ms=4, linewidth=1.35, color=COLORS["green"], label="Valid ratio")
    ax3.set_ylabel("Valid ratio", color=COLORS["green"])
    ax3.tick_params(axis="y", labelcolor=COLORS["green"])
    ax3.set_ylim(0, 1.0)
    panel_label(ax, "b")
    ax.set_title("Current-boundary sensitivity", fontsize=8.4, fontweight="bold")
    lines = [ax.lines[0], ax3.lines[0]]
    ax.legend(lines, [line.get_label() for line in lines], loc="center right", fontsize=6.8)

    save_all(fig, PROJECT / "Figure_8", dpi=600)
    plt.close(fig)


def main() -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()
    make_figure6()
    make_figure7()
    make_figure8()


if __name__ == "__main__":
    main()
