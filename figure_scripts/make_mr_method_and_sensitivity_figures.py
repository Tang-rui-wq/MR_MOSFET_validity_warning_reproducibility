from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT = Path(r"C:\Users\Tangrui\Desktop\MR_submission_latex")
RDS_HI_CSV = Path(
    r"D:\NASA\simulink\python_training\results\experiments\rds_delta_sensitivity\hi_metrics_by_delta.csv"
)
RDS_RUL_CSV = Path(
    r"D:\NASA\simulink\python_training\results\experiments\rds_delta_sensitivity\rul_metrics_by_delta.csv"
)


mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


COLORS = {
    "neutral": "#3D4451",
    "muted": "#F3F5F7",
    "line": "#B9C2CB",
    "blue": "#2E6F9E",
    "blue_light": "#E6F0F7",
    "orange": "#C7782A",
    "orange_light": "#FAEBD9",
    "green": "#3D7A57",
    "green_light": "#E8F3EC",
    "red": "#B85C5C",
    "red_light": "#F8E7E7",
}


def save_all(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    face: str,
    edge: str,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.045, title, va="top", ha="left", weight="bold", color=edge, fontsize=7.4)
    ax.text(x + 0.018, y + h - 0.102, body, va="top", ha="left", color=COLORS["neutral"], fontsize=6.5, linespacing=1.22)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = COLORS["neutral"]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def make_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.955,
        "Measurement-validity-aware HI construction and warning pipeline",
        ha="left",
        va="top",
        fontsize=8.8,
        weight="bold",
        color=COLORS["neutral"],
    )

    w, h = 0.265, 0.205
    add_box(
        ax,
        (0.04, 0.63),
        (w, h),
        "Raw online samples",
        "$V_{ds}$, $I_d$, $V_{gs}$, temperature\nPWM state and window statistics",
        COLORS["muted"],
        COLORS["neutral"],
    )
    add_box(
        ax,
        (0.367, 0.63),
        (w, h),
        "Validity mask",
        "$s_{on}=1$, $I_{d,corr}\\geq I_{min}$\n$V_{ds}\\in\\mathcal{V}_{on}$",
        COLORS["orange_light"],
        COLORS["orange"],
    )
    add_box(
        ax,
        (0.694, 0.63),
        (w, h),
        "Corrected $R_{ds(on)}$",
        "zero-offset correction\ncentral on-state median\nand temperature compensation",
        COLORS["blue_light"],
        COLORS["blue"],
    )

    add_box(
        ax,
        (0.04, 0.31),
        (w, h),
        "HI label and prior",
        "$D_R=(R_{ds}/R_{ref})-1$\n$HI=1-clip(D_R/\\Delta R_{ds,EOL})$",
        COLORS["blue_light"],
        COLORS["blue"],
    )
    add_box(
        ax,
        (0.367, 0.31),
        (w, h),
        "Retained training windows",
        "$\\Omega_{train}$ contains only\nvalid effective-conduction windows",
        COLORS["green_light"],
        COLORS["green"],
    )
    add_box(
        ax,
        (0.694, 0.31),
        (w, h),
        "PG-RGRU warning",
        "$\\hat{H}=clip(H_{phy}+\\alpha\\tanh(g_\\theta),0,1)$\nwarning when $\\hat{H}\\leq0.2$",
        COLORS["green_light"],
        COLORS["green"],
    )

    add_box(
        ax,
        (0.367, 0.035),
        (0.265, 0.14),
        "Rejected artifacts",
        "low-current/off-state samples",
        COLORS["red_light"],
        COLORS["red"],
    )

    add_arrow(ax, (0.305, 0.733), (0.367, 0.733))
    add_arrow(ax, (0.632, 0.733), (0.694, 0.733))
    add_arrow(ax, (0.827, 0.63), (0.172, 0.515), color=COLORS["blue"])
    add_arrow(ax, (0.305, 0.413), (0.367, 0.413), color=COLORS["green"])
    add_arrow(ax, (0.632, 0.413), (0.694, 0.413), color=COLORS["green"])
    add_arrow(ax, (0.50, 0.63), (0.50, 0.175), color=COLORS["red"])

    save_all(fig, PROJECT / "Figure_0")
    plt.close(fig)


def make_sensitivity_figure() -> None:
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    x = endpoint["rds_failure_delta"].to_numpy()
    ax.plot(x, endpoint["rmse"], marker="o", linewidth=1.5, color=COLORS["blue"], label="HI RMSE")
    ax.set_xlabel("$\\Delta R_{ds,EOL}$")
    ax.set_ylabel("Test HI RMSE", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax.set_ylim(0.025, 0.055)
    ax.axvline(0.20919872437861223, color=COLORS["neutral"], linestyle="--", linewidth=0.9)
    ax.text(0.211, 0.052, "calibrated", fontsize=6.3, rotation=90, va="top", ha="left", color=COLORS["neutral"])
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.45)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(
        x,
        endpoint["eol_mae_hours"],
        marker="s",
        linewidth=1.3,
        color=COLORS["orange"],
        label="EOL MAE",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("EOL MAE (h, log)", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax2.set_ylim(0.015, 2.0)
    ax.text(-0.13, 1.05, "a", transform=ax.transAxes, fontsize=9, weight="bold")
    ax.set_title("Endpoint sensitivity", fontsize=7.5, weight="bold")

    lines = [ax.lines[0], ax2.lines[0]]
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=6.2)

    ax = axes[1]
    x2 = current["id_corr_A"].to_numpy()
    ax.plot(x2, current["filtered_cv"], marker="o", linewidth=1.5, color=COLORS["blue"], label="Filtered CV")
    ax.axvline(0.10, color=COLORS["neutral"], linestyle="--", linewidth=0.9)
    ax.text(0.105, 0.277, "$I_{min}=0.10$ A", fontsize=6.3, va="top", color=COLORS["neutral"])
    ax.set_xlabel("$I_{d,corr}$ (A)")
    ax.set_ylabel("Filtered CV", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax.set_ylim(0, 0.30)
    ax.set_xlim(0, 0.66)
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.45)

    ax3 = ax.twinx()
    ax3.spines["right"].set_visible(True)
    ax3.plot(
        x2,
        current["valid_ratio"],
        marker="s",
        linewidth=1.3,
        color=COLORS["green"],
        label="Valid ratio",
    )
    ax3.set_ylabel("Valid ratio", color=COLORS["green"])
    ax3.tick_params(axis="y", labelcolor=COLORS["green"])
    ax3.set_ylim(0, 1.0)
    ax.text(-0.13, 1.05, "b", transform=ax.transAxes, fontsize=9, weight="bold")
    ax.set_title("Current-boundary sensitivity", fontsize=7.5, weight="bold")

    lines = [ax.lines[0], ax3.lines[0]]
    ax.legend(lines, [line.get_label() for line in lines], loc="center right", fontsize=6.2)

    save_all(fig, PROJECT / "Figure_8")
    plt.close(fig)


def main() -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    make_sensitivity_figure()


if __name__ == "__main__":
    main()
