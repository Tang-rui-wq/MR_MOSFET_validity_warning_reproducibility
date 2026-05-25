from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_nasa_mosfet_experiments as exp


OUT_DIR = exp.EXPERIMENT_DIR / "seed_repeat_pg_rgru"
SEEDS = [1, 7, 21, 42, 100]


def run_one_seed(seed: int, df: pd.DataFrame, loaders, mu: np.ndarray, sigma: np.ndarray) -> dict[str, float | int]:
    exp.SEED = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = exp.BASELINE_CONFIGS[-1]
    model, history = exp.train_model(config, loaders)
    pred = exp.predict_model(model, df, mu, sigma, f"PINN_HI_GRU_seed_{seed}")
    by_test, _ = exp.eol_rul_metrics(pred)
    hi = exp.hi_metrics(pred)
    rul = exp.aggregate_eol(by_test)

    pred.to_csv(OUT_DIR / f"PINN_HI_GRU_seed_{seed}_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / f"PINN_HI_GRU_seed_{seed}_training_history.csv", index=False)
    by_test.to_csv(OUT_DIR / f"PINN_HI_GRU_seed_{seed}_rul_by_test.csv", index=False)

    hi_test = hi[(hi["scope"] == "test")].iloc[0]
    rul_test = rul[(rul["scope"] == "test")].iloc[0]
    return {
        "seed": int(seed),
        "hi_mae": float(hi_test["mae"]),
        "hi_rmse": float(hi_test["rmse"]),
        "n_windows": int(hi_test["n_windows"]),
        "pred_crossed_tests": int(rul_test["pred_crossed_tests"]),
        "pred_cross_rate": float(rul_test["pred_cross_rate"]),
        "eol_mae_hours": float(rul_test["eol_mae_hours"]),
        "eol_rmse_hours": float(rul_test["eol_rmse_hours"]),
        "rul_mae_hours": float(rul_test["rul_mae_hours"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, excluded_tests, truncated_tests, calibration = exp.build_calibrated_modeling_dataframe()
    _, loaders, mu, sigma = exp.build_loaders(df)

    rows = []
    for seed in SEEDS:
        print(f"Running PG-RGRU seed repeat: seed={seed}")
        rows.append(run_one_seed(seed, df, loaders, mu, sigma))

    metrics = pd.DataFrame(rows)
    summary = metrics.agg(
        {
            "hi_mae": ["mean", "std", "min", "max"],
            "hi_rmse": ["mean", "std", "min", "max"],
            "pred_cross_rate": ["mean", "std", "min", "max"],
            "eol_mae_hours": ["mean", "std", "min", "max"],
            "rul_mae_hours": ["mean", "std", "min", "max"],
        }
    )
    metrics.to_csv(OUT_DIR / "pg_rgru_seed_repeat_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "pg_rgru_seed_repeat_summary.csv")

    report = {
        "seeds": SEEDS,
        "hidden_size": exp.HIDDEN_SIZE,
        "epochs": exp.EPOCHS,
        "rds_failure_delta": float(calibration["rds_failure_delta"]),
        "excluded_tests": excluded_tests,
        "post_eol_truncated_tests": truncated_tests,
        "metrics": rows,
        "summary": summary.to_dict(),
        "note": (
            "Seed repeat uses the same fixed test-level split, features, HI labels, "
            "Delta_Rds_EOL calibration and threshold rule as the manuscript. Only "
            "the PyTorch/NumPy random seed is changed before PG-RGRU training."
        ),
    }
    (OUT_DIR / "pg_rgru_seed_repeat_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# PG-RGRU Random-Seed Repeat",
        "",
        "This check repeats only the proposed PG-RGRU model with different random seeds.",
        "The train/validation/test split, feature set, HI labels, endpoint calibration and warning threshold are unchanged.",
        "",
        "## Per-seed metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Summary",
        "",
        summary.to_markdown(),
    ]
    (OUT_DIR / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(summary.to_string())


if __name__ == "__main__":
    main()
