from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch

import run_nasa_mosfet_experiments as exp
from train_nasa_mosfet_hi_pinn_gru import (
    LABEL_COLUMN,
    PINN_FEATURE_COLUMNS,
    RESIDUAL_LIMIT,
)


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
OUT_DIR = exp.EXPERIMENT_DIR / "simulink_64hidden_export"
PT_NAMED = WEIGHTS_DIR / "nasa_mosfet_hi_pinn_gru_64hidden.pt"
MAT_NAMED = WEIGHTS_DIR / "nasa_mosfet_hi_pinn_gru_64hidden_weights.mat"
PT_DEFAULT = WEIGHTS_DIR / "nasa_mosfet_hi_pinn_gru.pt"
MAT_DEFAULT = WEIGHTS_DIR / "nasa_mosfet_hi_pinn_gru_weights.mat"


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64)


def _backup_if_exists(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}_backup_before_64hidden_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def _export_mat(checkpoint: dict[str, object], mat_path: Path) -> None:
    state = checkpoint["state_dict"]
    payload = {
        "W_ih": _as_numpy(state["backbone.weight_ih_l0"]),
        "W_hh": _as_numpy(state["backbone.weight_hh_l0"]),
        "b_ih": _as_numpy(state["backbone.bias_ih_l0"]).reshape(-1, 1),
        "b_hh": _as_numpy(state["backbone.bias_hh_l0"]).reshape(-1, 1),
        "W1": _as_numpy(state["head.0.weight"]),
        "b1": _as_numpy(state["head.0.bias"]).reshape(-1, 1),
        "W2": _as_numpy(state["head.2.weight"]),
        "b2": _as_numpy(state["head.2.bias"]).reshape(-1, 1),
        "mu": np.asarray(checkpoint["mu"], dtype=np.float64).reshape(-1, 1),
        "sigma": np.asarray(checkpoint["sigma"], dtype=np.float64).reshape(-1, 1),
        "feature_columns": np.asarray(checkpoint["feature_columns"], dtype=object),
        "label_column": np.asarray([checkpoint.get("label_column", LABEL_COLUMN)], dtype=object),
        "input_size": np.asarray([[state["backbone.weight_ih_l0"].shape[1]]], dtype=np.float64),
        "hidden_size": np.asarray([[state["backbone.weight_hh_l0"].shape[1]]], dtype=np.float64),
        "residual_limit": np.asarray([[checkpoint["residual_limit"]]], dtype=np.float64),
        "rds_failure_delta": np.asarray([[checkpoint["rds_failure_delta"]]], dtype=np.float64),
    }
    sio.savemat(mat_path, payload)


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exp.SEED = 42
    torch.manual_seed(exp.SEED)
    np.random.seed(exp.SEED)

    df, excluded_tests, truncated_tests, calibration = exp.build_calibrated_modeling_dataframe()
    _, loaders, mu, sigma = exp.build_loaders(df)
    config = exp.BASELINE_CONFIGS[-1]
    if config.name != "PINN_HI_GRU" or not config.residual_prior:
        raise RuntimeError("Unexpected PG-RGRU experiment configuration")

    model, history = exp.train_model(config, loaders)
    pred = exp.predict_model(model, df, mu, sigma, config.name)
    by_test, _ = exp.eol_rul_metrics(pred)
    hi_metrics = exp.hi_metrics(pred)
    rul_metrics = exp.aggregate_eol(by_test)

    pred.to_csv(OUT_DIR / "PINN_HI_GRU_64hidden_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / "PINN_HI_GRU_64hidden_training_history.csv", index=False)
    hi_metrics.to_csv(OUT_DIR / "PINN_HI_GRU_64hidden_hi_metrics.csv", index=False)
    rul_metrics.to_csv(OUT_DIR / "PINN_HI_GRU_64hidden_rul_metrics.csv", index=False)
    by_test.to_csv(OUT_DIR / "PINN_HI_GRU_64hidden_rul_by_test.csv", index=False)

    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mu": mu,
        "sigma": sigma,
        "feature_columns": PINN_FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "residual_limit": RESIDUAL_LIMIT,
        "rds_failure_delta": float(calibration["rds_failure_delta"]),
        "hidden_size": exp.HIDDEN_SIZE,
        "epochs": exp.EPOCHS,
        "seed": exp.SEED,
        "model_source": "run_nasa_mosfet_experiments.py:PINN_HI_GRU",
        "excluded_tests": excluded_tests,
        "post_eol_truncated_tests": truncated_tests,
        "hi_metrics": hi_metrics.to_dict(orient="records"),
        "rul_metrics": rul_metrics.to_dict(orient="records"),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = {
        "pt_default_backup": str(_backup_if_exists(PT_DEFAULT, stamp)),
        "mat_default_backup": str(_backup_if_exists(MAT_DEFAULT, stamp)),
    }

    torch.save(checkpoint, PT_NAMED)
    torch.save(checkpoint, PT_DEFAULT)
    _export_mat(checkpoint, MAT_NAMED)
    _export_mat(checkpoint, MAT_DEFAULT)

    summary = {
        "status": "exported_64hidden_pg_rgru_for_simulink",
        "named_pt": str(PT_NAMED),
        "named_mat": str(MAT_NAMED),
        "default_pt": str(PT_DEFAULT),
        "default_mat": str(MAT_DEFAULT),
        "backups": backups,
        "hidden_size": exp.HIDDEN_SIZE,
        "epochs": exp.EPOCHS,
        "seed": exp.SEED,
        "hi_test": hi_metrics[hi_metrics["scope"] == "test"].to_dict(orient="records"),
        "rul_test": rul_metrics[rul_metrics["scope"] == "test"].to_dict(orient="records"),
    }
    (OUT_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
