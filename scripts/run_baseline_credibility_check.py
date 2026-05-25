from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_nasa_mosfet_experiments import (
    ExperimentConfig,
    SequenceModel,
    build_loaders,
    evaluate,
    hi_metrics,
    loss_for_config,
    predict_model,
)
from train_nasa_mosfet_hi_pinn_gru import (
    PINN_FEATURE_COLUMNS,
    RESULTS_DIR,
    build_calibrated_modeling_dataframe,
)


OUT_DIR = RESULTS_DIR / "experiments" / "baseline_credibility"
SEED = 42


def train_custom(
    config: ExperimentConfig,
    loaders,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = SequenceModel(in_dim=len(PINN_FEATURE_COLUMNS), config=config, hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _, _, x, y, phys, _, mask in loaders["train"]:
            pred = model(x, phys)
            loss = loss_for_config(pred, y, phys, mask, config)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 20 == 0:
            train_m = evaluate(model, loaders["train"])
            val_m = evaluate(model, loaders["val"])
            history.append(
                {
                    "model": config.name,
                    "hidden": hidden,
                    "epochs": epochs,
                    "lr": lr,
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "train_mae": train_m["mae"],
                    "train_rmse": train_m["rmse"],
                    "val_mae": val_m["mae"],
                    "val_rmse": val_m["rmse"],
                }
            )
            if val_m["rmse"] < best_val:
                best_val = val_m["rmse"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, excluded_tests, truncated_tests, calibration = build_calibrated_modeling_dataframe()
    _, loaders, mu, sigma = build_loaders(df)

    configs = [
        {
            "name": "GRU_128_180",
            "config": ExperimentConfig(name="GRU_128_180", family="credibility", cell="gru"),
            "hidden": 128,
            "epochs": 180,
            "lr": 8.0e-4,
            "weight_decay": 5.0e-5,
        },
        {
            "name": "LSTM_128_180",
            "config": ExperimentConfig(name="LSTM_128_180", family="credibility", cell="lstm"),
            "hidden": 128,
            "epochs": 180,
            "lr": 8.0e-4,
            "weight_decay": 5.0e-5,
        },
        {
            "name": "GRU_128_lr3e4",
            "config": ExperimentConfig(name="GRU_128_lr3e4", family="credibility", cell="gru"),
            "hidden": 128,
            "epochs": 180,
            "lr": 3.0e-4,
            "weight_decay": 5.0e-5,
        },
    ]

    all_metrics = []
    histories = []
    for spec in configs:
        print(f"Training credibility baseline: {spec['name']}")
        model, hist = train_custom(
            spec["config"],
            loaders,
            hidden=spec["hidden"],
            epochs=spec["epochs"],
            lr=spec["lr"],
            weight_decay=spec["weight_decay"],
        )
        pred = predict_model(model, df, mu, sigma, spec["name"])
        pred.to_csv(OUT_DIR / f"{spec['name']}_predictions.csv", index=False)
        m = hi_metrics(pred)
        for k, v in spec.items():
            if k != "config":
                m[k] = v
        all_metrics.append(m)
        histories.extend(hist)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(OUT_DIR / "hi_metrics.csv", index=False)
    pd.DataFrame(histories).to_csv(OUT_DIR / "training_history.csv", index=False)
    summary = {
        "purpose": "Check whether stronger pure data-driven LSTM/GRU baselines close the gap under the same split/features/labels.",
        "split_policy": "Same Test-level split and same Rds-HI label as the main experiment.",
        "excluded_tests": excluded_tests,
        "calibration": calibration,
        "configs": [{k: (asdict(v) if k == "config" else v) for k, v in spec.items()} for spec in configs],
        "test_metrics": metrics[metrics["scope"] == "test"].to_dict(orient="records"),
    }
    (OUT_DIR / "baseline_credibility_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(metrics[metrics["scope"] == "test"].sort_values("rmse").to_string(index=False))


if __name__ == "__main__":
    main()
