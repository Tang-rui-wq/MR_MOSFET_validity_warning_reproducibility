from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from common_data import (
    RUN_ORDER_COLUMN,
    SAMPLE_ID_COLUMN,
    SPLIT_COLUMN,
    TEST_ID_COLUMN,
)
from train_nasa_mosfet_hi_pinn_gru import (
    LABEL_COLUMN,
    PINN_FEATURE_COLUMNS,
    PINN_PHYS_COLUMNS,
    RESIDUAL_LIMIT,
    RESULTS_DIR,
    RUL_FAILURE_THRESHOLD,
    build_calibrated_modeling_dataframe,
    elapsed_axis_hours,
    first_threshold_crossing_time,
)


EXPERIMENT_DIR = RESULTS_DIR / "experiments"
BASELINE_DIR = EXPERIMENT_DIR / "baseline"
ABLATION_DIR = EXPERIMENT_DIR / "ablation"
EARLY_DIR = EXPERIMENT_DIR / "early_prediction"

EPOCHS = 90
BATCH_SIZE = 8
HIDDEN_SIZE = 64
SEED = 42


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    family: str
    cell: str
    residual_prior: bool = False
    dynamic_residual_bound: bool = False
    physics_prior_loss: bool = False
    monotonic_loss: bool = False
    slope_loss: bool = False
    damage_ode_loss: bool = False
    rds_dynamic_weight: bool = False
    smoothness_loss: bool = True
    boundary_loss: bool = True
    threshold_weight: float = 0.0
    slope_weight: float = 0.050
    residual_alpha_min: float = 0.05
    residual_alpha_max: float = RESIDUAL_LIMIT
    physics_prior_weight: float = 0.220
    damage_ode_weight: float = 0.050


BASELINE_CONFIGS = [
    ExperimentConfig(name="MLP", family="baseline", cell="mlp"),
    ExperimentConfig(name="RNN", family="baseline", cell="rnn"),
    ExperimentConfig(name="LSTM", family="baseline", cell="lstm"),
    ExperimentConfig(name="GRU", family="baseline", cell="gru"),
    ExperimentConfig(
        name="PINN_HI_GRU",
        family="baseline",
        cell="gru",
        residual_prior=True,
        physics_prior_loss=True,
        monotonic_loss=True,
        damage_ode_loss=True,
        rds_dynamic_weight=True,
        physics_prior_weight=0.240,
    ),
]

ABLATION_CONFIGS = [
    BASELINE_CONFIGS[-1],
    ExperimentConfig(
        name="Abl_NoPhysicsPrior",
        family="ablation",
        cell="gru",
        residual_prior=False,
        physics_prior_loss=False,
        monotonic_loss=True,
        damage_ode_loss=True,
        rds_dynamic_weight=True,
    ),
    ExperimentConfig(
        name="Abl_NoMonotonic",
        family="ablation",
        cell="gru",
        residual_prior=True,
        physics_prior_loss=True,
        monotonic_loss=False,
        damage_ode_loss=True,
        rds_dynamic_weight=True,
        physics_prior_weight=0.240,
    ),
    ExperimentConfig(
        name="Abl_NoDamageODE",
        family="ablation",
        cell="gru",
        residual_prior=True,
        physics_prior_loss=True,
        monotonic_loss=True,
        damage_ode_loss=False,
        rds_dynamic_weight=True,
        physics_prior_weight=0.240,
    ),
        ExperimentConfig(
        name="Abl_NoRdsDynamicWeight",
        family="ablation",
        cell="gru",
        residual_prior=True,
        physics_prior_loss=True,
        monotonic_loss=True,
        damage_ode_loss=True,
        rds_dynamic_weight=False,
        physics_prior_weight=0.240,
    ),
]

PG_RGRU_V2_CONFIG = ExperimentConfig(
    name="PG_RGRU_V2",
    family="method_update",
    cell="gru",
    residual_prior=True,
    dynamic_residual_bound=True,
    physics_prior_loss=True,
    monotonic_loss=True,
    slope_loss=True,
    damage_ode_loss=True,
    rds_dynamic_weight=True,
    threshold_weight=1.5,
    slope_weight=0.050,
    residual_alpha_min=0.05,
    residual_alpha_max=0.30,
    physics_prior_weight=0.150,
    damage_ode_weight=0.050,
)


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray):
        self.samples = []
        for test_id, sub in df.groupby(TEST_ID_COLUMN, sort=True):
            sub = sub.sort_values(RUN_ORDER_COLUMN)
            x = ((sub[PINN_FEATURE_COLUMNS].to_numpy(dtype=np.float64) - mu) / sigma).astype(np.float32)
            y = sub[LABEL_COLUMN].to_numpy(dtype=np.float32)
            phys = sub[PINN_PHYS_COLUMNS].to_numpy(dtype=np.float32)
            sample_ids = sub[SAMPLE_ID_COLUMN].to_numpy(dtype=np.int64)
            split = str(sub[SPLIT_COLUMN].iloc[0])
            self.samples.append(
                (int(test_id), split, torch.tensor(x), torch.tensor(y), torch.tensor(phys), torch.tensor(sample_ids))
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_sequences(batch):
    test_ids, splits, xs, ys, phys, sample_ids = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=-1.0)
    phys_pad = pad_sequence(phys, batch_first=True)
    id_pad = pad_sequence(sample_ids, batch_first=True, padding_value=-1)
    mask = y_pad >= 0
    return torch.tensor(test_ids, dtype=torch.long), list(splits), x_pad, y_pad, phys_pad, id_pad, mask


class SequenceModel(nn.Module):
    def __init__(self, in_dim: int, config: ExperimentConfig, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.config = config
        self.cell = config.cell
        self.residual_prior = config.residual_prior
        if self.cell == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            head_in = hidden
        elif self.cell == "rnn":
            self.backbone = nn.RNN(input_size=in_dim, hidden_size=hidden, batch_first=True)
            head_in = hidden + in_dim
        elif self.cell == "lstm":
            self.backbone = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True)
            head_in = hidden + in_dim
        elif self.cell == "gru":
            self.backbone = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True)
            head_in = hidden + in_dim
        else:
            raise ValueError(f"Unsupported cell: {self.cell}")
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.alpha_head = None
        if config.dynamic_residual_bound:
            self.alpha_head = nn.Sequential(nn.Linear(head_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        if self.cell == "mlp":
            z = self.backbone(x)
        else:
            h, _ = self.backbone(x)
            z = torch.cat([h, x], dim=-1)
        raw = self.head(z).squeeze(-1)
        if self.residual_prior:
            if self.config.dynamic_residual_bound:
                alpha_raw = self.alpha_head(z).squeeze(-1)
                alpha = self.config.residual_alpha_min + (
                    self.config.residual_alpha_max - self.config.residual_alpha_min
                ) * torch.sigmoid(alpha_raw)
            else:
                alpha = RESIDUAL_LIMIT
            return torch.clamp(phys[..., 0] + alpha * torch.tanh(raw), 0.0, 1.0)
        return torch.sigmoid(raw)


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 0.050,
    threshold_weight: float = 0.0,
) -> torch.Tensor:
    target_m = target[mask]
    err = pred[mask] - target_m
    abs_err = torch.abs(err)
    huber = torch.where(abs_err <= delta, 0.5 * (err ** 2) / delta, abs_err - 0.5 * delta)
    weights = 1.0 + 1.8 * (1.0 - target_m)
    if threshold_weight > 0:
        threshold_band = torch.exp(-0.5 * ((target_m - 0.35) / 0.15) ** 2)
        weights = weights + threshold_weight * threshold_band
    return torch.mean(weights * huber)


def smoothness_penalty(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = []
    for i in range(pred.shape[0]):
        p = pred[i][mask[i]]
        if p.numel() > 2:
            vals.append(torch.mean((p[1:] - p[:-1]) ** 2))
    return torch.stack(vals).mean() if vals else pred.new_tensor(0.0)


def monotonic_penalty(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = []
    for i in range(pred.shape[0]):
        p = pred[i][mask[i]]
        if p.numel() > 2:
            vals.append(torch.mean(torch.relu(p[1:] - p[:-1]) ** 2))
    return torch.stack(vals).mean() if vals else pred.new_tensor(0.0)


def slope_penalty(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = []
    for i in range(pred.shape[0]):
        p = pred[i][mask[i]]
        y = target[i][mask[i]]
        if p.numel() <= 2:
            continue
        dp = p[1:] - p[:-1]
        dy = y[1:] - y[:-1]
        err = dp - dy
        abs_err = torch.abs(err)
        huber = torch.where(abs_err <= 0.025, 0.5 * (err ** 2) / 0.025, abs_err - 0.0125)
        warning_band = ((y[1:] >= 0.20) & (y[1:] <= 0.50)).to(dtype=huber.dtype)
        vals.append(torch.mean((1.0 + 2.0 * warning_band) * huber))
    return torch.stack(vals).mean() if vals else pred.new_tensor(0.0)


def boundary_penalty(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = []
    for i in range(pred.shape[0]):
        p = pred[i][mask[i]]
        y = target[i][mask[i]]
        if p.numel() > 0:
            vals.extend([(p[0] - y[0]) ** 2, (p[-1] - y[-1]) ** 2])
    return torch.stack(vals).mean() if vals else pred.new_tensor(0.0)


def _rds_weight(phys: torch.Tensor, mask: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return torch.ones_like(phys[..., 0][mask])
    rds_confidence = phys[..., 3][mask]
    return 0.30 + 0.70 * torch.clamp(rds_confidence, 0.0, 1.0)


def physics_prior_penalty(pred: torch.Tensor, phys: torch.Tensor, mask: torch.Tensor, dynamic_weight: bool) -> torch.Tensor:
    hi_prior = phys[..., 0]
    err = pred[mask] - hi_prior[mask]
    abs_err = torch.abs(err)
    huber = torch.where(abs_err <= 0.08, 0.5 * (err ** 2) / 0.08, abs_err - 0.04)
    return torch.mean(_rds_weight(phys, mask, dynamic_weight) * huber) if huber.numel() else pred.new_tensor(0.0)


def damage_ode_penalty(pred: torch.Tensor, phys: torch.Tensor, mask: torch.Tensor, dynamic_weight: bool) -> torch.Tensor:
    vals = []
    damage_phys = phys[..., 1]
    conf = phys[..., 3]
    for i in range(pred.shape[0]):
        p = pred[i][mask[i]]
        d = damage_phys[i][mask[i]]
        c = conf[i][mask[i]]
        if p.numel() <= 2:
            continue
        d_pred = (1.0 - p[1:]) - (1.0 - p[:-1])
        d_phys = d[1:] - d[:-1]
        jump_weight = 1.0 + 4.0 * torch.clamp(d_phys, min=0.0)
        if dynamic_weight:
            jump_weight = jump_weight * (0.30 + 0.70 * torch.clamp(c[1:], 0.0, 1.0))
        vals.append(torch.mean(jump_weight * (d_pred - d_phys) ** 2))
    return torch.stack(vals).mean() if vals else pred.new_tensor(0.0)


def loss_for_config(pred: torch.Tensor, y: torch.Tensor, phys: torch.Tensor, mask: torch.Tensor, config: ExperimentConfig):
    loss = masked_huber(pred, y, mask, threshold_weight=config.threshold_weight)
    if config.smoothness_loss:
        loss = loss + 0.002 * smoothness_penalty(pred, mask)
    if config.monotonic_loss:
        loss = loss + 0.025 * monotonic_penalty(pred, mask)
    if config.slope_loss:
        loss = loss + config.slope_weight * slope_penalty(pred, y, mask)
    if config.boundary_loss:
        loss = loss + 0.080 * boundary_penalty(pred, y, mask)
    if config.physics_prior_loss:
        loss = loss + config.physics_prior_weight * physics_prior_penalty(pred, phys, mask, config.rds_dynamic_weight)
    if config.damage_ode_loss:
        loss = loss + config.damage_ode_weight * damage_ode_penalty(pred, phys, mask, config.rds_dynamic_weight)
    return loss


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    abs_err = []
    sq_err = []
    with torch.no_grad():
        for _, _, x, y, phys, _, mask in loader:
            pred = model(x, phys)
            diff = pred[mask] - y[mask]
            abs_err.append(torch.abs(diff))
            sq_err.append(diff ** 2)
    a = torch.cat(abs_err) if abs_err else torch.tensor([0.0])
    s = torch.cat(sq_err) if sq_err else torch.tensor([0.0])
    return {"mae": float(a.mean().item()), "rmse": float(torch.sqrt(s.mean()).item())}


def train_model(config: ExperimentConfig, loaders: dict[str, DataLoader]) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = SequenceModel(in_dim=len(PINN_FEATURE_COLUMNS), config=config)
    opt = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=5.0e-5)
    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for _, _, x, y, phys, _, mask in loaders["train"]:
            pred = model(x, phys)
            loss = loss_for_config(pred, y, phys, mask, config)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_losses.append(float(loss.item()))
        if epoch % 10 == 0 or epoch == 1:
            train_m = evaluate(model, loaders["train"])
            val_m = evaluate(model, loaders["val"])
            row = {
                "model": config.name,
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "train_mae": train_m["mae"],
                "train_rmse": train_m["rmse"],
                "val_mae": val_m["mae"],
                "val_rmse": val_m["rmse"],
            }
            history.append(row)
            if val_m["rmse"] < best_val:
                best_val = val_m["rmse"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_model(model: nn.Module, df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, model_name: str) -> pd.DataFrame:
    ds = SequenceDataset(df, mu, sigma)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_sequences)
    rows = []
    model.eval()
    with torch.no_grad():
        for test_ids, splits, x, y, phys, sample_ids, mask in loader:
            pred = model(x, phys)
            test_id = int(test_ids.item())
            split = splits[0]
            sub = df[df[TEST_ID_COLUMN] == test_id].sort_values(RUN_ORDER_COLUMN).reset_index(drop=True)
            meta_cols = ["sample_index_start", "sample_index_end", "t_epoch_start", "t_epoch_end"]
            meta = sub.reindex(columns=meta_cols).iloc[: int(mask[0].sum().item())]
            for meta_row, sid, yy, ph, pp in zip(
                meta.to_dict(orient="records"),
                sample_ids[0][mask[0]].cpu().numpy(),
                y[0][mask[0]].cpu().numpy(),
                phys[0][mask[0]].cpu().numpy(),
                pred[0][mask[0]].cpu().numpy(),
            ):
                rows.append(
                    {
                        "model": model_name,
                        TEST_ID_COLUMN: test_id,
                        SAMPLE_ID_COLUMN: int(sid),
                        SPLIT_COLUMN: split,
                        "HI_true": float(yy),
                        "HI_physics_prior": float(ph[0]),
                        "Damage_physics": float(ph[1]),
                        "HI_pred": float(pp),
                        "abs_err": float(abs(pp - yy)),
                        **meta_row,
                    }
                )
    return pd.DataFrame(rows)


def hi_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_name = str(pred["model"].iloc[0])
    for scope, sub in pred.groupby(SPLIT_COLUMN, sort=True):
        err = sub["HI_pred"].to_numpy(dtype=np.float64) - sub["HI_true"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": model_name,
                "scope": scope,
                "metric_group": "HI",
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "n_windows": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def eol_rul_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    series_rows = []
    model_name = str(pred["model"].iloc[0])
    for test_id, sub in pred.groupby(TEST_ID_COLUMN, sort=True):
        one = sub.sort_values(SAMPLE_ID_COLUMN).reset_index(drop=True)
        split = str(one[SPLIT_COLUMN].iloc[0])
        elapsed = elapsed_axis_hours(one)
        true_hi = one["HI_true"].to_numpy(dtype=np.float64)
        pred_hi = one["HI_pred"].to_numpy(dtype=np.float64)
        true_eol = first_threshold_crossing_time(elapsed, true_hi, RUL_FAILURE_THRESHOLD)
        pred_eol = first_threshold_crossing_time(elapsed, pred_hi, RUL_FAILURE_THRESHOLD)
        true_ok = bool(np.isfinite(true_eol))
        pred_ok = bool(np.isfinite(pred_eol))
        true_rul = np.maximum(true_eol - elapsed, 0.0) if true_ok else np.full_like(elapsed, np.nan)
        pred_rul = np.maximum(pred_eol - elapsed, 0.0) if pred_ok else np.full_like(elapsed, np.nan)
        err = pred_rul - true_rul
        rows.append(
            {
                "model": model_name,
                TEST_ID_COLUMN: int(test_id),
                SPLIT_COLUMN: split,
                "true_threshold_reached": true_ok,
                "pred_threshold_reached": pred_ok,
                "true_eol_hours": float(true_eol) if true_ok else np.nan,
                "pred_eol_hours": float(pred_eol) if pred_ok else np.nan,
                "eol_error_hours": float(pred_eol - true_eol) if true_ok and pred_ok else np.nan,
                "eol_abs_error_hours": float(abs(pred_eol - true_eol)) if true_ok and pred_ok else np.nan,
                "rul_mae_hours": float(np.nanmean(np.abs(err))) if pred_ok and true_ok else np.nan,
                "rul_rmse_hours": float(np.sqrt(np.nanmean(err ** 2))) if pred_ok and true_ok else np.nan,
                "n_windows": int(len(one)),
            }
        )
        for row, t, rt, rp in zip(one.to_dict(orient="records"), elapsed, true_rul, pred_rul):
            series_rows.append(
                {
                    **row,
                    "elapsed_hours": float(t),
                    "true_eol_hours": float(true_eol) if true_ok else np.nan,
                    "pred_eol_hours": float(pred_eol) if pred_ok else np.nan,
                    "rul_true_hours": float(rt),
                    "rul_pred_hours": float(rp),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(series_rows)


def aggregate_eol(by_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub in by_test.groupby(SPLIT_COLUMN, sort=True):
        valid = sub["true_threshold_reached"].to_numpy(dtype=bool) & sub["pred_threshold_reached"].to_numpy(dtype=bool)
        rows.append(_aggregate_row(str(scope), sub, valid))
    valid = by_test["true_threshold_reached"].to_numpy(dtype=bool) & by_test["pred_threshold_reached"].to_numpy(dtype=bool)
    rows.append(_aggregate_row("all", by_test, valid))
    return pd.DataFrame(rows)


def _aggregate_row(scope: str, sub: pd.DataFrame, valid: np.ndarray) -> dict[str, float | int | str]:
    n = int(len(sub))
    eol_err = sub.loc[valid, "eol_error_hours"].to_numpy(dtype=np.float64)
    rul_mae = sub.loc[valid, "rul_mae_hours"].to_numpy(dtype=np.float64)
    return {
        "model": str(sub["model"].iloc[0]) if n else "",
        "scope": scope,
        "metric_group": "RUL",
        "n_tests": n,
        "pred_crossed_tests": int(np.count_nonzero(sub["pred_threshold_reached"].to_numpy(dtype=bool))),
        "pred_cross_rate": float(np.mean(sub["pred_threshold_reached"].to_numpy(dtype=bool))) if n else np.nan,
        "eol_mae_hours": float(np.nanmean(np.abs(eol_err))) if eol_err.size else np.nan,
        "eol_rmse_hours": float(np.sqrt(np.nanmean(eol_err ** 2))) if eol_err.size else np.nan,
        "rul_mae_hours": float(np.nanmean(rul_mae)) if rul_mae.size else np.nan,
    }


def estimate_eol_from_prefix(elapsed: np.ndarray, hi: np.ndarray, ratio: float, threshold: float) -> tuple[float, float, float]:
    n_prefix = max(5, int(np.ceil(len(hi) * ratio)))
    t_prefix = elapsed[:n_prefix]
    h_prefix = hi[:n_prefix]
    crossed = first_threshold_crossing_time(t_prefix, h_prefix, threshold)
    if np.isfinite(crossed):
        return float(crossed), float("nan"), float("nan")

    tail_n = max(5, int(np.ceil(n_prefix * 0.40)))
    t_tail = t_prefix[-tail_n:]
    h_tail = h_prefix[-tail_n:]
    good = np.isfinite(t_tail) & np.isfinite(h_tail)
    if np.count_nonzero(good) < 3:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(t_tail[good], h_tail[good], deg=1)
    if not np.isfinite(slope) or slope >= -1.0e-6:
        return float("nan"), float(slope), float(intercept)
    pred_eol = (threshold - intercept) / slope
    if pred_eol < t_prefix[-1]:
        return float("nan"), float(slope), float(intercept)
    return float(pred_eol), float(slope), float(intercept)


def early_prediction_metrics(full_pred: pd.DataFrame, ratios=(0.30, 0.50, 0.70)) -> pd.DataFrame:
    rows = []
    test_pred = full_pred[full_pred[SPLIT_COLUMN] == "test"].copy()
    for test_id, sub in test_pred.groupby(TEST_ID_COLUMN, sort=True):
        one = sub.sort_values(SAMPLE_ID_COLUMN).reset_index(drop=True)
        elapsed = elapsed_axis_hours(one)
        true_hi = one["HI_true"].to_numpy(dtype=np.float64)
        pred_hi = one["HI_pred"].to_numpy(dtype=np.float64)
        true_eol = first_threshold_crossing_time(elapsed, true_hi, RUL_FAILURE_THRESHOLD)
        for ratio in ratios:
            pred_eol, slope, intercept = estimate_eol_from_prefix(elapsed, pred_hi, ratio, RUL_FAILURE_THRESHOLD)
            prefix_end_time = float(elapsed[max(0, int(np.ceil(len(elapsed) * ratio)) - 1)])
            rows.append(
                {
                    "model": str(one["model"].iloc[0]),
                    TEST_ID_COLUMN: int(test_id),
                    "prefix_ratio": ratio,
                    "n_total_windows": int(len(one)),
                    "n_observed_windows": int(np.ceil(len(one) * ratio)),
                    "prefix_end_hours": prefix_end_time,
                    "true_eol_hours": float(true_eol) if np.isfinite(true_eol) else np.nan,
                    "pred_eol_hours": pred_eol,
                    "eol_error_hours": pred_eol - true_eol if np.isfinite(pred_eol) and np.isfinite(true_eol) else np.nan,
                    "eol_abs_error_hours": abs(pred_eol - true_eol) if np.isfinite(pred_eol) and np.isfinite(true_eol) else np.nan,
                    "predicted": bool(np.isfinite(pred_eol)),
                    "linear_tail_slope": slope,
                    "linear_tail_intercept": intercept,
                }
            )
    return pd.DataFrame(rows)


def plot_metric_bars(metrics: pd.DataFrame, out_path: Path, title: str, value_cols: tuple[str, str]) -> None:
    test = metrics[metrics["scope"] == "test"].copy()
    if test.empty:
        return
    x = np.arange(len(test))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(test)), 4.2))
    ax.bar(x - width / 2, test[value_cols[0]], width, label=value_cols[0])
    ax.bar(x + width / 2, test[value_cols[1]], width, label=value_cols[1])
    ax.set_xticks(x)
    ax.set_xticklabels(test["model"], rotation=25, ha="right")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_test_overlay(preds: list[pd.DataFrame], out_path: Path, title: str) -> None:
    joined = pd.concat(preds, ignore_index=True)
    test_ids = sorted(joined.loc[joined[SPLIT_COLUMN] == "test", TEST_ID_COLUMN].unique())
    if not test_ids:
        return
    fig, axes = plt.subplots(len(test_ids), 1, figsize=(11, max(3.3, 3.0 * len(test_ids))), sharey=True)
    if len(test_ids) == 1:
        axes = [axes]
    for ax, test_id in zip(axes, test_ids):
        one_all = joined[(joined[SPLIT_COLUMN] == "test") & (joined[TEST_ID_COLUMN] == test_id)]
        first = one_all[one_all["model"] == one_all["model"].iloc[0]].sort_values(SAMPLE_ID_COLUMN)
        x = elapsed_axis_hours(first)
        ax.plot(x, first["HI_true"], color="black", linewidth=2.2, label="True HI")
        for model_name, one in one_all.groupby("model", sort=False):
            one = one.sort_values(SAMPLE_ID_COLUMN)
            ax.plot(elapsed_axis_hours(one), one["HI_pred"], linewidth=1.4, label=model_name)
        ax.set_title(f"Test_{int(test_id)}")
        ax.set_xlabel("Elapsed aging time (h)")
        ax.set_ylabel("HI")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rul_overlay(series_list: list[pd.DataFrame], out_path: Path, title: str) -> None:
    joined = pd.concat(series_list, ignore_index=True)
    test_ids = sorted(joined.loc[joined[SPLIT_COLUMN] == "test", TEST_ID_COLUMN].unique())
    if not test_ids:
        return
    fig, axes = plt.subplots(len(test_ids), 1, figsize=(11, max(3.3, 3.0 * len(test_ids))), sharey=False)
    if len(test_ids) == 1:
        axes = [axes]
    for ax, test_id in zip(axes, test_ids):
        one_all = joined[(joined[SPLIT_COLUMN] == "test") & (joined[TEST_ID_COLUMN] == test_id)]
        first = one_all[one_all["model"] == one_all["model"].iloc[0]].sort_values(SAMPLE_ID_COLUMN)
        ax.plot(first["elapsed_hours"], first["rul_true_hours"], color="black", linewidth=2.2, label="True RUL")
        for model_name, one in one_all.groupby("model", sort=False):
            one = one.sort_values(SAMPLE_ID_COLUMN)
            if one["rul_pred_hours"].notna().any():
                ax.plot(one["elapsed_hours"], one["rul_pred_hours"], linewidth=1.4, label=model_name)
        ax.set_title(f"Test_{int(test_id)}")
        ax.set_xlabel("Elapsed aging time (h)")
        ax.set_ylabel("RUL to HI=0.2 (h)")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_early_metrics(early: pd.DataFrame, out_dir: Path) -> None:
    summary = early.groupby("prefix_ratio", as_index=False).agg(
        eol_mae_hours=("eol_abs_error_hours", "mean"),
        predicted_rate=("predicted", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(summary))
    ax1.bar(x, summary["eol_mae_hours"], width=0.45, color="tab:blue", label="EOL MAE (h)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(r * 100)}%" for r in summary["prefix_ratio"]])
    ax1.set_ylabel("EOL MAE (h)")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["predicted_rate"], color="tab:orange", marker="o", label="Prediction success rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Success rate")
    ax1.set_title("Early EOL Prediction from Observed Prefix")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "early_prediction_summary.png", dpi=180)
    plt.close(fig)


def build_loaders(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, DataLoader], np.ndarray, np.ndarray]:
    split_dfs = {split: df[df[SPLIT_COLUMN] == split].copy() for split in ("train", "val", "test")}
    x_train = split_dfs["train"][PINN_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    loaders = {}
    for split, sub in split_dfs.items():
        ds = SequenceDataset(sub, mu, sigma)
        bs = min(BATCH_SIZE, max(1, len(ds))) if split == "train" else 1
        loaders[split] = DataLoader(ds, batch_size=bs, shuffle=(split == "train"), collate_fn=collate_sequences)
    return split_dfs, loaders, mu, sigma


def run_group(configs: list[ExperimentConfig], df: pd.DataFrame, loaders: dict[str, DataLoader], mu: np.ndarray, sigma: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_hi_metrics = []
    all_rul_metrics = []
    all_by_test = []
    histories = []
    predictions = []
    rul_series = []
    trained_models = {}

    for config in configs:
        print(f"Training {config.family}: {config.name}")
        model, history = train_model(config, loaders)
        pred = predict_model(model, df, mu, sigma, config.name)
        by_test, series = eol_rul_metrics(pred)
        hi_m = hi_metrics(pred)
        rul_m = aggregate_eol(by_test)
        predictions.append(pred)
        rul_series.append(series)
        all_hi_metrics.append(hi_m)
        all_rul_metrics.append(rul_m)
        all_by_test.append(by_test)
        histories.extend(history)
        trained_models[config.name] = model
        pred.to_csv(out_dir / f"{config.name}_predictions.csv", index=False)

    hi_all = pd.concat(all_hi_metrics, ignore_index=True)
    rul_all = pd.concat(all_rul_metrics, ignore_index=True)
    by_test_all = pd.concat(all_by_test, ignore_index=True)
    pd.DataFrame(histories).to_csv(out_dir / "training_history.csv", index=False)
    hi_all.to_csv(out_dir / "hi_metrics.csv", index=False)
    rul_all.to_csv(out_dir / "rul_metrics.csv", index=False)
    by_test_all.to_csv(out_dir / "rul_by_test.csv", index=False)
    plot_metric_bars(hi_all, out_dir / "test_hi_mae_rmse.png", f"{out_dir.name}: Test HI Error", ("mae", "rmse"))
    test_rul = rul_all[rul_all["scope"] == "test"].copy()
    if not test_rul.empty:
        plot_metric_bars(
            test_rul.rename(columns={"eol_mae_hours": "eol_mae", "rul_mae_hours": "rul_mae"}),
            out_dir / "test_rul_eol_mae.png",
            f"{out_dir.name}: Test RUL/EOL Error",
            ("eol_mae", "rul_mae"),
        )
    plot_test_overlay(predictions, out_dir / "test_hi_overlay.png", f"{out_dir.name}: Test HI Overlay")
    plot_rul_overlay(rul_series, out_dir / "test_rul_overlay.png", f"{out_dir.name}: Test RUL Overlay")
    return trained_models, predictions, hi_all, rul_all, by_test_all


def write_readme(baseline_hi: pd.DataFrame, baseline_rul: pd.DataFrame, ablation_hi: pd.DataFrame, early: pd.DataFrame) -> None:
    test_base = baseline_hi[baseline_hi["scope"] == "test"].sort_values("rmse")
    test_rul = baseline_rul[baseline_rul["scope"] == "test"].sort_values("eol_mae_hours")
    test_ab = ablation_hi[ablation_hi["scope"] == "test"].sort_values("rmse")
    early_summary = early.groupby("prefix_ratio", as_index=False).agg(
        eol_mae_hours=("eol_abs_error_hours", "mean"),
        predicted_rate=("predicted", "mean"),
    )
    text = [
        "# NASA MOSFET 对比实验结果说明",
        "",
        "本目录保存三类论文实验：基线对比、消融实验、早期预测实验。",
        "",
        "## 1. 是否需要 RUL 图像",
        "",
        "需要。HI 图说明模型能否重构健康状态；RUL 图说明能否用于寿命预测和维护决策。论文中建议同时给出 HI 曲线、RUL/EOL 曲线和误差表。",
        "",
        "## 2. 基线对比 Test 集 HI 排名",
        "",
        test_base.to_markdown(index=False),
        "",
        "## 3. 基线对比 Test 集 RUL/EOL 排名",
        "",
        test_rul.to_markdown(index=False),
        "",
        "## 4. 消融实验 Test 集 HI 排名",
        "",
        test_ab.to_markdown(index=False),
        "",
        "## 5. 早期预测 Test 集汇总",
        "",
        early_summary.to_markdown(index=False),
        "",
        "## 6. 结果文件",
        "",
        "- `baseline/test_hi_overlay.png`: MLP/RNN/LSTM/GRU/PINN-HI GRU 的 HI 对比。",
        "- `baseline/test_rul_overlay.png`: MLP/RNN/LSTM/GRU/PINN-HI GRU 的 RUL 对比。",
        "- `ablation/test_hi_overlay.png`: 消融实验 HI 对比。",
        "- `ablation/test_rul_overlay.png`: 消融实验 RUL 对比。",
        "- `early_prediction/early_prediction_summary.png`: 30%/50%/70% 前缀寿命预测结果。",
    ]
    (EXPERIMENT_DIR / "README.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for d in (EXPERIMENT_DIR, BASELINE_DIR, ABLATION_DIR, EARLY_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df, excluded_tests, truncated_tests, calibration = build_calibrated_modeling_dataframe()
    _, loaders, mu, sigma = build_loaders(df)

    baseline_models, baseline_preds, baseline_hi, baseline_rul, baseline_by_test = run_group(
        BASELINE_CONFIGS, df, loaders, mu, sigma, BASELINE_DIR
    )
    ablation_models, ablation_preds, ablation_hi, ablation_rul, ablation_by_test = run_group(
        ABLATION_CONFIGS, df, loaders, mu, sigma, ABLATION_DIR
    )

    full_pred = next(pred for pred in baseline_preds if str(pred["model"].iloc[0]) == "PINN_HI_GRU")
    early = early_prediction_metrics(full_pred)
    early.to_csv(EARLY_DIR / "early_prediction_metrics.csv", index=False)
    plot_early_metrics(early, EARLY_DIR)

    all_summary = {
        "epochs": EPOCHS,
        "hidden_size": HIDDEN_SIZE,
        "rul_threshold": RUL_FAILURE_THRESHOLD,
        "rds_failure_delta_calibration": calibration,
        "excluded_tests": excluded_tests,
        "post_eol_truncated_tests": truncated_tests,
        "baseline_hi_metrics": baseline_hi.to_dict(orient="records"),
        "baseline_rul_metrics": baseline_rul.to_dict(orient="records"),
        "ablation_hi_metrics": ablation_hi.to_dict(orient="records"),
        "ablation_rul_metrics": ablation_rul.to_dict(orient="records"),
        "early_prediction_metrics": early.to_dict(orient="records"),
    }
    (EXPERIMENT_DIR / "experiment_summary.json").write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(baseline_hi, baseline_rul, ablation_hi, early)

    print("NASA MOSFET experiments finished.")
    print("Baseline test HI:")
    print(baseline_hi[baseline_hi["scope"] == "test"].sort_values("rmse").to_string(index=False))
    print("Baseline test RUL:")
    print(baseline_rul[baseline_rul["scope"] == "test"].sort_values("eol_mae_hours").to_string(index=False))
    print("Ablation test HI:")
    print(ablation_hi[ablation_hi["scope"] == "test"].sort_values("rmse").to_string(index=False))
    print("Early prediction:")
    print(
        early.groupby("prefix_ratio", as_index=False)
        .agg(eol_mae_hours=("eol_abs_error_hours", "mean"), predicted_rate=("predicted", "mean"))
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
