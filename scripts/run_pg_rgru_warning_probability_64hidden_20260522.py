from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from common_data import RUN_ORDER_COLUMN, SAMPLE_ID_COLUMN, SPLIT_COLUMN, TEST_ID_COLUMN
from run_nasa_mosfet_experiments import aggregate_eol, eol_rul_metrics, hi_metrics
from run_pg_rgru_causal_hi_lowlag_experiment import build_original_and_lowlag_dataframes
from run_pg_rgru_v4_multitask_experiment import add_causal_rate_features
import run_pg_rgru_v7_damage_increment_stage as inc
from train_nasa_mosfet_hi_pinn_gru import LABEL_COLUMN, PINN_PHYS_COLUMNS, RESULTS_DIR, RUL_FAILURE_THRESHOLD, elapsed_axis_hours


OUT_DIR = RESULTS_DIR / "experiments" / "pg_rgru_warning_probability_64hidden_20260522"
SEEDS = [17, 42, 73]
EPOCHS = 120
BATCH_SIZE = 8
MAX_DT_HOURS = inc.MAX_DT_HOURS
WARNING_PROB_THRESHOLD = 0.50
ONSET_PROB_THRESHOLD = 0.50
FPT_EVENT_HORIZON_HOURS = 0.10
FPT_PERSIST_WINDOWS = 3
ACCEL_PERSIST_WINDOWS = 2
FEATURE_COLUMNS = inc.FEATURE_COLUMNS


@dataclass(frozen=True)
class GatedConfig:
    name: str
    hidden: int = 64
    max_increment: float = 0.090
    max_extra_increment: float = 0.0
    multiplier_span: float = 0.18
    hi_weight: float = 1.00
    increment_weight: float = 2.10
    rate_weight: float = 0.10
    fpt_weight: float = 0.34
    accel_weight: float = 0.0
    warning_weight: float = 0.95
    threshold_weight: float = 1.70
    endpoint_weight: float = 0.18
    prior_rate_weight: float = 0.32
    multiplier_anchor_weight: float = 0.28
    gate_sparsity_weight: float = 0.10
    extra_sparsity_weight: float = 0.08
    false_warning_weight: float = 2.10
    late_warning_weight: float = 0.80
    lr: float = 7.0e-4
    weight_decay: float = 5.0e-5


CONFIGS = [
    GatedConfig(
        name="FrozenPriorEventHeads",
        max_extra_increment=0.0,
        multiplier_span=0.0,
        gate_sparsity_weight=0.0,
        extra_sparsity_weight=0.0,
    ),
    GatedConfig(
        name="MultiplierEventPGRGRU",
        max_extra_increment=0.0,
        multiplier_span=0.18,
        gate_sparsity_weight=0.0,
        extra_sparsity_weight=0.0,
    ),
]


def _safe_quantile(values: np.ndarray, q: float, fallback: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, q)) if vals.size else float(fallback)


def _persistent_first(evidence: np.ndarray, n_windows: int) -> int | None:
    if evidence.size == 0:
        return None
    rolling = pd.Series(evidence.astype(np.int64)).rolling(n_windows, min_periods=n_windows).sum()
    hits = np.flatnonzero(rolling.to_numpy(dtype=np.float64) >= n_windows)
    return int(hits[0]) if hits.size else None


def add_onset_evidence(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, sub in df.groupby(TEST_ID_COLUMN, sort=True):
        g = sub.sort_values(RUN_ORDER_COLUMN).copy()
        t = pd.Series(elapsed_axis_hours(g), index=g.index, dtype="float64")
        incv = pd.Series(g["damage_increment_target"].to_numpy(dtype=np.float64), index=g.index)
        rate = pd.Series(g["damage_rate_target"].to_numpy(dtype=np.float64), index=g.index)

        rate_short = rate.rolling(window=3, min_periods=1).median()
        rate_long = rate.rolling(window=7, min_periods=1).median()
        increment_short = incv.rolling(window=3, min_periods=1).median()
        dt = t.diff().replace(0.0, np.nan)
        accel = ((rate_short - rate_long) / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        accel = accel.clip(lower=0.0, upper=12.0).rolling(window=3, min_periods=1).median()

        g["target_rate_short"] = np.nan_to_num(rate_short.to_numpy(dtype=np.float64), nan=0.0)
        g["target_rate_long"] = np.nan_to_num(rate_long.to_numpy(dtype=np.float64), nan=0.0)
        g["target_increment_short"] = np.nan_to_num(increment_short.to_numpy(dtype=np.float64), nan=0.0)
        g["target_accel_pos"] = np.nan_to_num(accel.to_numpy(dtype=np.float64), nan=0.0)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def calibrate_onset_thresholds(df: pd.DataFrame) -> dict[str, float]:
    train = df[df[SPLIT_COLUMN] == "train"].copy()
    prewarning = train[train["warning_target"] < 0.5]
    initial = prewarning[prewarning[LABEL_COLUMN] >= 0.95]

    initial_rate = initial["target_rate_short"].to_numpy(dtype=np.float64)
    initial_inc = initial["target_increment_short"].to_numpy(dtype=np.float64)
    positive_rate = prewarning.loc[prewarning["target_rate_short"] > 1.0e-8, "target_rate_short"].to_numpy(dtype=np.float64)
    positive_inc = prewarning.loc[prewarning["target_increment_short"] > 1.0e-8, "target_increment_short"].to_numpy(dtype=np.float64)
    positive_accel = prewarning.loc[prewarning["target_accel_pos"] > 1.0e-8, "target_accel_pos"].to_numpy(dtype=np.float64)

    noise_rate = _safe_quantile(initial_rate, 0.98, 0.025)
    noise_inc = _safe_quantile(initial_inc, 0.98, 0.0005)
    fpt_rate = max(1.25 * noise_rate, _safe_quantile(positive_rate, 0.35, 0.08))
    fpt_inc = max(1.25 * noise_inc, _safe_quantile(positive_inc, 0.35, 0.001))
    return {
        "initial_noise_rate_q98": float(noise_rate),
        "initial_noise_increment_q98": float(noise_inc),
        "fpt_rate_threshold": float(fpt_rate),
        "fpt_increment_threshold": float(fpt_inc),
        "accel_rate_threshold": float(max(1.8 * fpt_rate, _safe_quantile(positive_rate, 0.78, 0.28))),
        "accel_increment_threshold": float(max(1.8 * fpt_inc, _safe_quantile(positive_inc, 0.78, 0.004))),
        "accel_positive_threshold": float(_safe_quantile(positive_accel, 0.78, 0.20)),
    }


def add_onset_targets(df: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    onset_rows = []
    for test_id, sub in df.groupby(TEST_ID_COLUMN, sort=True):
        g = sub.sort_values(RUN_ORDER_COLUMN).copy()
        t = elapsed_axis_hours(g)
        fpt_evidence = (
            (g["target_rate_short"].to_numpy(dtype=np.float64) >= thresholds["fpt_rate_threshold"])
            | (g["target_increment_short"].to_numpy(dtype=np.float64) >= thresholds["fpt_increment_threshold"])
        )
        fpt_idx = _persistent_first(fpt_evidence, FPT_PERSIST_WINDOWS)

        accel_evidence = (
            (g["target_rate_short"].to_numpy(dtype=np.float64) >= thresholds["accel_rate_threshold"])
            | (g["target_increment_short"].to_numpy(dtype=np.float64) >= thresholds["accel_increment_threshold"])
            | (g["target_accel_pos"].to_numpy(dtype=np.float64) >= thresholds["accel_positive_threshold"])
        )
        if fpt_idx is not None:
            accel_evidence[:fpt_idx] = False
        accel_idx = _persistent_first(accel_evidence, ACCEL_PERSIST_WINDOWS)
        if fpt_idx is not None and accel_idx is not None:
            accel_idx = max(accel_idx, fpt_idx)

        fpt_state = np.zeros(len(g), dtype=np.float64)
        fpt_event = np.zeros(len(g), dtype=np.float64)
        accel_state = np.zeros(len(g), dtype=np.float64)
        if fpt_idx is not None:
            fpt_state[fpt_idx:] = 1.0
            time_to_fpt = t[fpt_idx] - t
            fpt_event = ((time_to_fpt <= FPT_EVENT_HORIZON_HOURS) & (time_to_fpt >= -1.0e-8)).astype(np.float64)
        if accel_idx is not None:
            accel_state[accel_idx:] = 1.0

        g["fpt_state_target"] = fpt_state
        g["fpt_event_target"] = fpt_event
        g["accel_state_target"] = accel_state
        parts.append(g)
        onset_rows.append(
            {
                TEST_ID_COLUMN: int(test_id),
                SPLIT_COLUMN: str(g[SPLIT_COLUMN].iloc[0]),
                "n_windows": int(len(g)),
                "fpt_detected": fpt_idx is not None,
                "fpt_index": int(fpt_idx) if fpt_idx is not None else np.nan,
                "fpt_time_hours": float(t[fpt_idx]) if fpt_idx is not None else np.nan,
                "accel_detected": accel_idx is not None,
                "accel_index": int(accel_idx) if accel_idx is not None else np.nan,
                "accel_time_hours": float(t[accel_idx]) if accel_idx is not None else np.nan,
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(onset_rows)


def build_causal_onset_dataframe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, float]]:
    original_df, causal_df, calibration = build_original_and_lowlag_dataframes()
    df = add_causal_rate_features(causal_df)
    df = inc.add_increment_stage_targets(df)
    df = add_onset_evidence(df)
    thresholds = calibrate_onset_thresholds(df)
    df, onset_reference = add_onset_targets(df, thresholds)
    return original_df, causal_df, df, onset_reference, calibration, thresholds


class OnsetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray):
        self.samples = []
        for test_id, sub in df.groupby(TEST_ID_COLUMN, sort=True):
            g = sub.sort_values(RUN_ORDER_COLUMN)
            x = ((g[FEATURE_COLUMNS].to_numpy(dtype=np.float64) - mu) / sigma).astype(np.float32)
            y = g[LABEL_COLUMN].to_numpy(dtype=np.float32)
            damage = g["Damage_target"].to_numpy(dtype=np.float32)
            incv = g["damage_increment_target"].to_numpy(dtype=np.float32)
            rate = g["damage_rate_target"].to_numpy(dtype=np.float32)
            fpt = g["fpt_event_target"].to_numpy(dtype=np.float32)
            accel = g["accel_state_target"].to_numpy(dtype=np.float32)
            warning = g["warning_target"].to_numpy(dtype=np.float32)
            phys = g[PINN_PHYS_COLUMNS].to_numpy(dtype=np.float32)
            ids = g[SAMPLE_ID_COLUMN].to_numpy(dtype=np.int64)
            times = elapsed_axis_hours(g).astype(np.float32)
            self.samples.append(
                (
                    int(test_id),
                    str(g[SPLIT_COLUMN].iloc[0]),
                    torch.tensor(x),
                    torch.tensor(y),
                    torch.tensor(damage),
                    torch.tensor(incv),
                    torch.tensor(rate),
                    torch.tensor(fpt),
                    torch.tensor(accel),
                    torch.tensor(warning),
                    torch.tensor(phys),
                    torch.tensor(ids),
                    torch.tensor(times),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate_sequences(batch):
    test_ids, splits, xs, ys, damages, incs, rates, fpts, accels, warnings, phys, ids, times = zip(*batch)
    x = pad_sequence(xs, batch_first=True)
    y = pad_sequence(ys, batch_first=True, padding_value=-1.0)
    damage = pad_sequence(damages, batch_first=True, padding_value=-1.0)
    incv = pad_sequence(incs, batch_first=True, padding_value=0.0)
    rate = pad_sequence(rates, batch_first=True, padding_value=0.0)
    fpt = pad_sequence(fpts, batch_first=True, padding_value=0.0)
    accel = pad_sequence(accels, batch_first=True, padding_value=0.0)
    warning = pad_sequence(warnings, batch_first=True, padding_value=0.0)
    phys_pad = pad_sequence(phys, batch_first=True)
    ids_pad = pad_sequence(ids, batch_first=True, padding_value=-1)
    t_pad = pad_sequence(times, batch_first=True, padding_value=0.0)
    mask = y >= 0.0
    return (
        torch.tensor(test_ids, dtype=torch.long),
        list(splits),
        x,
        y,
        damage,
        incv,
        rate,
        fpt,
        accel,
        warning,
        phys_pad,
        ids_pad,
        t_pad,
        mask,
    )


class GatedIncrementGRU(nn.Module):
    def __init__(self, in_dim: int, config: GatedConfig):
        super().__init__()
        self.config = config
        self.backbone = nn.GRU(input_size=in_dim, hidden_size=config.hidden, batch_first=True)
        head_in = config.hidden + in_dim + len(PINN_PHYS_COLUMNS) + 1
        self.shared = nn.Sequential(nn.Linear(head_in, config.hidden), nn.Tanh(), nn.Dropout(0.05))
        self.multiplier_head = nn.Linear(config.hidden, 1)
        self.gate_head = nn.Linear(config.hidden, 1)
        self.extra_head = nn.Linear(config.hidden, 1)
        self.fpt_head = nn.Linear(config.hidden, 1)
        self.accel_head = nn.Linear(config.hidden, 1)
        self.warning_head = nn.Linear(config.hidden, 1)
        nn.init.constant_(self.extra_head.bias, -6.2)
        nn.init.constant_(self.gate_head.bias, -2.8)

    def forward(self, x: torch.Tensor, phys: torch.Tensor, times: torch.Tensor) -> dict[str, torch.Tensor]:
        h, _ = self.backbone(x)
        time_feature = torch.log1p(torch.clamp(times, min=0.0)).unsqueeze(-1) / 4.0
        z = torch.cat([h, x, phys, time_feature], dim=-1)
        s = self.shared(z)
        prior_damage = torch.clamp(phys[..., 1], 0.0, 1.0)
        prior_inc = torch.zeros_like(prior_damage)
        prior_inc[:, 1:] = torch.relu(prior_damage[:, 1:] - prior_damage[:, :-1])
        dt = torch.zeros_like(times)
        dt[:, 1:] = torch.clamp(times[:, 1:] - times[:, :-1], min=0.0, max=MAX_DT_HOURS)

        span = self.config.multiplier_span
        multiplier = (1.0 - span) + 2.0 * span * torch.sigmoid(self.multiplier_head(s).squeeze(-1))
        gate = torch.sigmoid(self.gate_head(s).squeeze(-1))
        extra_base = self.config.max_extra_increment * torch.sigmoid(self.extra_head(s).squeeze(-1))
        extra_inc = gate * extra_base
        inc_pred = torch.clamp(prior_inc * multiplier + extra_inc, 0.0, self.config.max_increment)
        inc_pred = inc_pred * (dt > 0.0).to(dtype=inc_pred.dtype)
        damage0 = torch.clamp(phys[:, :1, 1], 0.0, 1.0)
        damage = torch.clamp(damage0 + torch.cumsum(inc_pred, dim=1), 0.0, 1.0)
        return {
            "hi": torch.clamp(1.0 - damage, 0.0, 1.0),
            "damage": damage,
            "increment": inc_pred,
            "prior_increment": prior_inc,
            "extra_increment": extra_inc,
            "multiplier": multiplier,
            "gate": gate,
            "rate": inc_pred / torch.clamp(dt, min=1.0e-5),
            "fpt_logit": self.fpt_head(s).squeeze(-1),
            "accel_logit": self.accel_head(s).squeeze(-1),
            "warning_logit": self.warning_head(s).squeeze(-1),
        }


def build_loaders(df: pd.DataFrame) -> tuple[dict[str, DataLoader], np.ndarray, np.ndarray]:
    train = df[df[SPLIT_COLUMN] == "train"]
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mu = np.nanmean(x_train, axis=0)
    sigma = np.nanstd(x_train, axis=0)
    mu[~np.isfinite(mu)] = 0.0
    sigma[~np.isfinite(sigma) | (sigma == 0.0)] = 1.0
    loaders = {}
    for split in ("train", "val", "test"):
        ds = OnsetDataset(df[df[SPLIT_COLUMN] == split].copy(), mu, sigma)
        bs = min(BATCH_SIZE, max(1, len(ds))) if split == "train" else 1
        loaders[split] = DataLoader(ds, batch_size=bs, shuffle=(split == "train"), collate_fn=collate_sequences)
    return loaders, mu, sigma


def state_loss(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, positive_weight: float) -> torch.Tensor:
    logits = logit[mask]
    labels = target[mask]
    if labels.numel() == 0:
        return logit.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(positive_weight, dtype=logits.dtype, device=logits.device),
    )


def warning_event_loss(logit: torch.Tensor, target: torch.Tensor, target_hi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logit[mask]
    labels = target[mask]
    hi = target_hi[mask]
    if labels.numel() == 0:
        return logit.new_tensor(0.0)
    prob = torch.sigmoid(logits)
    positive = labels >= 0.5
    far_negative = (~positive) & (hi > 0.42)
    near_negative = (~positive) & (hi > RUL_FAILURE_THRESHOLD) & (hi <= 0.42)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(4.0, dtype=logits.dtype, device=logits.device),
        reduction="none",
    )
    weights = torch.ones_like(bce)
    weights[positive] = 2.0
    weights[near_negative] = 1.35
    weights[far_negative] = 2.0
    far_false = torch.mean(torch.relu(prob[far_negative] - 0.10) ** 2) if torch.any(far_negative) else logits.new_tensor(0.0)
    near_false = torch.mean(torch.relu(prob[near_negative] - 0.24) ** 2) if torch.any(near_negative) else logits.new_tensor(0.0)
    positive_floor = torch.mean(torch.relu(0.68 - prob[positive]) ** 2) if torch.any(positive) else logits.new_tensor(0.0)
    return torch.mean(weights * bce) + 0.9 * far_false + 0.35 * near_false + 0.25 * positive_floor


def gate_sparsity_loss(outputs: dict[str, torch.Tensor], phys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.clone()
    active[:, 0] = False
    if not torch.any(active):
        return outputs["gate"].new_tensor(0.0)
    confidence = torch.clamp(phys[..., 3], 0.0, 1.0)
    gate_penalty = (0.35 + 0.65 * confidence[active]) * outputs["gate"][active]
    return torch.mean(gate_penalty)


def extra_sparsity_loss(outputs: dict[str, torch.Tensor], target_hi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.clone()
    active[:, 0] = False
    if not torch.any(active):
        return outputs["extra_increment"].new_tensor(0.0)
    hi = target_hi[active]
    weights = 0.35 + 0.65 * (hi > 0.45).to(dtype=hi.dtype)
    return torch.mean(weights * torch.abs(outputs["extra_increment"][active]))


def order_loss(outputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    fpt_prob = torch.sigmoid(outputs["fpt_logit"])[mask]
    accel_prob = torch.sigmoid(outputs["accel_logit"])[mask]
    if fpt_prob.numel() == 0:
        return outputs["fpt_logit"].new_tensor(0.0)
    return torch.mean(torch.relu(accel_prob - fpt_prob) ** 2)


def loss_for_config(outputs: dict[str, torch.Tensor], batch: tuple, config: GatedConfig) -> torch.Tensor:
    _, _, _, y, _, inc_true, rate_true, fpt_true, accel_true, warn_true, phys, _, _, mask = batch
    false_loss, late_loss = inc.threshold_guard_loss(outputs["hi"], y, mask)
    return (
        config.hi_weight * inc.weighted_hi_loss(outputs["hi"], y, mask, config.threshold_weight)
        + config.increment_weight * inc.increment_loss(outputs["increment"], inc_true, y, mask)
        + config.rate_weight * inc.masked_huber(outputs["rate"], rate_true, mask, delta=0.45)
        + config.fpt_weight * state_loss(outputs["fpt_logit"], fpt_true, mask, positive_weight=4.0)
        + config.accel_weight * state_loss(outputs["accel_logit"], accel_true, mask, positive_weight=1.6)
        + config.warning_weight * warning_event_loss(outputs["warning_logit"], warn_true, y, mask)
        + config.endpoint_weight * inc.endpoint_loss(outputs["hi"], y, mask)
        + config.prior_rate_weight * inc.prior_rate_loss(outputs["increment"], phys, None, mask)
        + config.multiplier_anchor_weight * inc.multiplier_anchor_loss(outputs["multiplier"], outputs["prior_increment"], mask)
        + config.gate_sparsity_weight * gate_sparsity_loss(outputs, phys, mask)
        + config.extra_sparsity_weight * extra_sparsity_loss(outputs, y, mask)
        + config.false_warning_weight * false_loss
        + config.late_warning_weight * late_loss
    )


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    hi_sq = []
    inc_sq = []
    fpt_brier = []
    accel_brier = []
    warning_brier = []
    hi_false = 0
    n_points = 0
    with torch.no_grad():
        for batch in loader:
            _, _, x, y, _, inc_true, _, fpt_true, accel_true, warn_true, phys, _, times, mask = batch
            out = model(x, phys, times)
            hi_sq.append((out["hi"][mask] - y[mask]) ** 2)
            active = mask.clone()
            active[:, 0] = False
            if torch.any(active):
                inc_sq.append((out["increment"][active] - inc_true[active]) ** 2)
            fpt_prob = torch.sigmoid(out["fpt_logit"])[mask]
            accel_prob = torch.sigmoid(out["accel_logit"])[mask]
            warning_prob = torch.sigmoid(out["warning_logit"])[mask]
            fpt_brier.append((fpt_prob - fpt_true[mask]) ** 2)
            accel_brier.append((accel_prob - accel_true[mask]) ** 2)
            warning_brier.append((warning_prob - warn_true[mask]) ** 2)
            hi_false += int(torch.count_nonzero((y[mask] > RUL_FAILURE_THRESHOLD + 0.035) & (out["hi"][mask] <= RUL_FAILURE_THRESHOLD)).item())
            n_points += int(torch.count_nonzero(mask).item())
    hi_s = torch.cat(hi_sq) if hi_sq else torch.tensor([0.0])
    inc_s = torch.cat(inc_sq) if inc_sq else torch.tensor([0.0])
    return {
        "hi_rmse": float(torch.sqrt(hi_s.mean()).item()),
        "increment_rmse": float(torch.sqrt(inc_s.mean()).item()),
        "fpt_brier": float(torch.cat(fpt_brier).mean().item()) if fpt_brier else 0.0,
        "accel_brier": float(torch.cat(accel_brier).mean().item()) if accel_brier else 0.0,
        "warning_brier": float(torch.cat(warning_brier).mean().item()) if warning_brier else 0.0,
        "hi_false_warning_frac": float(hi_false / max(n_points, 1)),
    }


def train_model(config: GatedConfig, loaders: dict[str, DataLoader], seed: int) -> tuple[nn.Module, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GatedIncrementGRU(len(FEATURE_COLUMNS), config)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state = None
    best_score = float("inf")
    rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            _, _, x, _, _, _, _, _, _, _, phys, _, times, _ = batch
            out = model(x, phys, times)
            loss = loss_for_config(out, batch, config)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 10 == 0:
            train_m = evaluate(model, loaders["train"])
            val_m = evaluate(model, loaders["val"])
            score = (
                val_m["hi_rmse"]
                + 1.7 * val_m["increment_rmse"]
                + 0.12 * val_m["warning_brier"]
                + 0.05 * val_m["fpt_brier"]
                + 0.45 * val_m["hi_false_warning_frac"]
            )
            rows.append(
                {
                    "model": config.name,
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(losses)),
                    **{f"train_{k}": v for k, v in train_m.items()},
                    **{f"val_{k}": v for k, v in val_m.items()},
                    "selection_score": float(score),
                }
            )
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"{config.name} seed {seed} epoch {epoch:03d} "
                f"val_hi={val_m['hi_rmse']:.4f} val_inc={val_m['increment_rmse']:.4f} "
                f"fpt_event={val_m['fpt_brier']:.4f} warn={val_m['warning_brier']:.4f}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_model(model: nn.Module, df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, model_name: str, seed: int) -> pd.DataFrame:
    ds = OnsetDataset(df, mu, sigma)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_sequences)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            test_ids, splits, x, y, damage, inc_true, rate_true, fpt_true, accel_true, warn_true, phys, ids, times, mask = batch
            out = model(x, phys, times)
            n = int(mask[0].sum().item())
            test_id = int(test_ids.item())
            sub = df[df[TEST_ID_COLUMN] == test_id].sort_values(RUN_ORDER_COLUMN).reset_index(drop=True).iloc[:n]
            meta_cols = ["sample_index_start", "sample_index_end", "t_epoch_start", "t_epoch_end", "elapsed_hours"]
            meta = sub.reindex(columns=meta_cols).to_dict(orient="records")
            for idx in range(n):
                row = dict(meta[idx])
                row.update(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        TEST_ID_COLUMN: test_id,
                        SAMPLE_ID_COLUMN: int(ids[0, idx].item()),
                        SPLIT_COLUMN: splits[0],
                        "time_hours": float(times[0, idx].item()),
                        "HI_true": float(y[0, idx].item()),
                        "HI_pred": float(out["hi"][0, idx].item()),
                        "damage_true": float(damage[0, idx].item()),
                        "damage_pred": float(out["damage"][0, idx].item()),
                        "increment_true": float(inc_true[0, idx].item()),
                        "increment_pred": float(out["increment"][0, idx].item()),
                        "rate_true": float(rate_true[0, idx].item()),
                        "fpt_event_target": float(fpt_true[0, idx].item()),
                        "accel_state_target": float(accel_true[0, idx].item()),
                        "warning_target": float(warn_true[0, idx].item()),
                        "fpt_prob": float(torch.sigmoid(out["fpt_logit"])[0, idx].item()),
                        "accel_prob": float(torch.sigmoid(out["accel_logit"])[0, idx].item()),
                        "warning_prob": float(torch.sigmoid(out["warning_logit"])[0, idx].item()),
                        "gate": float(out["gate"][0, idx].item()),
                        "multiplier": float(out["multiplier"][0, idx].item()),
                        "extra_increment": float(out["extra_increment"][0, idx].item()),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def prior_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "model": "Causal_Rds_prior",
            TEST_ID_COLUMN: df[TEST_ID_COLUMN].to_numpy(),
            SAMPLE_ID_COLUMN: df[SAMPLE_ID_COLUMN].to_numpy(),
            SPLIT_COLUMN: df[SPLIT_COLUMN].to_numpy(),
            "HI_true": df[LABEL_COLUMN].to_numpy(dtype=np.float64),
            "HI_pred": df["hi_physics_prior"].to_numpy(dtype=np.float64),
        }
    )
    for col in ["sample_index_start", "sample_index_end", "t_epoch_start", "t_epoch_end", "elapsed_hours"]:
        if col in df.columns:
            out[col] = df[col].to_numpy()
    return out


def load_original_causal_predictions() -> pd.DataFrame | None:
    path = RESULTS_DIR / "experiments" / "pg_rgru_causal_hi_lowlag_ewma_20260521" / "causal_lowlag_predictions_all_models.csv"
    if not path.exists():
        return None
    pred = pd.read_csv(path)
    pred = pred[pred["model"] == "PINN_HI_GRU_causalHI"].copy()
    pred["model"] = "Original_PG_RGRU_causal"
    return pred


def baseline_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [prior_predictions(df)]
    original = load_original_causal_predictions()
    if original is not None:
        frames.append(original)
    hi_rows = []
    rul_rows = []
    for frame in frames:
        hi_rows.append(hi_metrics(frame))
        by_test, _ = eol_rul_metrics(frame)
        rul_rows.append(aggregate_eol(by_test))
    return pd.concat(hi_rows, ignore_index=True), pd.concat(rul_rows, ignore_index=True)


def increment_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub in pred.groupby(SPLIT_COLUMN, sort=True):
        active = sub[sub["increment_true"].notna()]
        err = active["increment_pred"].to_numpy(dtype=np.float64) - active["increment_true"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": str(sub["model"].iloc[0]),
                "seed": int(sub["seed"].iloc[0]),
                "scope": str(scope),
                "increment_mae": float(np.mean(np.abs(err))) if err.size else np.nan,
                "increment_rmse": float(np.sqrt(np.mean(err**2))) if err.size else np.nan,
                "gate_mean": float(sub["gate"].mean()),
                "gate_p95": float(sub["gate"].quantile(0.95)),
                "extra_increment_mean": float(sub["extra_increment"].mean()),
                "multiplier_mean": float(sub["multiplier"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _first_time(t: np.ndarray, flag: np.ndarray) -> float:
    idx = np.flatnonzero(flag)
    return float(t[idx[0]]) if idx.size else float("nan")


def onset_metrics(pred: pd.DataFrame, target_col: str, prob_col: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    by_test_rows = []
    for scope, sub_scope in pred.groupby(SPLIT_COLUMN, sort=True):
        scoped = []
        for test_id, sub in sub_scope.groupby(TEST_ID_COLUMN, sort=True):
            one = sub.sort_values(SAMPLE_ID_COLUMN).reset_index(drop=True)
            t = one["time_hours"].to_numpy(dtype=np.float64)
            target = one[target_col].to_numpy(dtype=np.float64) >= 0.5
            prob = one[prob_col].to_numpy(dtype=np.float64)
            pred_state = prob >= ONSET_PROB_THRESHOLD
            true_time = _first_time(t, target)
            pred_time = _first_time(t, pred_state)
            target_present = bool(np.any(target))
            row = {
                "model": str(one["model"].iloc[0]),
                "seed": int(one["seed"].iloc[0]),
                "scope": str(scope),
                "boundary": label,
                TEST_ID_COLUMN: int(test_id),
                "target_onset_present": target_present,
                "predicted_onset_present": bool(np.any(pred_state)),
                "target_onset_hours": true_time,
                "predicted_onset_hours": pred_time,
                "onset_error_hours": float(pred_time - true_time) if np.isfinite(true_time) and np.isfinite(pred_time) else np.nan,
                "brier": float(np.mean((prob - target.astype(np.float64)) ** 2)),
                "n_windows": int(len(one)),
            }
            scoped.append(row)
            by_test_rows.append(row)
        scoped_df = pd.DataFrame(scoped)
        present = scoped_df[scoped_df["target_onset_present"]]
        finite = present["onset_error_hours"].dropna().to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": str(sub_scope["model"].iloc[0]),
                "seed": int(sub_scope["seed"].iloc[0]),
                "scope": str(scope),
                "boundary": label,
                "brier": float(np.mean((sub_scope[prob_col].to_numpy(dtype=np.float64) - sub_scope[target_col].to_numpy(dtype=np.float64)) ** 2)),
                "target_onset_tests": int(len(present)),
                "predicted_onset_tests": int(present["predicted_onset_present"].sum()) if not present.empty else 0,
                "missed_onset_tests": int((~present["predicted_onset_present"]).sum()) if not present.empty else 0,
                "onset_error_mae_hours": float(np.mean(np.abs(finite))) if finite.size else np.nan,
                "onset_error_mean_hours": float(np.mean(finite)) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(by_test_rows)


def event_window_metrics(
    pred: pd.DataFrame,
    target_col: str,
    prob_col: str,
    label: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    by_test_rows = []
    for scope, sub_scope in pred.groupby(SPLIT_COLUMN, sort=True):
        target = sub_scope[target_col].to_numpy(dtype=np.float64) >= 0.5
        prob = sub_scope[prob_col].to_numpy(dtype=np.float64)
        pred_pos = prob >= threshold
        false_window_rate = float(np.count_nonzero(pred_pos & ~target) / max(1, np.count_nonzero(~target)))
        scoped = []
        for test_id, sub in sub_scope.groupby(TEST_ID_COLUMN, sort=True):
            one = sub.sort_values(SAMPLE_ID_COLUMN).reset_index(drop=True)
            t = one["time_hours"].to_numpy(dtype=np.float64)
            y = one[target_col].to_numpy(dtype=np.float64) >= 0.5
            p = one[prob_col].to_numpy(dtype=np.float64)
            yp = p >= threshold
            target_time = _first_time(t, y)
            pred_time = _first_time(t, yp)
            has_target = bool(np.any(y))
            coverage = bool(np.any(y & yp)) if has_target else False
            row = {
                "model": str(one["model"].iloc[0]),
                "seed": int(one["seed"].iloc[0]),
                "scope": str(scope),
                "event": label,
                TEST_ID_COLUMN: int(test_id),
                "target_event_present": has_target,
                "probability_event_triggered": bool(np.any(yp)),
                "event_covered_in_target_window": coverage,
                "missed_event": bool(has_target and not coverage),
                "false_event_before_target": bool(has_target and np.any(yp & (t < target_time))),
                "target_event_window_start_hours": target_time,
                "pred_event_trigger_hours": pred_time,
                "event_lead_time_error_hours": float(target_time - pred_time) if np.isfinite(target_time) and np.isfinite(pred_time) else np.nan,
                "event_prob_max": float(np.max(p)) if p.size else np.nan,
            }
            scoped.append(row)
            by_test_rows.append(row)
        scoped_df = pd.DataFrame(scoped)
        present = scoped_df[scoped_df["target_event_present"]]
        finite = present["event_lead_time_error_hours"].dropna().to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": str(sub_scope["model"].iloc[0]),
                "seed": int(sub_scope["seed"].iloc[0]),
                "scope": str(scope),
                "event": label,
                "event_brier": float(np.mean((prob - target.astype(np.float64)) ** 2)),
                "false_event_window_rate": false_window_rate,
                "target_event_tests": int(len(present)),
                "event_covered_tests": int(present["event_covered_in_target_window"].sum()) if not present.empty else 0,
                "missed_event_tests": int(present["missed_event"].sum()) if not present.empty else 0,
                "missed_event_rate": float(present["missed_event"].mean()) if not present.empty else np.nan,
                "false_event_before_target_tests": int(present["false_event_before_target"].sum()) if not present.empty else 0,
                "event_lead_time_error_mae_hours": float(np.mean(np.abs(finite))) if finite.size else np.nan,
                "event_lead_time_error_mean_hours": float(np.mean(finite)) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(by_test_rows)


def warning_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, by_test = event_window_metrics(
        pred,
        "warning_target",
        "warning_prob",
        "Warning",
        WARNING_PROB_THRESHOLD,
    )
    return (
        summary.rename(
            columns={
                "event_brier": "warning_brier",
                "false_event_window_rate": "false_warning_window_rate",
                "target_event_tests": "target_warning_tests",
                "event_covered_tests": "warning_covered_tests",
                "missed_event_tests": "missed_warning_tests",
                "missed_event_rate": "missed_warning_rate",
                "false_event_before_target_tests": "false_warning_before_target_tests",
                "event_lead_time_error_mae_hours": "warning_lead_time_error_mae_hours",
                "event_lead_time_error_mean_hours": "warning_lead_time_error_mean_hours",
            }
        ),
        by_test.rename(
            columns={
                "target_event_present": "target_warning_present",
                "probability_event_triggered": "probability_warning_triggered",
                "event_covered_in_target_window": "warning_covered_in_target_window",
                "missed_event": "missed_warning",
                "false_event_before_target": "false_warning_before_target",
                "target_event_window_start_hours": "target_warning_onset_hours",
                "pred_event_trigger_hours": "pred_warning_onset_hours",
                "event_lead_time_error_hours": "warning_lead_time_error_hours",
                "event_prob_max": "warning_prob_max",
            }
        ),
    )


def aggregate_seed_metrics(metrics: pd.DataFrame, cols: list[str], table: str, extra_filters: dict[str, str] | None = None) -> pd.DataFrame:
    test = metrics[metrics["scope"] == "test"].copy()
    if extra_filters:
        for col, value in extra_filters.items():
            test = test[test[col] == value]
    rows = []
    for col in cols:
        vals = pd.to_numeric(test[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        rows.append(
            {
                "table": table,
                "metric": col,
                "n_seeds": int(vals.size),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "min": float(np.min(vals)) if vals.size else np.nan,
                "max": float(np.max(vals)) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_one(config: GatedConfig, seed: int, df: pd.DataFrame, loaders: dict[str, DataLoader], mu: np.ndarray, sigma: np.ndarray, calibration: dict[str, object]) -> dict[str, pd.DataFrame]:
    seed_dir = OUT_DIR / config.name / f"seed_{seed}"
    pred_path = seed_dir / "predictions.csv"
    history_path = seed_dir / "training_history.csv"
    if pred_path.exists() and history_path.exists():
        pred = pd.read_csv(pred_path)
        history = pd.read_csv(history_path)
        print(f"Reusing completed run: {config.name} seed {seed}")
    else:
        model, history = train_model(config, loaders, seed)
        pred = predict_model(model, df, mu, sigma, f"{config.name}_seed{seed}", seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        pred.to_csv(pred_path, index=False)
        history.to_csv(history_path, index=False)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_columns": FEATURE_COLUMNS,
                "physics_columns": PINN_PHYS_COLUMNS,
                "mu": mu,
                "sigma": sigma,
                "config": asdict(config),
                "seed": int(seed),
                "calibration": calibration,
                "label_rule": "causal low-lag HI; FPT event-in-horizon target; event-in-horizon warning",
            },
            seed_dir / "checkpoint.pt",
        )
    hi = hi_metrics(pred)
    hi["seed"] = int(seed)
    by_test, _ = eol_rul_metrics(pred)
    by_test["seed"] = int(seed)
    rul = aggregate_eol(by_test)
    rul["seed"] = int(seed)
    incr = increment_metrics(pred)
    fpt_summary, fpt_by_test = event_window_metrics(pred, "fpt_event_target", "fpt_prob", "FPT", ONSET_PROB_THRESHOLD)
    warning_summary, warning_by_test = warning_metrics(pred)

    by_test.to_csv(seed_dir / "hi_threshold_rul_by_test.csv", index=False)
    return {
        "pred": pred,
        "history": history,
        "hi": hi,
        "rul": rul,
        "rul_by_test": by_test,
        "increment": incr,
        "fpt": fpt_summary,
        "fpt_by_test": fpt_by_test,
        "warning": warning_summary,
        "warning_by_test": warning_by_test,
    }


def plot_test34_hi(df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    truth = df[df[TEST_ID_COLUMN] == 34].sort_values(RUN_ORDER_COLUMN).copy()
    t = elapsed_axis_hours(truth)
    focus = (t >= 41.30) & (t <= 41.93)
    prior = truth["hi_physics_prior"].to_numpy(dtype=np.float64)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    ax.plot(t[focus], truth.loc[focus, LABEL_COLUMN], color="#222222", linewidth=1.7, label="Causal reference HI")
    ax.plot(t[focus], prior[focus], color="#7A8797", linewidth=1.25, linestyle=(0, (3, 2)), label="Rds(on) prior")
    colors = {"MultiplierEventPGRGRU": "#1f77b4"}
    for name, sub in predictions[predictions[TEST_ID_COLUMN] == 34].groupby("model", sort=True):
        base_name = str(name).split("_seed")[0]
        if base_name == "FrozenPriorEventHeads":
            continue
        one = sub.sort_values(SAMPLE_ID_COLUMN)
        tp = one["time_hours"].to_numpy(dtype=np.float64)
        fp = (tp >= 41.30) & (tp <= 41.93)
        ax.plot(tp[fp], one.loc[fp, "HI_pred"], linewidth=1.35, color=colors.get(base_name, "#3A7D44"), label=base_name.replace("PGRGRU", " PG-RGRU"))
    ax.axhspan(0.0, RUL_FAILURE_THRESHOLD, color="#F2F5F9", zorder=0)
    ax.axhline(RUL_FAILURE_THRESHOLD, color="#687385", linewidth=0.8, linestyle=(0, (1.2, 1.7)))
    ax.set_xlim(41.30, 41.93)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("HI")
    ax.set_xlabel("Elapsed aging time (h)")
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.18)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"Figure_Test34_fpt_event_HI.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def write_definition(thresholds: dict[str, float]) -> None:
    definition = {
        "causal_label": "low-lag EWMA damage HI built with current and past corrected Rds(on) samples only",
        "fpt_rule": (
            f"FPT onset is the first window where trailing damage rate or trailing damage increment exceeds "
            f"train-calibrated FPT thresholds for {FPT_PERSIST_WINDOWS} consecutive windows."
        ),
        "fpt_event_rule": (
            f"FPT event-in-horizon target is positive from {FPT_EVENT_HORIZON_HOURS:.2f} h before that causal onset "
            f"through the onset window."
        ),
        "warning_rule": (
            f"Event-in-horizon warning inherited from the causal HI threshold event: HI<={RUL_FAILURE_THRESHOLD:.2f} "
            f"within {inc.WARNING_HORIZON_HOURS:.2f} h or post-threshold state."
        ),
        "thresholds_train_only": thresholds,
        "probability_thresholds": {
            "fpt_event": ONSET_PROB_THRESHOLD,
            "warning_event": WARNING_PROB_THRESHOLD,
        },
    }
    (OUT_DIR / "target_definition.json").write_text(json.dumps(definition, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original_df, causal_df, df, onset_reference, calibration, thresholds = build_causal_onset_dataframe()
    loaders, mu, sigma = build_loaders(df)
    baseline_hi, baseline_rul = baseline_metrics(df)
    write_definition(thresholds)

    original_df.to_csv(OUT_DIR / "offline_centered_reference_dataframe.csv", index=False)
    causal_df.to_csv(OUT_DIR / "causal_lowlag_reference_dataframe.csv", index=False)
    df.to_csv(OUT_DIR / "causal_fpt_event_training_dataframe.csv", index=False)
    onset_reference.to_csv(OUT_DIR / "onset_reference_by_test.csv", index=False)
    baseline_hi.to_csv(OUT_DIR / "baseline_hi_metrics.csv", index=False)
    baseline_rul.to_csv(OUT_DIR / "baseline_hi_threshold_rul_metrics.csv", index=False)

    outputs = [run_one(config, seed, df, loaders, mu, sigma, calibration) for config in CONFIGS for seed in SEEDS]
    preds = pd.concat([out["pred"] for out in outputs], ignore_index=True)
    histories = pd.concat([out["history"] for out in outputs], ignore_index=True)
    hi_all = pd.concat([out["hi"] for out in outputs], ignore_index=True)
    rul_all = pd.concat([out["rul"] for out in outputs], ignore_index=True)
    rul_by_test = pd.concat([out["rul_by_test"] for out in outputs], ignore_index=True)
    increment_all = pd.concat([out["increment"] for out in outputs], ignore_index=True)
    fpt_all = pd.concat([out["fpt"] for out in outputs], ignore_index=True)
    fpt_by_test = pd.concat([out["fpt_by_test"] for out in outputs], ignore_index=True)
    warning_all = pd.concat([out["warning"] for out in outputs], ignore_index=True)
    warning_by_test = pd.concat([out["warning_by_test"] for out in outputs], ignore_index=True)

    histories.to_csv(OUT_DIR / "training_history_all.csv", index=False)
    preds.to_csv(OUT_DIR / "predictions_all.csv", index=False)
    hi_all.to_csv(OUT_DIR / "proposed_hi_metrics_by_seed.csv", index=False)
    rul_all.to_csv(OUT_DIR / "proposed_hi_threshold_rul_metrics_by_seed.csv", index=False)
    rul_by_test.to_csv(OUT_DIR / "proposed_hi_threshold_rul_by_test.csv", index=False)
    increment_all.to_csv(OUT_DIR / "proposed_increment_gate_metrics_by_seed.csv", index=False)
    fpt_all.to_csv(OUT_DIR / "proposed_fpt_metrics_by_seed.csv", index=False)
    fpt_by_test.to_csv(OUT_DIR / "proposed_fpt_by_test.csv", index=False)
    warning_all.to_csv(OUT_DIR / "proposed_warning_metrics_by_seed.csv", index=False)
    warning_by_test.to_csv(OUT_DIR / "proposed_warning_by_test.csv", index=False)

    repeats = []
    for config in CONFIGS:
        name_filter = hi_all["model"].astype(str).str.startswith(config.name)
        hi_subset = hi_all[name_filter].copy()
        rul_subset = rul_all[rul_all["model"].astype(str).str.startswith(config.name)].copy()
        inc_subset = increment_all[increment_all["model"].astype(str).str.startswith(config.name)].copy()
        fpt_subset = fpt_all[fpt_all["model"].astype(str).str.startswith(config.name)].copy()
        warning_subset = warning_all[warning_all["model"].astype(str).str.startswith(config.name)].copy()
        summary = pd.concat(
            [
                aggregate_seed_metrics(hi_subset, ["mae", "rmse"], "HI"),
                aggregate_seed_metrics(rul_subset, ["pred_cross_rate", "eol_mae_hours", "rul_mae_hours"], "HI_threshold_RUL"),
                aggregate_seed_metrics(inc_subset, ["increment_mae", "increment_rmse", "gate_mean", "gate_p95", "extra_increment_mean", "multiplier_mean"], "Increment_gate"),
                aggregate_seed_metrics(
                    fpt_subset,
                    [
                        "event_brier",
                        "false_event_window_rate",
                        "event_covered_tests",
                        "missed_event_tests",
                        "event_lead_time_error_mae_hours",
                    ],
                    "FPT_event",
                ),
                aggregate_seed_metrics(
                    warning_subset,
                    [
                        "warning_brier",
                        "false_warning_window_rate",
                        "warning_covered_tests",
                        "missed_warning_tests",
                        "missed_warning_rate",
                        "warning_lead_time_error_mae_hours",
                    ],
                    "Warning",
                ),
            ],
            ignore_index=True,
        )
        summary.insert(0, "config", config.name)
        repeats.append(summary)
    repeat_summary = pd.concat(repeats, ignore_index=True)
    repeat_summary.to_csv(OUT_DIR / "multiseed_summary.csv", index=False)

    selected = []
    for config in CONFIGS:
        candidates = warning_all[(warning_all["scope"] == "test") & warning_all["model"].astype(str).str.startswith(config.name)]
        ranked = candidates.merge(hi_all[hi_all["scope"] == "test"][["model", "seed", "rmse"]], on=["model", "seed"], how="left")
        best = ranked.sort_values(["missed_warning_tests", "false_warning_window_rate", "warning_brier", "rmse", "seed"]).iloc[0]
        selected.append(preds[(preds["model"] == best["model"]) & (preds["seed"] == int(best["seed"]))])
    selected_pred = pd.concat(selected, ignore_index=True)
    selected_pred.to_csv(OUT_DIR / "selected_predictions_for_figures.csv", index=False)
    plot_test34_hi(df, selected_pred)

    summary = {
        "purpose": "prior-preserving FPT event-in-horizon and warning-probability experiment",
        "stable_submission_artifacts_overwritten": False,
        "seeds": SEEDS,
        "configs": [asdict(config) for config in CONFIGS],
        "fixed_test_split": sorted(df[df[SPLIT_COLUMN] == "test"][TEST_ID_COLUMN].unique().astype(int).tolist()),
        "train_only_onset_thresholds": thresholds,
        "baseline_test_hi": baseline_hi[baseline_hi["scope"] == "test"].to_dict(orient="records"),
        "repeat_summary": repeat_summary.to_dict(orient="records"),
    }
    (OUT_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved to:", OUT_DIR)
    print("\nTrain-only onset thresholds")
    print(pd.DataFrame([thresholds]).to_string(index=False))
    print("\nBaseline test HI")
    print(baseline_hi[baseline_hi["scope"] == "test"].to_string(index=False))
    print("\nProposed test HI by seed")
    print(hi_all[hi_all["scope"] == "test"].to_string(index=False))
    print("\nFPT metrics by seed")
    print(fpt_all[fpt_all["scope"] == "test"].to_string(index=False))
    print("\nWarning metrics by seed")
    print(warning_all[warning_all["scope"] == "test"].to_string(index=False))
    print("\nMultiseed summary")
    print(repeat_summary.to_string(index=False))


if __name__ == "__main__":
    main()
