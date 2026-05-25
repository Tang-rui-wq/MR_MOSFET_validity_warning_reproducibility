from __future__ import annotations

import json
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
    load_all_samples,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR = ROOT / "weights"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
SPLIT_DIRS = {split: RESULTS_DIR / split for split in ("train", "val", "test")}
RESULTS_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)
DIAGNOSTICS_DIR.mkdir(exist_ok=True)
for split_dir in SPLIT_DIRS.values():
    split_dir.mkdir(exist_ok=True)

LABEL_COLUMN = "HI_label"
OUTPUT_TAG = "hi_pinn_gru"
DEFAULT_RDS_FAILURE_DELTA = 0.20919872437861223
RDS_FAILURE_DELTA = DEFAULT_RDS_FAILURE_DELTA
RDS_EOL_CALIBRATION_QUANTILE = 0.25
RDS_EOL_DELTA_MIN = 0.15
RDS_EOL_DELTA_MAX = 0.35
RESIDUAL_LIMIT = 0.15
RDS_REFERENCE_MAX_FOR_TRAINING = 20.0
RDS_MIN_OBSERVABLE_SPAN = 0.04
RUL_FAILURE_THRESHOLD = 0.20
POST_EOL_HI_THRESHOLD = RUL_FAILURE_THRESHOLD
POST_EOL_KEEP_WINDOWS = 20

# Calibration-consistent paper split.
#
# Delta_Rds_EOL = 0.209199 is the lower quartile of the 95th-percentile
# Rds(on) residuals from the 26 training tests below. These tests must remain
# in the train split; otherwise the paper would calibrate the HI endpoint using
# validation/test information. The remaining valid tests are held out for
# validation and final evaluation.
PAPER_TRAIN_TEST_IDS = [
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    18,
    19,
    21,
    23,
    24,
    26,
    28,
    29,
    30,
    32,
    33,
    35,
    36,
    38,
    39,
    40,
]
PAPER_VAL_TEST_IDS = [37]
PAPER_TEST_TEST_IDS = [6, 13, 14, 20, 25, 34, 42]

# Network inputs follow the paper route: corrected electrical degradation
# features plus operating context. Formula-derived HI_proxy/DamageProxy are
# deliberately excluded so the network cannot replay an empirical proxy.
PINN_FEATURE_COLUMNS = [
    "supplyVoltage_mean",
    "packageTemperature_mean",
    "flangeTemperature_mean",
    "P_proxy_mean",
    "Rds_on_cond_median",
    "Rds_resid_online",
    "Tth_proxy_mean",
    "DegradationSlope_mean",
    "switchingFrequency_mean",
    "dutyCycle_mean",
]

# Physics variables: these are not network inputs. They are used only in the
# PINN-style loss to softly constrain the learned HI trajectory.
PINN_PHYS_COLUMNS = [
    "hi_physics_prior",
    "damage_physics",
    "rds_damage_online",
    "rds_confidence",
]


def prepare_hi_features(df: pd.DataFrame, rds_failure_delta: float = RDS_FAILURE_DELTA) -> pd.DataFrame:
    out = df.copy()
    out["age_window"] = 0.0
    out["age_sample"] = 0.0
    out["elapsed_hours"] = 0.0
    out["Rds_resid_online"] = 0.0
    out["rds_damage_online"] = 0.0
    out["rds_confidence"] = 1.0
    out["damage_physics"] = 0.0
    out["hi_physics_prior"] = 1.0
    for _, idx in out.groupby(TEST_ID_COLUMN).groups.items():
        sub = out.loc[idx].sort_values(RUN_ORDER_COLUMN)
        first_window = float(sub[SAMPLE_ID_COLUMN].iloc[0])
        first_sample = float(sub["sample_index_end"].iloc[0])
        first_time = float(sub["t_epoch_end"].iloc[0])
        resid = sub["Rds_on_temp_resid"].to_numpy(dtype=np.float64)
        ref_n = max(5, min(20, len(sub)))
        resid_ref = float(np.nanmedian(resid[:ref_n]))
        resid_online = np.maximum(resid - resid_ref, 0.0)
        rds_damage = np.clip(np.maximum.accumulate(resid_online) / rds_failure_delta, 0.0, 1.0)
        rds_confidence = rds_degradation_confidence(rds_damage)
        damage_physics = np.clip(rds_damage, 0.0, 1.0)
        damage_physics = np.maximum.accumulate(damage_physics)
        hi_physics_prior = np.clip(1.0 - damage_physics, 0.0, 1.0)
        out.loc[sub.index, "age_window"] = sub[SAMPLE_ID_COLUMN].to_numpy(dtype=np.float64) - first_window
        out.loc[sub.index, "age_sample"] = sub["sample_index_end"].to_numpy(dtype=np.float64) - first_sample
        out.loc[sub.index, "elapsed_hours"] = (sub["t_epoch_end"].to_numpy(dtype=np.float64) - first_time) * 24.0
        out.loc[sub.index, "Rds_resid_online"] = resid_online
        out.loc[sub.index, "rds_damage_online"] = rds_damage
        out.loc[sub.index, "rds_confidence"] = rds_confidence
        out.loc[sub.index, "damage_physics"] = damage_physics
        out.loc[sub.index, "hi_physics_prior"] = hi_physics_prior
    out["log_elapsed_hours"] = np.log1p(np.maximum(out["elapsed_hours"].to_numpy(dtype=np.float64), 0.0))
    out["log_thermal_exposure"] = np.log1p(np.maximum(out["ThermalExposure_end"].to_numpy(dtype=np.float64), 0.0))
    return out


def build_calibrated_modeling_dataframe() -> tuple[pd.DataFrame, list[dict[str, float]], list[dict[str, float]], dict[str, object]]:
    """Load samples, assign the paper split, calibrate Delta_Rds_EOL, and rebuild labels.

    The endpoint scale is calibrated strictly from the paper train split. This
    keeps the train/calibration definition consistent and avoids using any
    validation/test trajectory when identifying Delta_Rds_EOL.
    """
    df_raw = prepare_hi_features(load_all_samples(), DEFAULT_RDS_FAILURE_DELTA)
    df_valid, excluded_tests = drop_invalid_on_state_tests(df_raw)
    df_valid = assign_paper_test_level_splits(df_valid)
    calibration = calibrate_rds_failure_delta(df_valid)
    calibration.update(
        {
            "method": "paper_train_split_lower_quartile_of_per_test_rds_resid_p95",
            "calibration_train_tests": sorted(PAPER_TRAIN_TEST_IDS),
            "validation_tests": sorted(PAPER_VAL_TEST_IDS),
            "test_tests": sorted(PAPER_TEST_TEST_IDS),
            "note": (
                "Delta_Rds_EOL is calibrated only from the paper train split. "
                "No validation/test trajectory contributes to the endpoint scale."
            ),
        }
    )
    df_calibrated = apply_rds_failure_delta(df_valid, float(calibration["rds_failure_delta"]))
    df, truncated_tests = truncate_post_eol_windows(df_calibrated)
    return df, excluded_tests, truncated_tests, calibration


def assign_paper_test_level_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a fixed Test-level split consistent with Delta_Rds_EOL calibration."""
    out = df.copy()
    train_ids = set(PAPER_TRAIN_TEST_IDS)
    val_ids = set(PAPER_VAL_TEST_IDS)
    test_ids = set(PAPER_TEST_TEST_IDS)
    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"paper split test IDs overlap: {sorted(overlap)}")

    valid_ids = set(map(int, out[TEST_ID_COLUMN].unique()))
    missing = (train_ids | val_ids | test_ids) - valid_ids
    if missing:
        raise ValueError(f"paper split contains IDs not present after Rds quality filtering: {sorted(missing)}")

    split_by_test = {}
    for test_id in valid_ids:
        if test_id in train_ids:
            split_by_test[test_id] = "train"
        elif test_id in val_ids:
            split_by_test[test_id] = "val"
        elif test_id in test_ids:
            split_by_test[test_id] = "test"
        else:
            raise ValueError(f"valid Test_{test_id} is not assigned to a paper split")
    out[SPLIT_COLUMN] = out[TEST_ID_COLUMN].map(split_by_test).fillna(out[SPLIT_COLUMN])
    return out


def pick_evenly_spaced(items: list[int], count: int) -> list[int]:
    """Pick representative IDs over a sorted list without random sampling."""
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    positions: list[int] = []
    n = len(items)
    for i in range(count):
        pos = int(round((i + 0.5) * n / count - 0.5))
        pos = max(0, min(n - 1, pos))
        while pos in positions and pos + 1 < n:
            pos += 1
        while pos in positions and pos - 1 >= 0:
            pos -= 1
        positions.append(pos)
    return [items[pos] for pos in sorted(positions)]


def calibrate_rds_failure_delta(df: pd.DataFrame) -> dict[str, object]:
    """Calibrate the Rds(on) EOL delta using only valid training tests.

    Each training test contributes the 95th percentile of its online
    temperature-compensated Rds(on) residual. The lower quartile is used as a
    conservative endpoint threshold: it avoids being dominated by tests that
    continued long after failure, while still coming from observed train data.
    """
    candidates: list[dict[str, float]] = []
    train = df[df[SPLIT_COLUMN] == "train"].copy()
    for test_id, sub in train.groupby(TEST_ID_COLUMN, sort=True):
        sub = sub.sort_values(RUN_ORDER_COLUMN)
        resid = sub["Rds_resid_online"].to_numpy(dtype=np.float64)
        resid = resid[np.isfinite(resid)]
        if resid.size < 5:
            continue
        terminal_p95 = float(np.nanpercentile(resid, 95))
        terminal_p90 = float(np.nanpercentile(resid, 90))
        terminal_max = float(np.nanmax(resid))
        trend = series_trend(resid)
        if not np.isfinite(terminal_p95) or terminal_p95 < RDS_MIN_OBSERVABLE_SPAN or trend < -0.20:
            continue
        candidates.append(
            {
                "test_id": int(test_id),
                "rds_resid_p90": terminal_p90,
                "rds_resid_p95": terminal_p95,
                "rds_resid_max": terminal_max,
                "rds_trend": float(trend),
            }
        )
    if not candidates:
        return {
            "method": "fallback_default_no_valid_train_candidates",
            "rds_failure_delta": DEFAULT_RDS_FAILURE_DELTA,
            "raw_delta": DEFAULT_RDS_FAILURE_DELTA,
            "quantile": RDS_EOL_CALIBRATION_QUANTILE,
            "clip_min": RDS_EOL_DELTA_MIN,
            "clip_max": RDS_EOL_DELTA_MAX,
            "n_candidates": 0,
            "candidates": [],
        }
    values = np.asarray([row["rds_resid_p95"] for row in candidates], dtype=np.float64)
    raw_delta = float(np.nanquantile(values, RDS_EOL_CALIBRATION_QUANTILE))
    calibrated = float(np.clip(raw_delta, RDS_EOL_DELTA_MIN, RDS_EOL_DELTA_MAX))
    return {
        "method": "train_split_lower_quartile_of_per_test_rds_resid_p95",
        "rds_failure_delta": calibrated,
        "raw_delta": raw_delta,
        "quantile": RDS_EOL_CALIBRATION_QUANTILE,
        "clip_min": RDS_EOL_DELTA_MIN,
        "clip_max": RDS_EOL_DELTA_MAX,
        "n_candidates": len(candidates),
        "candidates": candidates,
    }


def apply_rds_failure_delta(df: pd.DataFrame, rds_failure_delta: float) -> pd.DataFrame:
    """Recompute HI labels and physics prior using the supplied EOL delta."""
    out = df.copy()
    if "Damage_label" not in out.columns:
        out["Damage_label"] = np.nan
    for _, idx in out.groupby(TEST_ID_COLUMN).groups.items():
        sub = out.loc[idx].sort_values(RUN_ORDER_COLUMN)
        resid_online = sub["Rds_resid_online"].to_numpy(dtype=np.float64)
        rds_damage = np.clip(np.maximum.accumulate(resid_online) / rds_failure_delta, 0.0, 1.0)
        rds_confidence = rds_degradation_confidence(rds_damage)
        damage_physics = np.maximum.accumulate(np.clip(rds_damage, 0.0, 1.0))
        hi_physics_prior = np.clip(1.0 - damage_physics, 0.0, 1.0)
        hi_label, damage_label = compute_rds_hi_label_from_resid(
            sub["Rds_on_temp_resid"].to_numpy(dtype=np.float64),
            rds_failure_delta,
        )
        out.loc[sub.index, "rds_damage_online"] = rds_damage
        out.loc[sub.index, "rds_confidence"] = rds_confidence
        out.loc[sub.index, "damage_physics"] = damage_physics
        out.loc[sub.index, "hi_physics_prior"] = hi_physics_prior
        out.loc[sub.index, LABEL_COLUMN] = hi_label
        out.loc[sub.index, "Damage_label"] = damage_label
    return out


def compute_rds_hi_label_from_resid(rds_temp_resid: np.ndarray, rds_failure_delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Python mirror of the MATLAB HI label rule after temp compensation."""
    resid = fill_series(rds_temp_resid)
    n = resid.size
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    ref_count = max(8, min(round(0.18 * n), max(8, int(np.floor(0.35 * n)))))
    ref_count = min(ref_count, n)
    resid_ref = float(np.nanmedian(resid[:ref_count]))
    damage_raw = (resid - resid_ref) / max(rds_failure_delta, np.finfo(float).eps)
    damage_raw = smooth_series(damage_raw, n)
    damage_raw = np.maximum.accumulate(np.clip(damage_raw, 0.0, 1.0))
    damage_label = smooth_series(damage_raw, n)
    damage_label = np.maximum.accumulate(np.clip(damage_label, 0.0, 1.0))
    hi_label = np.clip(1.0 - damage_label, 0.0, 1.0)
    return hi_label, damage_label


def fill_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x, dtype=np.float64)
    if np.all(good):
        return x
    t = np.arange(x.size, dtype=np.float64)
    x[~good] = np.interp(t[~good], t[good], x[good])
    return x


def smooth_series(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 3:
        return x
    span1 = max(5, min(21, 2 * int(np.floor(n / 30)) + 1))
    span2 = max(3, min(11, 2 * int(np.floor(n / 60)) + 1))
    y = pd.Series(x).rolling(window=span2, center=True, min_periods=1).median()
    y = y.rolling(window=span1, center=True, min_periods=1).mean()
    return y.to_numpy(dtype=np.float64)


def rds_degradation_confidence(rds_damage: np.ndarray) -> float:
    """Confidence that Rds(on) is a valid monotonic aging precursor."""
    rds_damage = np.asarray(rds_damage, dtype=np.float64)
    if rds_damage.size < 3:
        return 0.0
    rds_span = float(np.nanmax(rds_damage) - np.nanmin(rds_damage))
    rds_trend = series_trend(rds_damage)
    if rds_span < 0.05:
        return 0.0
    if rds_trend < 0.05:
        return 0.25
    if rds_span >= 0.40 and rds_trend >= 0.25:
        return 1.0
    return 0.65


def series_trend(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    if np.count_nonzero(good) < 3:
        return 0.0
    y = x[good]
    t = np.arange(y.size, dtype=np.float64)
    y = y - np.mean(y)
    t = t - np.mean(t)
    den = np.sqrt(np.sum(y ** 2) * np.sum(t ** 2))
    if den <= np.finfo(float).eps:
        return 0.0
    return float(np.sum(y * t) / den)


def drop_invalid_on_state_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Remove tests whose Rds(on) trajectory cannot support an electrical HI.

    Rds(on)=Vds/Id is used only as an ON-state degradation precursor. If the
    early reference or the full trajectory is not physically interpretable,
    that Test is excluded from the Rds(on)-based paper model and reported.
    """
    excluded: list[dict[str, float]] = []
    keep_ids: list[int] = []
    for test_id, sub in df.groupby(TEST_ID_COLUMN, sort=True):
        sub = sub.sort_values(RUN_ORDER_COLUMN)
        ref_n = max(5, min(20, int(np.ceil(0.10 * len(sub)))))
        rds_ref = float(np.nanmedian(sub["Rds_on_cond_median"].iloc[:ref_n]))
        p_ref = float(np.nanmedian(sub["P_proxy_mean"].iloc[:ref_n]))
        rds_resid = sub["Rds_resid_online"].to_numpy(dtype=np.float64)
        rds_span = float(np.nanpercentile(rds_resid, 95) - np.nanmedian(rds_resid[:ref_n]))
        rds_trend = series_trend(rds_resid)
        reason = ""
        if (not np.isfinite(rds_ref)) or rds_ref <= 0 or rds_ref > RDS_REFERENCE_MAX_FOR_TRAINING:
            reason = "invalid_on_state_rds_reference"
        elif (not np.isfinite(rds_span)) or rds_span < RDS_MIN_OBSERVABLE_SPAN:
            reason = "insufficient_observable_rds_degradation"
        elif rds_trend < -0.20:
            reason = "nonmonotonic_rds_degradation"
        if reason:
            excluded.append(
                {
                    "test_id": int(test_id),
                    "rds_ref": rds_ref,
                    "rds_span_p95": rds_span,
                    "rds_trend": rds_trend,
                    "p_ref": p_ref,
                    "reason": reason,
                }
            )
        else:
            keep_ids.append(int(test_id))
    filtered = df[df[TEST_ID_COLUMN].isin(keep_ids)].copy().reset_index(drop=True)
    return filtered, excluded


def truncate_post_eol_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Drop long post-failure tails so training focuses on degradation.

    The NASA files can contain many windows after the Rds(on)-based HI has
    already reached failure. Keeping all of them creates visually long flat
    HI=0 lines and overweights the easiest part of the learning problem.
    """
    parts: list[pd.DataFrame] = []
    report: list[dict[str, float]] = []
    for test_id, sub in df.groupby(TEST_ID_COLUMN, sort=True):
        sub = sub.sort_values(RUN_ORDER_COLUMN).copy()
        hi = sub[LABEL_COLUMN].to_numpy(dtype=np.float64)
        hit = np.flatnonzero(hi <= POST_EOL_HI_THRESHOLD)
        original_n = int(len(sub))
        if hit.size:
            keep_n = min(original_n, int(hit[0]) + 1 + POST_EOL_KEEP_WINDOWS)
            trimmed = sub.iloc[:keep_n].copy()
        else:
            keep_n = original_n
            trimmed = sub
        parts.append(trimmed)
        if keep_n < original_n:
            report.append(
                {
                    "test_id": int(test_id),
                    "original_windows": original_n,
                    "kept_windows": keep_n,
                    "dropped_post_eol_windows": original_n - keep_n,
                    "first_hi_le_threshold_window": int(hit[0] + 1),
                    "post_eol_hi_threshold": POST_EOL_HI_THRESHOLD,
                }
            )
    if not parts:
        return df.copy(), report
    return pd.concat(parts, ignore_index=True), report


def summarize_valid_test_splits(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for test_id, sub in df.groupby(TEST_ID_COLUMN, sort=True):
        sub = sub.sort_values(RUN_ORDER_COLUMN)
        t = sub["t_epoch_end"].to_numpy(dtype=np.float64)
        elapsed_h = float((np.nanmax(t) - np.nanmin(t)) * 24.0) if t.size and np.isfinite(t).any() else float(len(sub))
        hi = sub[LABEL_COLUMN].to_numpy(dtype=np.float64)
        rds_resid = sub["Rds_resid_online"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "test_id": int(test_id),
                "split": str(sub[SPLIT_COLUMN].iloc[0]),
                "n_windows": int(len(sub)),
                "duration_h": elapsed_h,
                "hi_start": float(hi[0]) if hi.size else np.nan,
                "hi_end": float(hi[-1]) if hi.size else np.nan,
                "hi_min": float(np.nanmin(hi)) if hi.size else np.nan,
                "true_eol_reached": bool(np.nanmin(hi) <= RUL_FAILURE_THRESHOLD) if hi.size else False,
                "rds_resid_p95": float(np.nanpercentile(rds_resid, 95)) if rds_resid.size else np.nan,
                "rds_trend": float(series_trend(rds_resid)) if rds_resid.size else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "duration_h", "test_id"]).reset_index(drop=True)


def summarize_excluded_tests(excluded_tests: list[dict[str, float]]) -> pd.DataFrame:
    if not excluded_tests:
        return pd.DataFrame(
            columns=["test_id", "reason", "rds_ref", "rds_span_p95", "rds_trend", "p_ref"]
        )
    return pd.DataFrame(excluded_tests).sort_values("test_id").reset_index(drop=True)


class HISequenceDataset(Dataset):
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

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_sequences(batch):
    test_ids, splits, xs, ys, phys, sample_ids = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=-1.0)
    phys_pad = pad_sequence(phys, batch_first=True)
    id_pad = pad_sequence(sample_ids, batch_first=True, padding_value=-1)
    mask = y_pad >= 0
    return torch.tensor(test_ids, dtype=torch.long), list(splits), x_pad, y_pad, phys_pad, id_pad, mask


class HIGRU(nn.Module):
    """Physics-informed GRU for HI prediction.

    The network predicts a bounded residual around the physical HI prior.
    The residual range is deliberately wide so the GRU can correct a poor
    prior on out-of-distribution NASA tests instead of being locked to it.
    """

    def __init__(self, in_dim: int, hidden: int = 72):
        super().__init__()
        self.in_dim = in_dim
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            # Current electro-thermal features are concatenated to the recurrent
            # state, so abrupt RDS(on) degradation is not over-smoothed by GRU
            # memory alone.
            nn.Linear(hidden + in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        hx = torch.cat([h, x], dim=-1)
        residual = RESIDUAL_LIMIT * torch.tanh(self.head(hx).squeeze(-1))
        hi_prior = phys[..., 0]
        return torch.clamp(hi_prior + residual, 0.0, 1.0)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 0.050) -> torch.Tensor:
    err = pred[mask] - target[mask]
    abs_err = torch.abs(err)
    huber = torch.where(abs_err <= delta, 0.5 * (err ** 2) / delta, abs_err - 0.5 * delta)
    weights = 1.0 + 1.8 * (1.0 - target[mask])
    return torch.mean(weights * huber)


def smoothness_penalty(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    penalties = []
    for i in range(pred.shape[0]):
        valid = pred[i][mask[i]]
        if valid.numel() <= 2:
            continue
        diff = valid[1:] - valid[:-1]
        penalties.append(torch.mean(diff ** 2))
    if not penalties:
        return pred.new_tensor(0.0)
    return torch.stack(penalties).mean()


def monotonic_penalty(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    penalties = []
    for i in range(pred.shape[0]):
        valid = pred[i][mask[i]]
        if valid.numel() <= 2:
            continue
        upward = torch.relu(valid[1:] - valid[:-1])
        penalties.append(torch.mean(upward ** 2))
    if not penalties:
        return pred.new_tensor(0.0)
    return torch.stack(penalties).mean()


def boundary_penalty(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    penalties = []
    for i in range(pred.shape[0]):
        valid_pred = pred[i][mask[i]]
        valid_target = target[i][mask[i]]
        if valid_pred.numel() == 0:
            continue
        penalties.append((valid_pred[0] - valid_target[0]) ** 2)
        penalties.append((valid_pred[-1] - valid_target[-1]) ** 2)
    if not penalties:
        return pred.new_tensor(0.0)
    return torch.stack(penalties).mean()


def physics_prior_penalty(pred: torch.Tensor, phys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Keep HI close to the corrected Rds(on) electrical-degradation prior."""
    hi_prior = phys[..., 0]
    err = pred[mask] - hi_prior[mask]
    abs_err = torch.abs(err)
    huber = torch.where(abs_err <= 0.08, 0.5 * (err ** 2) / 0.08, abs_err - 0.04)
    return torch.mean(huber) if huber.numel() else pred.new_tensor(0.0)


def damage_ode_penalty(pred: torch.Tensor, phys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """PINN-HI residual: d(1-HI)/dt should follow Rds(on) damage.

    This is the discrete counterpart of dD/dt from the corrected Rds drift.
    It does not force the network to copy the label point by point; it only
    regularizes the degradation direction and rate.
    """
    penalties = []
    damage_phys = phys[..., 1]
    for i in range(pred.shape[0]):
        valid_pred = pred[i][mask[i]]
        valid_damage = damage_phys[i][mask[i]]
        if valid_pred.numel() <= 2:
            continue
        d_pred = (1.0 - valid_pred[1:]) - (1.0 - valid_pred[:-1])
        d_phys = valid_damage[1:] - valid_damage[:-1]
        weight = 1.0 + 4.0 * torch.clamp(d_phys, min=0.0)
        penalties.append(torch.mean(weight * (d_pred - d_phys) ** 2))
    if not penalties:
        return pred.new_tensor(0.0)
    return torch.stack(penalties).mean()


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
    all_abs = torch.cat(abs_err) if abs_err else torch.tensor([0.0])
    all_sq = torch.cat(sq_err) if sq_err else torch.tensor([0.0])
    return {"mae": float(all_abs.mean().item()), "rmse": float(torch.sqrt(all_sq.mean()).item())}


def predict_all(model: nn.Module, df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray) -> pd.DataFrame:
    dataset = HISequenceDataset(df, mu, sigma)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_sequences)
    rows = []
    model.eval()
    with torch.no_grad():
        for test_ids, splits, x, y, phys, sample_ids, mask in loader:
            pred = model(x, phys)
            test_id = int(test_ids.item())
            split = splits[0]
            valid_ids = sample_ids[0][mask[0]].cpu().numpy()
            valid_y = y[0][mask[0]].cpu().numpy()
            valid_phys = phys[0][mask[0]].cpu().numpy()
            valid_pred = pred[0][mask[0]].cpu().numpy()
            sub = df[df[TEST_ID_COLUMN] == test_id].sort_values(RUN_ORDER_COLUMN).reset_index(drop=True)
            meta_cols = [
                "sample_index_start",
                "sample_index_end",
                "t_epoch_start",
                "t_epoch_end",
                "source_run_start",
                "source_run_end",
            ]
            meta = sub.reindex(columns=meta_cols).iloc[: len(valid_ids)]
            for meta_row, sid, yy, ph, pp in zip(
                meta.to_dict(orient="records"), valid_ids, valid_y, valid_phys, valid_pred
            ):
                rows.append(
                    {
                        TEST_ID_COLUMN: test_id,
                        SAMPLE_ID_COLUMN: int(sid),
                        SPLIT_COLUMN: split,
                        "HI_true": float(yy),
                        "HI_physics_prior": float(ph[0]),
                        "Damage_physics": float(ph[1]),
                        "HI_pred_gru": float(pp),
                        "abs_err": float(abs(pp - yy)),
                        **meta_row,
                    }
                )
    return pd.DataFrame(rows)


def metrics_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub in pred.groupby(SPLIT_COLUMN, sort=True):
        err = sub["HI_pred_gru"].to_numpy(dtype=np.float64) - sub["HI_true"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": "PINN_GRU",
                "scope": scope,
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "n_windows": len(sub),
            }
        )
    for test_id, sub in pred.groupby(TEST_ID_COLUMN, sort=True):
        err = sub["HI_pred_gru"].to_numpy(dtype=np.float64) - sub["HI_true"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": "PINN_GRU",
                "scope": f"Test_{int(test_id)}",
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "n_windows": len(sub),
            }
        )
    return pd.DataFrame(rows)


def elapsed_axis_hours(one: pd.DataFrame) -> np.ndarray:
    """Return a per-test elapsed-time axis in hours for RUL metrics."""
    if "t_epoch_end" in one.columns and one["t_epoch_end"].notna().any():
        t = one["t_epoch_end"].to_numpy(dtype=np.float64)
        return (t - np.nanmin(t)) * 24.0
    if "sample_index_end" in one.columns and one["sample_index_end"].notna().any():
        t = one["sample_index_end"].to_numpy(dtype=np.float64)
        return t - np.nanmin(t)
    return np.arange(len(one), dtype=np.float64)


def first_threshold_crossing_time(time: np.ndarray, hi: np.ndarray, threshold: float) -> float:
    """First linearly interpolated time when HI falls below the failure threshold."""
    time = np.asarray(time, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    good = np.isfinite(time) & np.isfinite(hi)
    time = time[good]
    hi = hi[good]
    if time.size == 0:
        return np.nan
    hit = np.flatnonzero(hi <= threshold)
    if hit.size == 0:
        return np.nan
    i = int(hit[0])
    if i == 0:
        return float(time[0])
    t0, t1 = float(time[i - 1]), float(time[i])
    h0, h1 = float(hi[i - 1]), float(hi[i])
    if abs(h1 - h0) <= np.finfo(float).eps:
        return t1
    alpha = (threshold - h0) / (h1 - h0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return t0 + alpha * (t1 - t0)


def build_rul_outputs(pred: pd.DataFrame, threshold: float = RUL_FAILURE_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert HI trajectories into threshold-based RUL metrics.

    This is an offline RUL metric computed from the predicted HI trajectory.
    Early-prediction RUL, where only a prefix of each test is visible, should be
    evaluated separately.
    """
    rul_rows: list[dict[str, float | int | str | bool]] = []
    test_rows: list[dict[str, float | int | str | bool]] = []

    for test_id, sub in pred.groupby(TEST_ID_COLUMN, sort=True):
        one = sub.sort_values(SAMPLE_ID_COLUMN).reset_index(drop=True)
        split = str(one[SPLIT_COLUMN].iloc[0])
        elapsed = elapsed_axis_hours(one)
        true_hi = one["HI_true"].to_numpy(dtype=np.float64)
        pred_hi = one["HI_pred_gru"].to_numpy(dtype=np.float64)
        prior_hi = one["HI_physics_prior"].to_numpy(dtype=np.float64) if "HI_physics_prior" in one.columns else None

        true_eol = first_threshold_crossing_time(elapsed, true_hi, threshold)
        pred_eol = first_threshold_crossing_time(elapsed, pred_hi, threshold)
        prior_eol = first_threshold_crossing_time(elapsed, prior_hi, threshold) if prior_hi is not None else np.nan
        true_reached = bool(np.isfinite(true_eol))
        pred_reached = bool(np.isfinite(pred_eol))
        prior_reached = bool(np.isfinite(prior_eol))

        true_rul = np.maximum(true_eol - elapsed, 0.0) if true_reached else np.full_like(elapsed, np.nan)
        pred_rul = np.maximum(pred_eol - elapsed, 0.0) if pred_reached else np.full_like(elapsed, np.nan)
        prior_rul = np.maximum(prior_eol - elapsed, 0.0) if prior_reached else np.full_like(elapsed, np.nan)
        pred_rul_err = pred_rul - true_rul
        prior_rul_err = prior_rul - true_rul

        for row, t, rt, rp, rprior, ep, eprior in zip(
            one.to_dict(orient="records"),
            elapsed,
            true_rul,
            pred_rul,
            prior_rul,
            pred_rul_err,
            prior_rul_err,
        ):
            rul_rows.append(
                {
                    **row,
                    "rul_threshold": threshold,
                    "elapsed_hours": float(t),
                    "true_eol_hours": float(true_eol) if true_reached else np.nan,
                    "pred_eol_gru_hours": float(pred_eol) if pred_reached else np.nan,
                    "pred_eol_physics_prior_hours": float(prior_eol) if prior_reached else np.nan,
                    "rul_true_hours": float(rt),
                    "rul_pred_gru_hours": float(rp),
                    "rul_pred_physics_prior_hours": float(rprior),
                    "rul_err_gru_hours": float(ep),
                    "rul_abs_err_gru_hours": float(abs(ep)) if np.isfinite(ep) else np.nan,
                    "rul_err_physics_prior_hours": float(eprior),
                    "rul_abs_err_physics_prior_hours": float(abs(eprior)) if np.isfinite(eprior) else np.nan,
                    "pred_threshold_reached": pred_reached,
                    "physics_prior_threshold_reached": prior_reached,
                }
            )

        finite_pred = np.isfinite(pred_rul_err)
        finite_prior = np.isfinite(prior_rul_err)
        test_rows.append(
            {
                "model": "PINN_GRU",
                TEST_ID_COLUMN: int(test_id),
                SPLIT_COLUMN: split,
                "rul_threshold": threshold,
                "n_windows": int(len(one)),
                "true_threshold_reached": true_reached,
                "pred_threshold_reached": pred_reached,
                "physics_prior_threshold_reached": prior_reached,
                "true_eol_hours": float(true_eol) if true_reached else np.nan,
                "pred_eol_gru_hours": float(pred_eol) if pred_reached else np.nan,
                "pred_eol_physics_prior_hours": float(prior_eol) if prior_reached else np.nan,
                "eol_err_gru_hours": float(pred_eol - true_eol) if pred_reached and true_reached else np.nan,
                "eol_abs_err_gru_hours": float(abs(pred_eol - true_eol)) if pred_reached and true_reached else np.nan,
                "eol_err_physics_prior_hours": float(prior_eol - true_eol) if prior_reached and true_reached else np.nan,
                "eol_abs_err_physics_prior_hours": float(abs(prior_eol - true_eol)) if prior_reached and true_reached else np.nan,
                "rul_mae_gru_hours": float(np.nanmean(np.abs(pred_rul_err[finite_pred]))) if np.any(finite_pred) else np.nan,
                "rul_rmse_gru_hours": float(np.sqrt(np.nanmean(pred_rul_err[finite_pred] ** 2))) if np.any(finite_pred) else np.nan,
                "rul_mae_physics_prior_hours": float(np.nanmean(np.abs(prior_rul_err[finite_prior]))) if np.any(finite_prior) else np.nan,
                "rul_rmse_physics_prior_hours": float(np.sqrt(np.nanmean(prior_rul_err[finite_prior] ** 2))) if np.any(finite_prior) else np.nan,
            }
        )

    rul = pd.DataFrame(rul_rows)
    by_test = pd.DataFrame(test_rows)
    aggregate_rows = []
    for scope, sub in by_test.groupby(SPLIT_COLUMN, sort=True):
        aggregate_rows.append(rul_aggregate_row(scope, sub))
    aggregate_rows.append(rul_aggregate_row("all", by_test))
    aggregate = pd.DataFrame(aggregate_rows)
    return rul, by_test, aggregate


def rul_aggregate_row(scope: str, sub: pd.DataFrame) -> dict[str, float | int | str]:
    pred_valid = sub["pred_threshold_reached"].to_numpy(dtype=bool) & sub["true_threshold_reached"].to_numpy(dtype=bool)
    prior_valid = sub["physics_prior_threshold_reached"].to_numpy(dtype=bool) & sub["true_threshold_reached"].to_numpy(dtype=bool)

    def _mae(col: str, valid: np.ndarray) -> float:
        x = sub.loc[valid, col].to_numpy(dtype=np.float64)
        return float(np.nanmean(np.abs(x))) if x.size else np.nan

    def _rmse(col: str, valid: np.ndarray) -> float:
        x = sub.loc[valid, col].to_numpy(dtype=np.float64)
        return float(np.sqrt(np.nanmean(x ** 2))) if x.size else np.nan

    n_tests = int(len(sub))
    return {
        "model": "PINN_GRU",
        "scope": scope,
        "rul_threshold": RUL_FAILURE_THRESHOLD,
        "n_tests": n_tests,
        "pred_crossed_tests": int(np.count_nonzero(sub["pred_threshold_reached"].to_numpy(dtype=bool))),
        "pred_missed_tests": int(n_tests - np.count_nonzero(sub["pred_threshold_reached"].to_numpy(dtype=bool))),
        "pred_cross_rate": float(np.mean(sub["pred_threshold_reached"].to_numpy(dtype=bool))) if n_tests else np.nan,
        "eol_mae_gru_hours": _mae("eol_err_gru_hours", pred_valid),
        "eol_rmse_gru_hours": _rmse("eol_err_gru_hours", pred_valid),
        "rul_mae_gru_hours": _mae("rul_mae_gru_hours", pred_valid),
        "rul_rmse_gru_hours": _rmse("rul_rmse_gru_hours", pred_valid),
        "physics_prior_crossed_tests": int(np.count_nonzero(sub["physics_prior_threshold_reached"].to_numpy(dtype=bool))),
        "physics_prior_missed_tests": int(n_tests - np.count_nonzero(sub["physics_prior_threshold_reached"].to_numpy(dtype=bool))),
        "physics_prior_cross_rate": float(np.mean(sub["physics_prior_threshold_reached"].to_numpy(dtype=bool))) if n_tests else np.nan,
        "eol_mae_physics_prior_hours": _mae("eol_err_physics_prior_hours", prior_valid),
        "eol_rmse_physics_prior_hours": _rmse("eol_err_physics_prior_hours", prior_valid),
        "rul_mae_physics_prior_hours": _mae("rul_mae_physics_prior_hours", prior_valid),
        "rul_rmse_physics_prior_hours": _rmse("rul_rmse_physics_prior_hours", prior_valid),
    }


def export_rul_outputs(pred: pd.DataFrame, results_dir: Path = RESULTS_DIR, output_tag: str = OUTPUT_TAG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rul, by_test, aggregate = build_rul_outputs(pred)
    rul.to_csv(results_dir / f"nasa_mosfet_{output_tag}_rul_predictions.csv", index=False)
    by_test.to_csv(results_dir / f"nasa_mosfet_{output_tag}_rul_by_test.csv", index=False)
    aggregate.to_csv(results_dir / f"{output_tag}_rul_metrics.csv", index=False)

    for split in ("train", "val", "test"):
        split_dir = results_dir / split
        plot_rul_split(rul, split, "summary_rul_true_vs_pred.png", out_dir=split_dir)
        plot_rul_split_per_test(rul, split, out_dir=split_dir)
    return rul, by_test, aggregate


def clean_split_figures() -> None:
    """Remove stale split figures before writing the current run outputs."""
    for split_dir in SPLIT_DIRS.values():
        for png in split_dir.glob("summary_*.png"):
            png.unlink()
        per_test_dir = split_dir / "per_test"
        if per_test_dir.exists():
            for png in per_test_dir.glob("*.png"):
                png.unlink()


def plot_rul_split(rul: pd.DataFrame, split: str, out_name: str, out_dir: Path = RESULTS_DIR) -> None:
    sub = rul[rul[SPLIT_COLUMN] == split].copy()
    if sub.empty:
        return
    test_ids = sorted(sub[TEST_ID_COLUMN].unique())
    fig, axes = plt.subplots(len(test_ids), 1, figsize=(10, max(3.2, 3.0 * len(test_ids))), sharey=False)
    if len(test_ids) == 1:
        axes = [axes]
    for ax, test_id in zip(axes, test_ids):
        one = sub[sub[TEST_ID_COLUMN] == test_id].sort_values(SAMPLE_ID_COLUMN)
        x = one["elapsed_hours"].to_numpy(dtype=np.float64)
        ax.plot(x, one["rul_true_hours"], label="True RUL", linewidth=2.0)
        if one["rul_pred_physics_prior_hours"].notna().any():
            ax.plot(x, one["rul_pred_physics_prior_hours"], label="Physics prior RUL", linewidth=1.5, linestyle="--", alpha=0.75)
        if one["rul_pred_gru_hours"].notna().any():
            ax.plot(x, one["rul_pred_gru_hours"], label="GRU Pred RUL", linewidth=1.8)
        else:
            ax.text(0.02, 0.82, "GRU did not cross HI threshold", transform=ax.transAxes, fontsize=9)
        ax.set_title(f"RUL: Test_{int(test_id)}")
        ax.set_xlabel("Elapsed aging time (h)")
        ax.set_ylabel("RUL to HI threshold (h)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / out_name, dpi=180)
    plt.close(fig)


def plot_rul_split_per_test(rul: pd.DataFrame, split: str, out_dir: Path = RESULTS_DIR) -> None:
    sub = rul[rul[SPLIT_COLUMN] == split].copy()
    if sub.empty:
        return
    per_test_dir = out_dir / "per_test"
    per_test_dir.mkdir(parents=True, exist_ok=True)
    for test_id in sorted(sub[TEST_ID_COLUMN].unique()):
        one = sub[sub[TEST_ID_COLUMN] == test_id].sort_values(SAMPLE_ID_COLUMN)
        x = one["elapsed_hours"].to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.4))
        ax.plot(x, one["rul_true_hours"], label="True RUL", linewidth=2.0)
        if one["rul_pred_physics_prior_hours"].notna().any():
            ax.plot(x, one["rul_pred_physics_prior_hours"], label="Physics prior RUL", linewidth=1.5, linestyle="--", alpha=0.75)
        if one["rul_pred_gru_hours"].notna().any():
            ax.plot(x, one["rul_pred_gru_hours"], label="GRU Pred RUL", linewidth=1.8)
        else:
            ax.text(0.02, 0.82, "GRU did not cross HI threshold", transform=ax.transAxes, fontsize=9)
        true_eol = one["true_eol_hours"].dropna()
        pred_eol = one["pred_eol_gru_hours"].dropna()
        if not true_eol.empty:
            ax.axvline(float(true_eol.iloc[0]), color="tab:blue", linewidth=1.0, alpha=0.35)
        if not pred_eol.empty:
            ax.axvline(float(pred_eol.iloc[0]), color="tab:orange", linewidth=1.0, alpha=0.35)
        ax.set_title(f"RUL: Test_{int(test_id)} ({split})")
        ax.set_xlabel("Elapsed aging time (h)")
        ax.set_ylabel("RUL to HI threshold (h)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(per_test_dir / f"Test_{int(test_id):02d}_rul_true_vs_pred.png", dpi=180)
        plt.close(fig)


def plot_split(pred: pd.DataFrame, split: str, out_name: str, show_prior: bool = False, out_dir: Path = RESULTS_DIR) -> None:
    sub = pred[pred[SPLIT_COLUMN] == split].copy()
    if sub.empty:
        return
    test_ids = sorted(sub[TEST_ID_COLUMN].unique())
    fig, axes = plt.subplots(len(test_ids), 1, figsize=(10, max(3.2, 3.0 * len(test_ids))), sharey=True)
    if len(test_ids) == 1:
        axes = [axes]
    for ax, test_id in zip(axes, test_ids):
        one = sub[sub[TEST_ID_COLUMN] == test_id].sort_values(SAMPLE_ID_COLUMN)
        if "t_epoch_end" in one.columns and one["t_epoch_end"].notna().any():
            x = (one["t_epoch_end"].to_numpy(dtype=np.float64) - one["t_epoch_end"].min()) * 24.0
            xlabel = "Elapsed aging time (h)"
        elif "sample_index_end" in one.columns and one["sample_index_end"].notna().any():
            x = one["sample_index_end"].to_numpy(dtype=np.float64)
            xlabel = "Sample index / time step"
        else:
            x = np.arange(len(one))
            xlabel = "Window index"
        ax.plot(x, one["HI_true"], label="True HI", linewidth=2.0)
        if show_prior and "HI_physics_prior" in one.columns:
            ax.plot(x, one["HI_physics_prior"], label="Physics prior", linewidth=1.5, linestyle="--", alpha=0.75)
        ax.plot(x, one["HI_pred_gru"], label="GRU Pred", linewidth=1.8)
        ax.set_title(f"GRU: Test_{int(test_id)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized HI")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / out_name, dpi=180)
    plt.close(fig)


def plot_split_per_test(pred: pd.DataFrame, split: str, show_prior: bool = False, out_dir: Path = RESULTS_DIR) -> None:
    sub = pred[pred[SPLIT_COLUMN] == split].copy()
    if sub.empty:
        return
    suffix = "diagnostic_with_prior" if show_prior else "true_vs_pred"
    per_test_dir = out_dir / "per_test"
    per_test_dir.mkdir(parents=True, exist_ok=True)
    for test_id in sorted(sub[TEST_ID_COLUMN].unique()):
        one = sub[sub[TEST_ID_COLUMN] == test_id].sort_values(SAMPLE_ID_COLUMN)
        if "t_epoch_end" in one.columns and one["t_epoch_end"].notna().any():
            x = (one["t_epoch_end"].to_numpy(dtype=np.float64) - one["t_epoch_end"].min()) * 24.0
            xlabel = "Elapsed aging time (h)"
        elif "sample_index_end" in one.columns and one["sample_index_end"].notna().any():
            x = one["sample_index_end"].to_numpy(dtype=np.float64)
            xlabel = "Sample index / time step"
        else:
            x = np.arange(len(one))
            xlabel = "Window index"

        fig, ax = plt.subplots(1, 1, figsize=(10, 3.4))
        ax.plot(x, one["HI_true"], label="True HI", linewidth=2.0)
        if show_prior and "HI_physics_prior" in one.columns:
            ax.plot(x, one["HI_physics_prior"], label="Physics prior", linewidth=1.5, linestyle="--", alpha=0.75)
        ax.plot(x, one["HI_pred_gru"], label="GRU Pred", linewidth=1.8)
        ax.set_title(f"GRU: Test_{int(test_id)} ({split})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized HI")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(per_test_dir / f"Test_{int(test_id):02d}_{suffix}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    df, excluded_tests, truncated_tests, calibration = build_calibrated_modeling_dataframe()
    rds_failure_delta = float(calibration["rds_failure_delta"])
    split_summary = summarize_valid_test_splits(df)
    excluded_summary = summarize_excluded_tests(excluded_tests)
    split_summary.to_csv(RESULTS_DIR / "paper_test_level_split_summary.csv", index=False)
    excluded_summary.to_csv(RESULTS_DIR / "paper_excluded_invalid_tests.csv", index=False)
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    val_df = df[df[SPLIT_COLUMN] == "val"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    x_train = train_df[PINN_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma == 0] = 1.0

    train_ds = HISequenceDataset(train_df, mu, sigma)
    val_ds = HISequenceDataset(val_df, mu, sigma)
    test_ds = HISequenceDataset(test_df, mu, sigma)
    train_loader = DataLoader(train_ds, batch_size=min(8, len(train_ds)), shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_sequences)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_sequences)

    model = HIGRU(in_dim=len(PINN_FEATURE_COLUMNS), hidden=72)
    opt = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=5.0e-5)
    best_state = None
    best_val = float("inf")
    history = []

    for epoch in range(1, 261):
        model.train()
        epoch_losses = []
        for _, _, x, y, phys, _, mask in train_loader:
            pred = model(x, phys)
            loss = (
                masked_huber(pred, y, mask)
                + 0.002 * smoothness_penalty(pred, mask)
                + 0.025 * monotonic_penalty(pred, mask)
                + 0.080 * boundary_penalty(pred, y, mask)
                + 0.220 * physics_prior_penalty(pred, phys, mask)
                + 0.050 * damage_ode_penalty(pred, phys, mask)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_losses.append(float(loss.item()))

        if epoch % 5 == 0 or epoch == 1:
            train_metrics = evaluate(model, train_loader)
            val_metrics = evaluate(model, val_loader)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(epoch_losses)),
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                }
            )
            if val_metrics["rmse"] < best_val:
                best_val = val_metrics["rmse"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    pred = predict_all(model, df, mu, sigma)
    metrics = metrics_from_predictions(pred)
    pred.to_csv(RESULTS_DIR / f"nasa_mosfet_{OUTPUT_TAG}_predictions.csv", index=False)
    metrics.to_csv(RESULTS_DIR / f"{OUTPUT_TAG}_comparison_metrics.csv", index=False)
    clean_split_figures()
    _, _, rul_metrics = export_rul_outputs(pred)
    pd.DataFrame(history).to_csv(DIAGNOSTICS_DIR / f"nasa_mosfet_{OUTPUT_TAG}_history.csv", index=False)

    for split in ("train", "val", "test"):
        split_dir = SPLIT_DIRS[split]
        plot_split(pred, split, "summary_true_vs_pred.png", show_prior=False, out_dir=split_dir)
        plot_split(pred, split, "summary_diagnostic_with_prior.png", show_prior=True, out_dir=split_dir)
        plot_split_per_test(pred, split, show_prior=False, out_dir=split_dir)
        plot_split_per_test(pred, split, show_prior=True, out_dir=split_dir)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": PINN_FEATURE_COLUMNS,
            "mu": mu,
            "sigma": sigma,
            "label_column": LABEL_COLUMN,
            "residual_limit": RESIDUAL_LIMIT,
            "rds_failure_delta": rds_failure_delta,
            "rds_failure_delta_calibration": calibration,
            "note": "Electrical-degradation HI GRU: corrected temperature-compensated Rds(on) prior plus bounded GRU residual correction. Delta_Rds_EOL is calibrated only from the paper train split; HI_proxy/DamageProxy are not network inputs.",
        },
        WEIGHTS_DIR / f"nasa_mosfet_{OUTPUT_TAG}.pt",
    )
    summary = {
        "label_column": LABEL_COLUMN,
        "feature_columns": PINN_FEATURE_COLUMNS,
        "physics_columns": PINN_PHYS_COLUMNS,
        "loss": {
            "data_huber": 1.0,
            "smoothness": 0.002,
            "monotonicity": 0.025,
            "boundary": 0.080,
            "physics_prior": 0.220,
            "damage_ode_residual": 0.050,
        },
        "residual_limit": RESIDUAL_LIMIT,
        "rds_failure_delta": rds_failure_delta,
        "rds_failure_delta_calibration": calibration,
        "quality_filter": {
            "rds_reference_max_for_training": RDS_REFERENCE_MAX_FOR_TRAINING,
            "rds_min_observable_span": RDS_MIN_OBSERVABLE_SPAN,
            "excluded_tests": excluded_tests,
            "paper_split_counts": split_summary.groupby(SPLIT_COLUMN)[TEST_ID_COLUMN].nunique().to_dict(),
            "paper_split_tests": {
                split: [int(x) for x in split_summary.loc[split_summary[SPLIT_COLUMN] == split, TEST_ID_COLUMN].tolist()]
                for split in ("train", "val", "test")
            },
            "post_eol_truncation": {
                "hi_threshold": POST_EOL_HI_THRESHOLD,
                "keep_windows_after_threshold": POST_EOL_KEEP_WINDOWS,
                "truncated_tests": truncated_tests,
            },
        },
        "note": "GRU predicts a bounded residual around a corrected Rds(on) electrical-degradation HI prior. Delta_Rds_EOL is calibrated only from the paper train split, not hand-picked. Tests without an observable Rds(on) degradation trajectory are excluded from the Rds-based paper model and reported in quality_filter.",
        "metrics": metrics.to_dict(orient="records"),
        "rul_metrics": rul_metrics.to_dict(orient="records"),
    }
    (RESULTS_DIR / f"nasa_mosfet_{OUTPUT_TAG}_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("PINN-HI GRU training finished.")
    print(f"Calibrated Delta_Rds_EOL: {rds_failure_delta:.6f}")
    print("Paper Test-level split counts:")
    print(split_summary.groupby(SPLIT_COLUMN)[TEST_ID_COLUMN].nunique().to_string())
    if excluded_tests:
        print("Excluded invalid Rds(on) tests:")
        for item in excluded_tests:
            print(item)
    print(metrics[metrics["scope"].isin(["train", "val", "test", "Test_14", "Test_37"])].to_string(index=False))


if __name__ == "__main__":
    main()

