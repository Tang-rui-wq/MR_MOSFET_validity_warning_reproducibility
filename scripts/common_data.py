from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT.parent.parent / "MOSFET_Thermal_Overstress_Aging" / "cleaned"
RUN_ALL_SAMPLES = CLEAN_DIR / "mosfet_run_samples_all.csv"
RUN_TRAIN_SAMPLES = CLEAN_DIR / "mosfet_train_samples.csv"
RUN_VAL_SAMPLES = CLEAN_DIR / "mosfet_val_samples.csv"
RUN_TEST_SAMPLES = CLEAN_DIR / "mosfet_test_samples.csv"

WINDOW_ALL_SAMPLES = CLEAN_DIR / "mosfet_window_samples_all.csv"
WINDOW_TRAIN_SAMPLES = CLEAN_DIR / "mosfet_window_train_samples.csv"
WINDOW_VAL_SAMPLES = CLEAN_DIR / "mosfet_window_val_samples.csv"
WINDOW_TEST_SAMPLES = CLEAN_DIR / "mosfet_window_test_samples.csv"

ALL_SAMPLES = WINDOW_ALL_SAMPLES
TRAIN_SAMPLES = WINDOW_TRAIN_SAMPLES
VAL_SAMPLES = WINDOW_VAL_SAMPLES
TEST_SAMPLES = WINDOW_TEST_SAMPLES

FEATURE_COLUMNS: List[str] = [
    "supplyVoltage_mean",
    "packageTemperature_mean",
    "flangeTemperature_mean",
    "P_proxy_mean",
    "Rds_on_cond_median",
    "Rds_on_temp_resid",
    "Tth_proxy_mean",
    "ThermalExposure_end",
    "DegradationSlope_mean",
    "DamageProxy_end",
    "HI_proxy_end",
    "switchingFrequency_mean",
    "dutyCycle_mean",
]

# Default target is now the nonlinear HI label.
LABEL_COLUMN = "HI_label"
HI_COLUMN = "HI_label"
TEST_ID_COLUMN = "test_id"
SAMPLE_ID_COLUMN = "window_id"
RUN_ORDER_COLUMN = "window_id"
SPLIT_COLUMN = "split"
# Fallback for helpers that rebuild the HI label outside the main trainer.
# The value is the paper-train-only Delta_Rds_EOL calibration used by the
# current fixed Test-level split, not a validation/test-derived threshold.
RDS_FAILURE_DELTA = 0.20919872437861223


def label_alias() -> str:
    if LABEL_COLUMN == "RUL_norm":
        return "RUL"
    if LABEL_COLUMN == "HI_label":
        return "HI"
    if LABEL_COLUMN == "Damage_label":
        return "Damage"
    return LABEL_COLUMN


@dataclass
class Normalizer:
    mu: np.ndarray
    sigma: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma


@dataclass
class HIBaseline:
    damage_weight: float = 0.22
    damage_gamma: float = 1.0
    damage_offset: float = 0.0
    low_percentile: float = 5.0
    high_percentile: float = 95.0

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = np.zeros(len(df), dtype=np.float64)
        for _, sub in df.groupby(TEST_ID_COLUMN, sort=False):
            idx = sub.index.to_numpy()
            hi = _physics_guided_hi_label(
                rds_cond=sub["Rds_on_cond_median"].to_numpy(dtype=np.float64),
                damage_proxy=sub["DamageProxy_end"].to_numpy(dtype=np.float64),
                package_temp=sub["packageTemperature_mean"].to_numpy(dtype=np.float64),
                damage_weight=self.damage_weight,
                gamma=self.damage_gamma,
            )
            pred[df.index.get_indexer(idx)] = hi
        return np.clip(pred, 0.0, 1.0)


def load_all_samples() -> pd.DataFrame:
    df = pd.read_csv(ALL_SAMPLES)
    df = df.sort_values([TEST_ID_COLUMN, RUN_ORDER_COLUMN]).reset_index(drop=True)
    return df


def load_split(split: str) -> pd.DataFrame:
    df = load_all_samples()
    return df[df[SPLIT_COLUMN] == split].copy().reset_index(drop=True)


def fit_normalizer(train_df: pd.DataFrame, feature_cols: List[str] | None = None) -> Normalizer:
    feature_cols = feature_cols or FEATURE_COLUMNS
    x = train_df[feature_cols].to_numpy(dtype=np.float64)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0] = 1.0
    return Normalizer(mu=mu.astype(np.float64), sigma=sigma.astype(np.float64))


def apply_normalizer(df: pd.DataFrame, normalizer: Normalizer, feature_cols: List[str] | None = None) -> np.ndarray:
    feature_cols = feature_cols or FEATURE_COLUMNS
    x = df[feature_cols].to_numpy(dtype=np.float64)
    return normalizer.transform(x).astype(np.float32)


def build_sequence_groups(df: pd.DataFrame, feature_cols: List[str] | None = None, label_col: str = LABEL_COLUMN):
    feature_cols = feature_cols or FEATURE_COLUMNS
    groups = []
    for test_id, sub in df.groupby(TEST_ID_COLUMN, sort=True):
        sub = sub.sort_values(RUN_ORDER_COLUMN)
        groups.append(
            {
                "test_id": int(test_id),
                "sample_ids": sub[SAMPLE_ID_COLUMN].to_numpy(dtype=np.int64),
                "x": sub[feature_cols].to_numpy(dtype=np.float32),
                "y": sub[label_col].to_numpy(dtype=np.float32),
                "split": str(sub[SPLIT_COLUMN].iloc[0]),
            }
        )
    return groups


def all_feature_columns() -> List[str]:
    return FEATURE_COLUMNS.copy()


def fit_hi_baseline(train_df: pd.DataFrame, label_col: str = LABEL_COLUMN) -> HIBaseline:
    if label_col != "HI_label":
        return HIBaseline(damage_weight=0.22, damage_gamma=1.18, damage_offset=0.0)
    return HIBaseline(damage_weight=0.22, damage_gamma=1.18, damage_offset=0.0)


def _within_test_degradation(x: np.ndarray, low_p: float, high_p: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    valid = np.isfinite(x)
    if not valid.any():
        return np.zeros_like(x, dtype=np.float64)
    filled = x.copy()
    med = np.nanmedian(filled[valid])
    filled[~valid] = med
    lo = np.nanpercentile(filled, low_p)
    hi = np.nanpercentile(filled, high_p)
    if not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(filled, dtype=np.float64)
    return np.clip((filled - lo) / (hi - lo), 0.0, 1.0)


def _physics_guided_hi_label(
    rds_cond: np.ndarray,
    damage_proxy: np.ndarray,
    package_temp: np.ndarray,
    damage_weight: float,
    gamma: float,
) -> np.ndarray:
    """Python mirror of nasa_compute_nonlinear_hi_labels.m for model baselines.

    The baseline follows the electrical-degradation HI label:
    temperature-compensated Rds(on) drift normalized by a fixed EOL delta.
    """
    rds = _fill_series(rds_cond)
    pkg = _fill_series(package_temp)
    n = len(rds)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    ref_count = max(8, min(round(0.18 * n), max(8, int(np.floor(0.35 * n)))))
    ref_count = min(ref_count, n)
    ref_idx = np.arange(ref_count)

    x = np.column_stack([np.ones(ref_count), pkg[ref_idx]])
    coef, *_ = np.linalg.lstsq(x, rds[ref_idx], rcond=None)
    rds_ref = coef[0] + coef[1] * pkg
    ref_floor = max(np.finfo(float).eps, 0.10 * np.nanmedian(rds[ref_idx]))
    rds_ref = np.maximum(rds_ref, ref_floor)

    rds_temp_ratio = rds / rds_ref
    rds_temp_resid = np.maximum(rds_temp_ratio - 1.0, 0.0)
    rds_temp_resid = _smooth_series(rds_temp_resid, n)
    resid_ref = np.nanmedian(rds_temp_resid[ref_idx])
    damage_raw = (rds_temp_resid - resid_ref) / RDS_FAILURE_DELTA
    damage_raw = np.clip(_smooth_series(damage_raw, n), 0.0, 1.0)
    damage_raw = np.maximum.accumulate(damage_raw)

    damage = np.power(damage_raw, 1.0)
    damage = np.clip(_smooth_series(damage, n), 0.0, 1.0)
    damage = np.maximum.accumulate(damage)
    return np.clip(1.0 - damage, 0.0, 1.0)


def _fill_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    valid = np.isfinite(x)
    if not valid.any():
        return np.zeros_like(x, dtype=np.float64)
    if valid.all():
        return x
    t = np.arange(len(x), dtype=np.float64)
    x[~valid] = np.interp(t[~valid], t[valid], x[valid])
    return x


def _smooth_series(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 3:
        return x
    span1 = max(5, min(21, 2 * int(np.floor(n / 30)) + 1))
    span2 = max(3, min(11, 2 * int(np.floor(n / 60)) + 1))
    s = pd.Series(x)
    y = s.rolling(window=span2, center=True, min_periods=1).median()
    y = y.rolling(window=span1, center=True, min_periods=1).mean()
    return y.to_numpy(dtype=np.float64)


def _robust_scale(x: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    base = np.nanmedian(x[ref_idx])
    hi = np.nanpercentile(x, 95)
    if not np.isfinite(hi) or hi <= base:
        hi = np.nanmax(x)
    if not np.isfinite(hi) or hi <= base:
        hi = base + 1.0
    return np.clip((x - base) / max(hi - base, np.finfo(float).eps), 0.0, 1.0)


def _rds_weight(
    rds_trend: float,
    rds_span: float,
    dmg_span: float,
    early_rds_peak: float,
    early_rds_median: float,
) -> float:
    if early_rds_peak > 0.60 and early_rds_median < 0.30:
        return 0.05
    if dmg_span < 0.05 and rds_span >= 0.05:
        return 0.78
    if rds_span < 0.05:
        return 0.10
    if rds_trend < -0.10:
        return 0.05
    if rds_trend < 0.15:
        return 0.30
    return 0.78


def _time_trend(x: np.ndarray, t: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(t)
    if np.count_nonzero(good) < 3:
        return 0.0
    x = x[good] - np.mean(x[good])
    t = t[good] - np.mean(t[good])
    den = np.sqrt(np.sum(x ** 2) * np.sum(t ** 2))
    if den <= np.finfo(float).eps:
        return 0.0
    return float(np.sum(x * t) / den)
