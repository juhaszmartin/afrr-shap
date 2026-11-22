"""Metrics, naive baseline and target grouping helpers."""

import re

import numpy as np

HORIZON_SUFFIX = re.compile(r"_t\+\d+(min|h)$")


def smape(y_true, y_pred, eps: float = 1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom, axis=0)


def error_metrics(y_true, y_pred):
    """Per-target MAE, RMSE, R2 and SMAPE for (samples, targets) arrays, in original units."""
    err = y_pred - y_true
    ss_res = (err**2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return {
        "MAE": np.abs(err).mean(axis=0),
        "RMSE": np.sqrt((err**2).mean(axis=0)),
        "R2": 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan),
        "SMAPE": smape(y_true, y_pred),
    }


def base_series(target):
    """'imbalance_price_long_t+15min' -> 'imbalance_price_long'."""
    return HORIZON_SUFFIX.sub("", target)


def horizon(target):
    """'imbalance_price_long_t+15min' -> 't+15min'."""
    return target[len(base_series(target)) + 1 :]


def naive_columns(target_names, feature_names):
    """Index of the feature holding each target's value at t (the naive forecast for every horizon)."""
    idx = []
    for t in target_names:
        base = base_series(t)
        now = [f"{base}_t+0min", f"{base}_t+0h"]
        idx.append(next(feature_names.index(c) for c in now if c in feature_names))
    return idx


GROUP_LONG = "long price [EUR/MWh]"
GROUP_SHORT = "short price [EUR/MWh]"
GROUP_SINGLE = "short price = long price (single pricing)"
GROUP_VOLUME = "volume [MWh]"


def target_groups(target_names, y_true):
    """Group targets without averaging prices and volumes together. A short-price target whose values
    equal the long price in at least 99% of rows is marked as single pricing and left out of summaries."""
    groups = {}
    for i, target in enumerate(target_names):
        base = base_series(target)
        if base == "imbalance_volume":
            groups[target] = GROUP_VOLUME
        elif base == "imbalance_price_short":
            j = target_names.index(target.replace("_short", "_long"))
            same = np.isclose(y_true[:, i], y_true[:, j]).mean() >= 0.99
            groups[target] = GROUP_SINGLE if same else GROUP_SHORT
        elif base == "imbalance_price_long":
            groups[target] = GROUP_LONG
        else:
            raise ValueError(f"Unknown target {target}")
    return groups
