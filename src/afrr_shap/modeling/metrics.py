"""Metrics and target grouping helpers."""

import numpy as np


def smape(y_true, y_pred, eps: float = 1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom, axis=0)


def target_groups(target_names, y_test):
    """Group targets without averaging prices and volumes together."""
    groups = {}
    for i, target in enumerate(target_names):
        if "imbalance_volumes" in target:
            groups[target] = "volume [MWh]"
        elif "Short" in target:
            long_target = target.replace("Short", "Long")
            j = target_names.index(long_target)
            same = np.isclose(y_test[:, i], y_test[:, j]).mean() >= 0.99
            groups[target] = (
                "price Short = Long (single pricing)"
                if same
                else "price Short [EUR/MWh]"
            )
        else:
            groups[target] = "price Long [EUR/MWh]"
    return groups
