# Metrics of single variants and ensembles, per run and aggregated over the runs.

import numpy as np

EPS_SINGLE = 0.0
EPS_ENSEMBLE = 1e-8

MEAN_METRICS = ["MRE", "MAE", "MBRE", "MIBRE"]
MEDIAN_METRICS = ["MdMRE", "MdAE", "MdBRE", "MdIBRE"]
RUN_METRICS = MEAN_METRICS + MEDIAN_METRICS


def config_errors(y_true, y_pred, eps):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    abs_err = np.abs(y_true - y_pred)
    return {
        "AE": abs_err,
        "RE": abs_err / (np.abs(y_true) + eps),
        "BRE": abs_err / (np.minimum(y_true, y_pred) + eps),
        "IBRE": abs_err / (np.maximum(y_true, y_pred) + eps),
    }


def run_metrics(y_true, y_pred, eps):
    e = config_errors(y_true, y_pred, eps)
    return {
        "MRE": float(np.mean(e["RE"]) * 100),
        "MAE": float(np.mean(e["AE"])),
        "MBRE": float(np.mean(e["BRE"])),
        "MIBRE": float(np.mean(e["IBRE"])),
        "MdMRE": float(np.median(e["RE"]) * 100),
        "MdAE": float(np.median(e["AE"])),
        "MdBRE": float(np.median(e["BRE"])),
        "MdIBRE": float(np.median(e["IBRE"])),
    }


def single_run_metrics(y_true, y_pred):
    return run_metrics(y_true, y_pred, EPS_SINGLE)


def ensemble_run_metrics(y_true, y_pred):
    return run_metrics(y_true, y_pred, EPS_ENSEMBLE)


def aggregate_runs(per_run):
    metrics = {k: [m[k] for m in per_run] for k in RUN_METRICS}
    mean_metrics = {f"Mean_{k}": float(np.mean(metrics[k])) for k in metrics}
    median_metrics = {f"Median_{k}": float(np.median(metrics[k])) for k in metrics}
    return {
        "Metrics": metrics,
        **mean_metrics,
        **median_metrics,
    }
