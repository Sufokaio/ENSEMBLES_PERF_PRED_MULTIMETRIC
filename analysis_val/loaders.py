# Load results_new/ (singles, old ensembles, baseline) and results_ens_val/ (rankings, new ensembles) into flat tables.

import json
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd

import config as cfg

sys.path.insert(0, cfg.ROOT)
from perf_metrics import single_run_metrics


def list_scenarios():
    scenarios = []
    for dataset in sorted(os.listdir(cfg.RESULTS_OLD_DIR)):
        ds_path = os.path.join(cfg.RESULTS_OLD_DIR, dataset)
        if not os.path.isdir(ds_path):
            continue
        for size in os.listdir(ds_path):
            if size.isdigit():
                scenarios.append((dataset, int(size)))
    return sorted(scenarios)


def load_baseline():
    rows = []
    for dataset, size in list_scenarios():
        path = os.path.join(cfg.RESULTS_OLD_DIR, dataset, str(size), "baseline_metrics_results.json")
        with open(path) as f:
            d = json.load(f)
        rows.append({"dataset": dataset, "sample_size": size,
                     "MAEp0": float(d["MAEp0"]), "Sp0": float(d["Sp0"]),
                     "Q5p0": float(d["Q5p0"]), "SA_5": float(d["SA_5"])})
    return pd.DataFrame(rows)


def _pairwise_mean_corr(m):
    if m.shape[0] < 2:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(m)
    vals = c[np.triu_indices(m.shape[0], 1)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


# Diversity of one member set on one run, from the members' test predictions
def diversity_measures(preds, y):
    errors = preds - y
    mean_pred = preds.mean(axis=0)
    ambiguity = float(((preds - mean_pred) ** 2).mean())
    member_mse = float((errors ** 2).mean())
    return {
        "err_corr": _pairwise_mean_corr(errors),
        "pred_corr": _pairwise_mean_corr(preds),
        "ambiguity": ambiguity,
        "member_mse": member_mse,
        "ens_mse": float(((mean_pred - y) ** 2).mean()),
        "rel_ambiguity": ambiguity / member_mse if member_mse > 0 else np.nan,
    }


def _ens_file(model, selector, k, rule):
    return f"{model}_{selector}{k}{cfg.RULE_FILE_SUFFIX[rule]}_predictions.json"


def load_task(task):
    dataset, size, model = task
    old_dir = os.path.join(cfg.RESULTS_OLD_DIR, dataset, str(size))
    new_dir = os.path.join(cfg.RESULTS_ENS_DIR, dataset, str(size))
    key = {"dataset": dataset, "sample_size": size, "model": model}

    with open(os.path.join(old_dir, f"{model}_metrics_results.json")) as f:
        singles = json.load(f)
    with open(os.path.join(new_dir, f"{model}_val_ranking.json")) as f:
        ranking = json.load(f)
    rank_info = {e["Rank"]: e for e in ranking}
    val_order = [e["Rank"] for e in sorted(ranking, key=lambda e: e["Val_Borda_Rank"])]

    y_test, preds = {}, {}
    variant_rows = []
    max_stored_diff = 0.0
    for entry in singles:
        rank = entry["Rank"]
        info = rank_info[rank]
        preds[rank] = []
        for run, r in enumerate(entry["Runs"]):
            y = np.asarray(r["y_test"], dtype=float).ravel()
            p = np.asarray(r["y_pred"], dtype=float).ravel()
            y_test[run] = y
            preds[rank].append(p)
            row = dict(key, variant=rank, run=run,
                       val_borda_rank=info["Val_Borda_Rank"],
                       test_borda_rank=entry.get("Borda_Rank"),
                       n_test=len(p), n_negative=int((p < 0).sum()))
            metrics = single_run_metrics(y, p)
            for m in cfg.ALL_METRICS:
                row[f"test_{m}"] = metrics[m]
                row[f"val_{m}"] = info["Val_Metrics"][m][run]
            for m in cfg.METRICS:
                stored = entry["Metrics"][m][run]
                denom = max(abs(stored), 1e-300)
                max_stored_diff = max(max_stored_diff, abs(stored - metrics[m]) / denom)
            variant_rows.append(row)
    del singles

    ens_rows, div_rows = [], []
    random_members = {}
    for selector in ["top", "random"]:
        for k in cfg.SELECTOR_K[selector]:
            for rule in cfg.SELECTOR_RULES[selector]:
                with open(os.path.join(new_dir, _ens_file(model, selector, k, rule))) as f:
                    obj = json.load(f)
                for run in range(cfg.NUM_RUNS):
                    row = dict(key, selector=selector, k=k, rule=rule, run=run)
                    for m in cfg.ALL_METRICS:
                        row[f"test_{m}"] = obj["Metrics"][m][run]
                        row[f"val_{m}"] = obj["Val_Metrics"][m][run]
                    ens_rows.append(row)
                if selector == "random" and rule == "MEAN":
                    random_members[k] = [r["Members"] for r in obj["Runs"]]
                del obj

    for selector in ["top", "random"]:
        for k in cfg.SELECTOR_K[selector]:
            for run in range(cfg.NUM_RUNS):
                members = val_order[:k] if selector == "top" else random_members[k][run]
                p = np.stack([preds[m][run] for m in members])
                div_rows.append(dict(key, selector=selector, k=k, run=run,
                                     **diversity_measures(p, y_test[run])))

    old_rows = []
    for k in cfg.K_TOP:
        for rule in cfg.RULES:
            with open(os.path.join(old_dir, _ens_file(model, "top", k, rule))) as f:
                obj = json.load(f)
            for run in range(cfg.NUM_RUNS):
                row = dict(key, k=k, rule=rule, run=run)
                for m in cfg.METRICS:
                    row[f"test_{m}"] = obj["Metrics"][m][run]
                old_rows.append(row)
            del obj

    return variant_rows, ens_rows, div_rows, old_rows, max_stored_diff


def load_all(n_jobs):
    tasks = [(ds, size, model) for ds, size in list_scenarios() for model in cfg.MODEL_TYPES]
    variants, ensembles, diversity, old = [], [], [], []
    worst = 0.0
    with multiprocessing.Pool(n_jobs) as pool:
        for i, (v, e, d, o, diff) in enumerate(pool.imap_unordered(load_task, tasks, chunksize=1), start=1):
            variants += v
            ensembles += e
            diversity += d
            old += o
            worst = max(worst, diff)
            if i % 40 == 0:
                print(f"  loaded {i}/{len(tasks)} learner-scenarios", flush=True)
    print(f"  max relative difference between stored and recomputed single metrics: {worst:.3e}")
    return (pd.DataFrame(variants), pd.DataFrame(ensembles),
            pd.DataFrame(diversity), pd.DataFrame(old))
