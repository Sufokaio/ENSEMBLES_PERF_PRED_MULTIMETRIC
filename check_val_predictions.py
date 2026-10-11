import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

import base
import deepperf
from main import MODEL_REGISTRY
from HINNPERF.HINNPerf_data_preproc import DataPreproc
from HINNPERF.HINNPerf_models import MLPHierarchicalModel
from HINNPERF.HINNPerf_model_runner import ModelRunner
from val_predictions import TRADITIONAL, NUM_RUNS, load_variants, run_split, run_path, deepperf_split

SMALLEST = [("apache", 9), ("x264", 16), ("bdbc", 18)]


class RecordingGridSearchCV(GridSearchCV):
    records = []

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y, **kwargs)
        RecordingGridSearchCV.records.append(pd.DataFrame(self.cv_results_))
        return self


def clean_params(params):
    return {k: type(v).__name__ if k in ('model', 'scaler') else v for k, v in params.items()}


def load_val(val_dir, dataset, s, model_name, run):
    return np.load(run_path(os.path.join(val_dir, dataset, str(s), model_name), run))


def has_val(val_dir, dataset, s, model_name, runs):
    return all(os.path.exists(run_path(os.path.join(val_dir, dataset, str(s), model_name), run)) for run in runs)


# Regenerated splits against the y_train / y_test stored in results_new
def check_splits(evaluator, val_dir, dataset, s, model_name, variants, stored):
    for run in range(NUM_RUNS):
        val = load_val(val_dir, dataset, s, model_name, run)
        train_idx, test_idx = run_split(len(evaluator.y), s, run)
        y_train_stored, y_test_stored = stored[run]
        ok = (
            np.array_equal(val["train_idx"], train_idx)
            and np.array_equal(evaluator.y[train_idx], y_train_stored)
            and np.array_equal(evaluator.y[test_idx], y_test_stored)
            and np.array_equal(val["y_true"], evaluator.y[val["config_idx"]])
            and np.array_equal(val["ranks"], [rank for rank, _ in variants])
        )
        if not ok:
            return False
    return True


# Original grid search re-run; per-fold scores against the MAE of the out-of-fold predictions
def check_traditional(evaluator, val_dir, dataset, s, model_name, variants):
    RecordingGridSearchCV.records = []
    top_10 = evaluator.get_top_10()
    records = RecordingGridSearchCV.records

    grid_params = [params for params, _ in top_10]
    saved_params = [params for _, params in variants]
    same_order = grid_params == saved_params
    same_set = sorted(json.dumps(p, sort_keys=True) for p in grid_params) == sorted(json.dumps(p, sort_keys=True) for p in saved_params)

    rows = []
    for run in range(NUM_RUNS):
        val = load_val(val_dir, dataset, s, model_name, run)
        results = records[run]
        keys = [json.dumps(clean_params(p), sort_keys=True) for p in results["params"]]
        for i, (rank, params) in enumerate(variants):
            row = results.iloc[keys.index(json.dumps(params, sort_keys=True))]
            for f in range(5):
                mask = val["fold"] == f
                oof_mae = np.mean(np.abs(val["y_true"][mask] - val["y_pred"][i][mask]))
                grid_mae = -row[f"split{f}_test_score"]
                rows.append({
                    "model": model_name, "dataset": dataset, "size": s, "run": run, "Rank": rank, "fold": f,
                    "grid_mae": grid_mae, "oof_mae": oof_mae, "abs_diff": abs(grid_mae - oof_mae)
                })
    return pd.DataFrame(rows), same_order, same_set


# Validation error of the original search code for the same variant against the new predictions
def check_deepperf(evaluator, val_dir, dataset, s, variants, runs):
    rows = []
    for run in runs:
        val = load_val(val_dir, dataset, s, "DeepPerf", run)
        train_idx, _ = run_split(len(evaluator.y), s, run)
        X_train1, Y_train1, X_train2, Y_train2, max_Y = deepperf_split(evaluator, train_idx)
        for i, (rank, params) in enumerate(variants):
            orig_error, _ = deepperf.nn_l1_val(X_train1, Y_train1, X_train2, Y_train2,
                                               params['num_layer'], params['lambda'], params['lr'])
            new_error = np.mean(np.abs(val["y_pred"][i] - val["y_true"])) / max_Y
            rows.append({
                "model": "DeepPerf", "dataset": dataset, "size": s, "run": run, "Rank": rank,
                "orig_val_error": orig_error, "new_val_error": new_error, "abs_diff": abs(orig_error - new_error)
            })
    return pd.DataFrame(rows)


def check_hinnperf(evaluator, val_dir, dataset, s, variants, runs):
    rows = []
    for run in runs:
        val = load_val(val_dir, dataset, s, "HINNPerf", run)
        train_idx, test_idx = run_split(len(evaluator.y), s, run)
        for i, (rank, params) in enumerate(variants):
            data_gen = DataPreproc(evaluator.whole_data, train_idx, test_idx)
            runner = ModelRunner(data_gen, MLPHierarchicalModel)
            _, orig_error = runner.train(dict(params))
            new_error = np.mean(np.abs(val["y_pred"][i] - val["y_true"]))
            rows.append({
                "model": "HINNPerf", "dataset": dataset, "size": s, "run": run, "Rank": rank,
                "orig_val_error": orig_error, "new_val_error": new_error, "abs_diff": abs(orig_error - new_error)
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Check of the validation predictions')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (default: the 3 smallest scenarios)')
    parser.add_argument('--size', type=int, default=None, help='Training size')
    parser.add_argument('--model', type=str, default="all", help='Model name')
    parser.add_argument('--dl_runs', type=int, default=3, help='Number of runs checked for DeepPerf and HINNPerf')
    parser.add_argument('--val_dir', type=str, default="results_val", help='Folder with the validation predictions')
    parser.add_argument('--out', type=str, default="results_val_check", help='Output folder')

    args = parser.parse_args()

    scenarios = SMALLEST if args.dataset is None else [(args.dataset, args.size)]
    if args.model == "all":
        models = [m for m in MODEL_REGISTRY.keys() if m != "baseline"]
    else:
        models = [args.model]

    os.makedirs(args.out, exist_ok=True)
    base.GridSearchCV = RecordingGridSearchCV

    summary = []
    for dataset, s in scenarios:
        for model_name in models:
            if not has_val(args.val_dir, dataset, s, model_name, range(NUM_RUNS)):
                print(f"Skipping {model_name} on {dataset} sample {s}, validation predictions missing")
                continue

            evaluator = MODEL_REGISTRY[model_name](dataset, split_mode=False, split=s, num_runs=NUM_RUNS)
            variants, stored = load_variants(dataset, s, model_name)
            entry = {
                "model": model_name, "dataset": dataset, "size": s,
                "splits_match": check_splits(evaluator, args.val_dir, dataset, s, model_name, variants, stored)
            }

            if model_name in TRADITIONAL:
                details, same_order, same_set = check_traditional(evaluator, args.val_dir, dataset, s, model_name, variants)
                entry["top10_same_order"] = same_order
                entry["top10_same_set"] = same_set
            elif model_name == "DeepPerf":
                details = check_deepperf(evaluator, args.val_dir, dataset, s, variants, range(args.dl_runs))
            else:
                details = check_hinnperf(evaluator, args.val_dir, dataset, s, variants, range(args.dl_runs))

            details.to_csv(os.path.join(args.out, f"{dataset}_{s}_{model_name}_check.csv"), index=False)
            entry["max_abs_diff"] = details["abs_diff"].max()
            entry["mean_abs_diff"] = details["abs_diff"].mean()
            summary.append(entry)
            print(entry, flush=True)

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out, "check_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
