import argparse
import json
import multiprocessing
import os
import time
import zlib

import numpy as np
import tensorflow as tf

from main import MODEL_REGISTRY, DATASET_REGISTRY
from perf_metrics import single_run_metrics, ensemble_run_metrics, aggregate_runs
from val_predictions import RESULTS_DIR, INIT_SEED, NUM_RUNS, run_path

BORDA_METRICS = ["MAE", "MRE", "MBRE", "MIBRE"]
TOP_K = [2, 3, 4, 5, 6, 7, 8, 9, 10]
RANDOM_K = [2, 3, 4, 5]
SUFFIX = {"mean": "", "nn": "_nn", "irwm": "_irwm"}

cpu_count = os.cpu_count()


# Single-variant results of results_new and the validation predictions, variants in grid-search rank order
def load_scenario(dataset, s, model_name, val_dir):
    path = os.path.join(RESULTS_DIR, dataset, str(s), f"{model_name}_metrics_results.json")
    with open(path, "r") as f:
        data = json.load(f)
    data.sort(key=lambda x: x["Rank"])

    ranks = [entry["Rank"] for entry in data]
    params = [json.dumps(entry["Params"], sort_keys=True) for entry in data]
    val_runs = []
    for run in range(NUM_RUNS):
        val = np.load(run_path(os.path.join(val_dir, dataset, str(s), model_name), run))
        if list(val["ranks"]) != ranks or list(val["params"]) != params:
            raise ValueError(f"Variant mismatch: {model_name} {dataset} {s} run {run}")
        val_runs.append({"config_idx": val["config_idx"], "y_true": val["y_true"], "y_pred": val["y_pred"]})
    return data, val_runs


# Borda ranking of the variants on validation metrics averaged over the runs
def val_borda(data, val_runs):
    entries = []
    for i, entry in enumerate(data):
        per_run = [single_run_metrics(val["y_true"], val["y_pred"][i]) for val in val_runs]
        per_run_test = [single_run_metrics(run["y_test"], run["y_pred"]) for run in entry["Runs"]]
        ranking = {
            "Rank": entry["Rank"],
            "Params": entry["Params"],
            "Test_Borda_Rank": entry.get("Borda_Rank", None),
            "Val_Metrics": {k: [m[k] for m in per_run] for k in per_run[0]},
            "Test_Metrics": {k: [m[k] for m in per_run_test] for k in per_run_test[0]},
        }
        for k in per_run[0]:
            ranking[f"Val_Mean_{k}"] = float(np.mean(ranking["Val_Metrics"][k]))
            ranking[f"Test_Mean_{k}"] = float(np.mean(ranking["Test_Metrics"][k]))
        entries.append(ranking)

    entries.sort(key=lambda x: x["Val_Mean_MAE"])

    for metric in BORDA_METRICS:
        sorted_entries = sorted(entries, key=lambda x: x[f"Val_Mean_{metric}"])
        for rank, entry in enumerate(sorted_entries, start=1):
            entry[f"Val_Rank_{metric}"] = rank

    for entry in entries:
        entry["Val_Borda_Score"] = sum(entry[f"Val_Rank_{metric}"] for metric in BORDA_METRICS)

    sorted_by_borda = sorted(entries, key=lambda x: (x["Val_Borda_Score"], x["Val_Mean_MAE"]))
    for rank, entry in enumerate(sorted_by_borda, start=1):
        entry["Val_Borda_Rank"] = rank
    return sorted_by_borda


def combine_mean(y_preds):
    return np.mean(y_preds, axis=0)


def combine_irwm(y_preds, borda_ranks):
    weights = 1.0 / np.array(borda_ranks, dtype=float)
    weights /= weights.sum()
    return np.average(y_preds, axis=0, weights=weights)


# NN combination rule of main.py, also applied to the validation predictions
def combine_nn(y_train_preds, y_train, y_preds, y_val_preds, run_idx,
               epochs=600, activation='linear', loss='mae', early_stopping=True, patience=20):
    n = y_train_preds.shape[1]
    tf.keras.backend.clear_session()
    tf.random.set_seed(25 + run_idx)

    mean = np.mean(y_train_preds, axis=0)
    std = np.std(y_train_preds, axis=0) + 1e-8

    y_train_preds_norm = (y_train_preds - mean) / std
    y_preds_norm = (y_preds - mean) / std
    y_val_preds_norm = (y_val_preds - mean) / std

    mean_y = np.mean(y_train)
    std_y = np.std(y_train) + 1e-8
    y_train_norm = (y_train - mean_y) / std_y

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n,)),
        tf.keras.layers.Dense(1, activation=activation)
    ])
    model.compile(optimizer='adam', loss=loss)

    callbacks = []
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=patience, restore_best_weights=True
        ))

    model.fit(y_train_preds_norm, y_train_norm, epochs=epochs, verbose=0, callbacks=callbacks, shuffle=False)

    y_pred_nn = model.predict(y_preds_norm, verbose=0).flatten() * std_y + mean_y
    y_val_nn = model.predict(y_val_preds_norm, verbose=0).flatten() * std_y + mean_y
    return y_pred_nn, y_val_nn


# Members of one run as positions in the rank-ordered variant list, best validation Borda rank first
def select_members(selector, k, run, borda_order, seed_key):
    if selector == "top":
        return borda_order[:k]
    rng = np.random.default_rng([INIT_SEED + run, k, seed_key])
    drawn = set(rng.choice(len(borda_order), size=k, replace=False).tolist())
    return [i for i in borda_order if i in drawn]


def build_ensemble(data, val_runs, ranking, selector, k, combiner, seed_key):
    position = {entry["Rank"]: i for i, entry in enumerate(data)}
    borda_order = [position[entry["Rank"]] for entry in ranking]
    borda_rank = {position[entry["Rank"]]: entry["Val_Borda_Rank"] for entry in ranking}

    test_metrics, val_metrics, runs = [], [], []
    for run in range(NUM_RUNS):
        members = select_members(selector, k, run, borda_order, seed_key)
        val = val_runs[run]

        y_test = np.array(data[0]["Runs"][run]["y_test"])
        y_preds = [np.array(data[i]["Runs"][run]["y_pred"]) for i in members]
        y_val_preds = [val["y_pred"][i] for i in members]

        if combiner == "mean":
            y_pred = combine_mean(y_preds)
            y_val_pred = combine_mean(y_val_preds)
        elif combiner == "irwm":
            member_ranks = [borda_rank[i] for i in members]
            y_pred = combine_irwm(y_preds, member_ranks)
            y_val_pred = combine_irwm(y_val_preds, member_ranks)
        else:
            y_train_preds = np.stack([np.array(data[i]["Runs"][run]["y_train_pred"]) for i in members], axis=1)
            y_train = np.array(data[0]["Runs"][run]["y_train"])
            y_pred, y_val_pred = combine_nn(y_train_preds, y_train, np.stack(y_preds, axis=1),
                                            np.stack(y_val_preds, axis=1), run)

        test_metrics.append(ensemble_run_metrics(y_test, y_pred))
        val_metrics.append(ensemble_run_metrics(val["y_true"], y_val_pred))
        runs.append({
            "y_test": y_test.tolist(),
            "y_pred": np.asarray(y_pred).tolist(),
            "y_val": val["y_true"].tolist(),
            "y_val_pred": np.asarray(y_val_pred).tolist(),
            "val_config_idx": val["config_idx"].tolist(),
            "Members": [data[i]["Rank"] for i in members]
        })

    val_agg = aggregate_runs(val_metrics)
    return {
        **aggregate_runs(test_metrics),
        **{f"Val_{key}": value for key, value in val_agg.items()},
        "Selector": selector,
        "K": k,
        "Combiner": combiner,
        "Variants": [{
            "Rank": entry["Rank"],
            "Params": entry["Params"],
            "Val_Borda_Rank": entry["Val_Borda_Rank"],
            "Val_Borda_Score": entry["Val_Borda_Score"]
        } for entry in ranking],
        "Runs": runs
    }


def log_time(out_root, message):
    with open(os.path.join(out_root, "times_ens.txt"), "a") as f:
        f.write(message + "\n")


def run_ensemble_task(model_name, dataset, s, val_dir, out_root):
    s = int(s)
    results_dir = os.path.join(out_root, dataset, str(s))
    os.makedirs(results_dir, exist_ok=True)

    ensembles = []
    for k in TOP_K:
        for combiner in ["mean", "nn", "irwm"]:
            ensembles.append(("top", k, combiner, f"{model_name}_top{k}{SUFFIX[combiner]}_predictions.json"))
    for k in RANDOM_K:
        for combiner in ["mean", "nn"]:
            ensembles.append(("random", k, combiner, f"{model_name}_random{k}{SUFFIX[combiner]}_predictions.json"))

    ranking_path = os.path.join(results_dir, f"{model_name}_val_ranking.json")
    pending = [e for e in ensembles if not os.path.exists(os.path.join(results_dir, e[3]))]
    if not pending and os.path.exists(ranking_path):
        print(f"Skipping {model_name} on {dataset} sample {s}, all files exist", flush=True)
        return f"Skipped {model_name} on {dataset} sample {s}"

    print(f"PID {os.getpid()} is running ensembles for {model_name} on {dataset} sample {s}", flush=True)
    t_task = time.time()
    data, val_runs = load_scenario(dataset, s, model_name, val_dir)
    ranking = val_borda(data, val_runs)
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2)

    seed_key = zlib.crc32(f"{model_name}|{dataset}|{s}".encode())
    for selector, k, combiner, file_name in pending:
        t0 = time.time()
        result = build_ensemble(data, val_runs, ranking, selector, k, combiner, seed_key)
        tmp_path = os.path.join(results_dir, file_name + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, os.path.join(results_dir, file_name))
        elapsed = time.time() - t0
        log_time(out_root, f"Completed {model_name} {selector}-{k} {combiner} on {dataset} sample {s} in {elapsed:.2f} seconds")
        print(f"{model_name} {dataset} {s} {selector}-{k} {combiner} done in {elapsed:.2f} seconds", flush=True)

    log_time(out_root, f"Completed ensembles {model_name} on {dataset} sample {s} in {time.time() - t_task:.2f} seconds")
    return f"Completed {model_name} on {dataset} sample {s}"


def main():
    parser = argparse.ArgumentParser(description='Ensembles ranked on validation predictions')
    parser.add_argument('--dataset', type=str, default='all', help='Dataset name')
    parser.add_argument('--model', type=str, default="all", help='Model name')
    parser.add_argument('--size', type=int, default=None, help='Training size (default: all sizes of the dataset)')
    parser.add_argument('--val_dir', type=str, default="results_val", help='Folder with the validation predictions')
    parser.add_argument('--out', type=str, default="results_ens_val", help='Output folder')
    parser.add_argument('--processes', type=int, default=max(1, cpu_count - 3), help='Number of processes')

    args = parser.parse_args()

    if args.dataset == "all":
        datasets = DATASET_REGISTRY
    else:
        datasets = {args.dataset: DATASET_REGISTRY[args.dataset]}

    if args.model == "all":
        models = [m for m in MODEL_REGISTRY.keys() if m != "baseline"]
    else:
        models = [args.model]

    os.makedirs(args.out, exist_ok=True)

    task_args = []
    for model_name in models:
        for dataset, sample_sizes in datasets.items():
            for s in sample_sizes:
                if args.size is None or int(s) == args.size:
                    task_args.append((model_name, dataset, int(s), args.val_dir, args.out))

    t0 = time.time()
    if args.processes == 1 or len(task_args) == 1:
        results = [run_ensemble_task(*task) for task in task_args]
    else:
        with multiprocessing.Pool(processes=min(args.processes, len(task_args))) as pool:
            results = pool.starmap(run_ensemble_task, task_args)
    for res in results:
        print(res)
    print(f"All ensemble tasks completed in {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    main()
