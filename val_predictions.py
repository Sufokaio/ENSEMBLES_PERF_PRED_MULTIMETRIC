import argparse
import ast
import json
import multiprocessing
import os
import sys
import time
import traceback

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from main import MODEL_REGISTRY, DATASET_REGISTRY
from DEEPPERF.mlp_sparse_model_tf2 import MLPSparseModel
from HINNPERF.HINNPerf_data_preproc import DataPreproc
from HINNPERF.HINNPerf_models import MLPHierarchicalModel
from HINNPERF.HINNPerf_model_runner import ModelRunner

RESULTS_DIR = "results_new"
TRADITIONAL = ["LR", "SVR", "KNN", "RF", "RT", "KRR"]
SCALERS = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler}
INIT_SEED = 25
NUM_RUNS = 30

cpu_count = os.cpu_count()


def run_split(n_total, num_train, run):
    np.random.seed(INIT_SEED + run)
    indices = np.random.permutation(n_total)
    return indices[:num_train], indices[num_train:]


# Top-10 variants sorted by grid-search rank, and the stored y_train / y_test of every run
def load_variants(dataset, s, model_name):
    path = os.path.join(RESULTS_DIR, dataset, str(s), f"{model_name}_metrics_results.json")
    with open(path, "r") as f:
        data = json.load(f)
    variants = sorted([(entry["Rank"], entry["Params"]) for entry in data], key=lambda x: x[0])
    ranks = [rank for rank, _ in variants]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"Duplicate Rank in {path}")
    stored = [(np.array(r["y_train"]), np.array(r["y_test"])) for r in data[0]["Runs"]]
    return variants, stored


def val_cross(num_train):
    return int(np.ceil(num_train * 2 / 3))


def build_pipeline(params, base_models):
    params_copy = params.copy()
    model_name = params_copy.pop('model')
    scaler_name = params_copy.pop('scaler', 'StandardScaler')

    formatted_params = {}
    for k, v in params_copy.items():
        param_name = k.split('__')[1]
        if v == 'None':
            formatted_params[param_name] = None
        elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
            formatted_params[param_name] = ast.literal_eval(v)
        else:
            formatted_params[param_name] = v

    model = clone(base_models[model_name]).set_params(**formatted_params)
    return Pipeline([
        ('scaler', SCALERS[scaler_name]()),
        ('model', model)
    ])


# Out-of-fold predictions with the folds of the original GridSearchCV
def traditional_val(evaluator, variants, train_idx):
    X_train, y_train = evaluator.X[train_idx], evaluator.y[train_idx]
    base_models = {type(m).__name__: m for m in evaluator.param_grid['model']}

    kf = KFold(n_splits=5)
    fold = np.empty(len(train_idx), dtype=int)
    for i, (_, val_part) in enumerate(kf.split(X_train)):
        fold[val_part] = i

    y_pred = []
    for _, params in variants:
        pipeline = build_pipeline(params, base_models)
        y_pred.append(cross_val_predict(pipeline, X_train, y_train, cv=kf))
    return fold, np.array(y_pred)


def deepperf_split(evaluator, train_idx):
    X_train = evaluator.X[train_idx]
    Y_train = evaluator.y[train_idx][:, np.newaxis]

    max_X = np.amax(X_train, axis=0)
    max_X[max_X == 0] = 1
    X_train = np.divide(X_train, max_X)
    max_Y = np.max(Y_train) / 100
    if max_Y == 0:
        max_Y = 1
    Y_train = np.divide(Y_train, max_Y)

    N_cross = val_cross(len(train_idx))
    X_train1 = X_train[0:N_cross, :]
    Y_train1 = Y_train[0:N_cross]
    X_train2 = X_train[N_cross:, :]
    Y_train2 = Y_train[N_cross:]
    return X_train1, Y_train1, X_train2, Y_train2, max_Y


def split_fold(num_train):
    fold = np.full(num_train, -1, dtype=int)
    fold[val_cross(num_train):] = 0
    return fold


# Original internal validation split: train on the first 2/3, predict the last 1/3
def deepperf_val(evaluator, variants, train_idx):
    X_train1, Y_train1, X_train2, _, max_Y = deepperf_split(evaluator, train_idx)

    y_pred = []
    for _, params in variants:
        config = dict()
        config['num_input'] = X_train1.shape[1]
        config['num_layer'] = params['num_layer']
        config['num_neuron'] = params['num_neuron']
        config['lambda'] = params['lambda']
        config['verbose'] = 0

        model = MLPSparseModel(config)
        model.build_train()
        model.train(X_train1, Y_train1, params['lr'])
        y_pred.append((max_Y * model.predict(X_train2)).ravel())
        model.sess.close()
    return split_fold(len(train_idx)), np.array(y_pred)


def hinnperf_val(evaluator, variants, train_idx):
    N_cross = val_cross(len(train_idx))
    fit_index = train_idx[:N_cross]
    val_index = train_idx[N_cross:]

    y_pred = []
    for _, params in variants:
        data_gen = DataPreproc(evaluator.whole_data, fit_index, val_index)
        runner = ModelRunner(data_gen, MLPHierarchicalModel)
        y_pred_val, _, _, _, _ = runner.test(dict(params))
        y_pred.append(np.asarray(y_pred_val).ravel())
    return split_fold(len(train_idx)), np.array(y_pred)


def run_path(out_dir, run):
    return os.path.join(out_dir, f"run_{run:02d}.npz")


def log_time(out_root, message):
    with open(os.path.join(out_root, "times_val.txt"), "a") as f:
        f.write(message + "\n")


def run_val_task(model_name, dataset, s, out_root):
    s = int(s)
    out_dir = os.path.join(out_root, dataset, str(s), model_name)
    os.makedirs(out_dir, exist_ok=True)

    pending = [run for run in range(NUM_RUNS) if not os.path.exists(run_path(out_dir, run))]
    if not pending:
        print(f"Skipping {model_name} on {dataset} sample {s}, all runs exist", flush=True)
        return f"Skipped {model_name} on {dataset} sample {s}"

    print(f"PID {os.getpid()} is running validation for {model_name} on {dataset} sample {s}", flush=True)
    t0 = time.time()
    evaluator = MODEL_REGISTRY[model_name](dataset, split_mode=False, split=s, num_runs=NUM_RUNS)
    variants, stored = load_variants(dataset, s, model_name)

    for run in pending:
        t_run = time.time()
        train_idx, test_idx = run_split(len(evaluator.y), s, run)

        y_train_stored, y_test_stored = stored[run]
        if not np.array_equal(evaluator.y[train_idx], y_train_stored):
            raise ValueError(f"y_train mismatch: {model_name} {dataset} {s} run {run}")
        if not np.array_equal(evaluator.y[test_idx], y_test_stored):
            raise ValueError(f"y_test mismatch: {model_name} {dataset} {s} run {run}")

        if model_name in TRADITIONAL:
            fold, y_pred = traditional_val(evaluator, variants, train_idx)
        elif model_name == "DeepPerf":
            fold, y_pred = deepperf_val(evaluator, variants, train_idx)
        else:
            fold, y_pred = hinnperf_val(evaluator, variants, train_idx)

        config_idx = train_idx[fold >= 0]
        tmp_path = os.path.join(out_dir, f"run_{run:02d}.tmp.npz")
        np.savez_compressed(
            tmp_path,
            run=run,
            seed=INIT_SEED + run,
            train_idx=train_idx,
            fold=fold,
            config_idx=config_idx,
            y_true=evaluator.y[config_idx],
            y_pred=y_pred,
            ranks=np.array([rank for rank, _ in variants]),
            params=np.array([json.dumps(params, sort_keys=True) for _, params in variants]),
        )
        os.replace(tmp_path, run_path(out_dir, run))
        print(f"{model_name} {dataset} {s} run {run + 1}/{NUM_RUNS} done in {time.time() - t_run:.2f} seconds", flush=True)

    elapsed = time.time() - t0
    log_time(out_root, f"Completed validation {model_name} on {dataset} sample {s} ({len(pending)} runs) in {elapsed:.2f} seconds")
    return f"Completed {model_name} on {dataset} sample {s}"


def run_task(task):
    try:
        return run_val_task(*task)
    except Exception:
        traceback.print_exc()
        return f"FAILED {task[0]} on {task[1]} sample {task[2]}"


def main():
    parser = argparse.ArgumentParser(description='Validation predictions of the top-10 variants')
    parser.add_argument('--dataset', type=str, default='all', help='Dataset name')
    parser.add_argument('--model', type=str, default="all", help='Model name')
    parser.add_argument('--size', type=int, default=None, help='Training size (default: all sizes of the dataset)')
    parser.add_argument('--out', type=str, default="results_val", help='Output folder')
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
                    task_args.append((model_name, dataset, int(s), args.out))

    t0 = time.time()
    if args.processes == 1 or len(task_args) == 1:
        results = [run_task(task) for task in task_args]
    else:
        with multiprocessing.Pool(processes=min(args.processes, len(task_args))) as pool:
            results = list(pool.imap_unordered(run_task, task_args, chunksize=1))
    for res in results:
        print(res)
    print(f"All validation tasks finished in {time.time() - t0:.2f} seconds")
    if any(res.startswith("FAILED") for res in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
