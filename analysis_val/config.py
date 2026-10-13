# Paths and shared constants for the validation-based (main) and test-based (oracle) analyses.

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.normpath(os.path.join(_HERE, ".."))

RESULTS_OLD_DIR = os.path.join(ROOT, "results_new")
RESULTS_VAL_DIR = os.path.join(ROOT, "results_val")
RESULTS_ENS_DIR = os.path.join(ROOT, "results_ens_val")

RAW_CACHE_DIR = os.path.join(_HERE, "output_artifacts", "cache")

MODEL_TYPES    = ["LR", "SVR", "RT", "RF", "KNN", "KRR", "DeepPerf", "HINNPerf"]
METRICS        = ["MRE", "MAE", "MBRE", "MIBRE"]
MEDIAN_METRICS = ["MdMRE", "MdAE", "MdBRE", "MdIBRE"]
ALL_METRICS    = METRICS + MEDIAN_METRICS
BORDA_METRICS  = ["MAE", "MRE", "MBRE", "MIBRE"]
RULES          = ["MEAN", "IRWM", "NN"]
K_TOP          = list(range(2, 11))
K_RANDOM       = list(range(2, 6))
NUM_RUNS       = 30

RULE_FILE_SUFFIX = {"MEAN": "", "IRWM": "_irwm", "NN": "_nn"}
SELECTOR_RULES   = {"top": ["MEAN", "IRWM", "NN"], "random": ["MEAN", "NN"]}
SELECTOR_K       = {"top": K_TOP, "random": K_RANDOM}

# Candidate set for the best ensemble of a scenario; random-k is used only in the ablation.
BEST_ENS_SELECTORS = ["top"]

# Selection of the best single / best ensemble: "val" (validation Borda) or "test" (lowest mean test MRE).
SELECTIONS = ["val", "test"]

SK_KWARGS = {"a12_threshold": 0.60, "conf": 0.01, "seed": 42}

N_JOBS = 7

KEYS = ["dataset", "sample_size", "model"]

MODEL_COLORS = {
    "LR":       "#1f77b4",
    "SVR":      "#ff7f0e",
    "RT":       "#2ca02c",
    "RF":       "#d62728",
    "KNN":      "#9467bd",
    "KRR":      "#8c564b",
    "DeepPerf": "#e377c2",
    "HINNPerf": "#7f7f7f",
}
RULE_COLORS  = {"MEAN": "#2166ac", "IRWM": "#4dac26", "NN": "#d01c8b"}
RULE_MARKERS = {"MEAN": "o", "IRWM": "s", "NN": "^"}
BASE_MARKERS = {
    "LR": "o", "SVR": "s", "RT": "^", "RF": "D",
    "KNN": "v", "KRR": "P", "DeepPerf": "X", "HINNPerf": "*",
}
