# Homogeneous Ensembles for Configurable Software Performance Prediction: A Multi-Metric Empirical Study

This repository contains the source code, the data used, and the raw
results for each experiment from the corresponding paper.

## Documents

### /data

This folder contains the data used for all experiments. There is one CSV
file for each subject system (eight in total). Each column represents a
configuration option, and the last column is the performance value to be
predicted.

### /results_paper
This folder does not contain the raw results directly. Due to their large size, separate download links are provided for each dataset in `results_paper/results_paper.txt`. Download and extract them here to reproduce the paper's original results. The folder/file structure is explained below.

Any new execution of `main.py` writes to `/results_new` instead (see `main.py`), so `/results_paper` is never overwritten. `main.py` creates `/results_new` automatically on first run.

The format of the results is as follows:

One folder per dataset, and within each dataset folder, one folder per
sample size (five per dataset).

In each sample size subfolder, there are eight files named
`{Model-Name}_metrics_results.json`. These files represent the results
of the single variants of the corresponding model type (eight model
types).

Each file has the following format:

A JSON file containing one list with ten dict entries. Each dict
represents one variant of the corresponding model with different
hyperparameters (visible in the `"Params"` entry). Each dict contains
the original `"Rank"` entry from the previous hyperparameter tuning, the
new `"Borda_Rank"` entry based on our evaluation method, and the mean
values across 30 runs for each evaluation metric (`Mean_MAE`,
`Mean_MRE`, `Mean_MBRE`, `Mean_MIBRE`). Additionally, `"Metrics"` and
`"Runs"` contain the evaluation metric values of each run and the raw
predictions of each run for each element in the corresponding test sets.
Each entry includes its SA comparison to assess whether the selected
learners perform meaningfully better than chance.

Our experiments include 27 different combinations of ensemble variants
for each learning technique: 3 combination rules (Mean, IRWM, and
NN) and 9 different numbers of single learners (from 2 to 10).
Similar to the single variants, we have:

-   `{Model-Name}_top{k}_predictions.json` for the Mean combination rule
-   `{Model-Name}_top{k}_irwm_predictions.json` for the IRWM combination
    rule
-   `{Model-Name}_top{k}_nn_predictions.json` for the NN combination
    rule

with `k` representing the number of single learners.

These files contain a single dict with the mean values across 30 runs
for each evaluation metric (`Mean_MAE`, `Mean_MRE`, `Mean_MBRE`,
`Mean_MIBRE`). `"Metrics"` and `"Runs"` contain the evaluation metric
values of each run and the raw predictions of each run for each element
in the corresponding test sets.

### /DEEPPERF and /HINNPERF

These directories contain the implementations of both state-of-the-art
approaches, using their publicly available resources:
https://github.com/DeepPerf/DeepPerf and https://drive.google.com/drive/folders/1qxYzd5Om0HE1rK0syYQsTPhTQEBjghLh

### ML Techniques

Each ML technique has its own class based on the `Base` class in
`base.py`. It contains the general logic described in the paper for
constructing the single variants.

### main.py

This file contains the general logic described in the paper for
evaluating the single variants from the previous step and constructing
and evaluating the ensemble variants.

### /analysis_new

Analysis code (statistics, tables, figures) for the results of `main.py`.
It only reads result JSON files already produced by `main.py` (it never
re-runs any model or experiment). See "How to Run the Analysis" below.

### val_predictions.py and ensembles_val.py

We also rank the single variants and select the ensembles on validation
data. `val_predictions.py` computes validation predictions for the top-10
variants of each model type, with the same hyperparameters: 5-fold
out-of-fold predictions on the training set of each run for LR, SVR, RT,
RF, KNN and KRR, and the 2/3-1/3 split of the training set for DeepPerf
and HINNPerf. The results are written to `/results_val`.

`ensembles_val.py` ranks the variants by Borda over the validation MAE,
MRE, MBRE and MIBRE, builds top-k ensembles
(k = 2..10; Mean, IRWM and NN) and random-k ensembles (k = 2..5; Mean
and NN), and evaluates them on the test sets. The results are written to
`/results_ens_val`.

### /analysis_val

Analysis of these results: selection of the best single variant and of
the best ensemble on validation (test values are reported), Scott-Knott
and Borda rankings, member diversity (error correlation, relative
ambiguity and accuracy spread) and the tables and figures of the paper in
`output_artifacts/paper_v2/`. `output_artifacts/cache/` holds the results
in compact tables, so the analysis can be rerun without the raw results.

### /dataset_chars

Characteristics of the subject systems (options, sampling density,
distribution of the performance values, variance explained per option).

## Prerequisites and Installation

Use `requirements.txt` and follow any runtime messages. The experiments were executed with Python 3.9.x.

## How to Run

Comment out the datasets and models you want to exclude. Otherwise, it
will run all experiments by default, as described in the paper.

To reproduce the experiments from the paper:

    python main.py --model baseline
    python main.py

Note: Results from the next execution will be saved in `/results_new`.

## How to Run the Analysis

The code under `/analysis_new` only reads existing result JSON files (from
`/results_paper` or `/results_new`) and never re-runs any experiment.

1. (If needed) Install its dependencies (on top of `requirements.txt`):

       pip install -r analysis_new/requirements_analysis.txt

2. Set `RESULTS_DIR` in `analysis_new/config.py`:

       RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results_new"))

   Change `results_new` to `results_paper` (or any other results folder) to
   analyze a different set of results.

3. Run the pipeline from `analysis_new/`:

       cd analysis_new
       python run_all.py --all --sel-agg mean

   `--sel-agg mean` is the aggregation used for the tables/figures in the
   paper; it writes to `output_artifacts/latex_mean/` and
   `output_artifacts/figures_mean/` (the default `--sel-agg median` writes
   to `latex/`/`figures/`).

   Or run individual stages (each caches its output, so later stages reuse
   it): `--load`, `--aggregate`, `--tables`, `--figures`. Add `--force` to
   ignore the cache and recompute. Outputs are written to
   `analysis_new/output_artifacts/` (`cache/`, `latex/`, `figures/`).

## Validation-Based Ranking and Selection

### Results

Download links for the validation predictions and the validation-based
ensembles are provided in `results_paper/results_val.txt`, one archive per
dataset. Extract all archives directly in the repository root (not into a
separate folder per archive); this creates:

    results_val/<dataset>/<size>/<Model>/run_00.npz ... run_29.npz
    results_ens_val/<dataset>/<size>/<Model>_val_ranking.json
    results_ens_val/<dataset>/<size>/<Model>_top{k}_predictions.json        (k = 2..10, Mean)
    results_ens_val/<dataset>/<size>/<Model>_top{k}_irwm_predictions.json   (IRWM)
    results_ens_val/<dataset>/<size>/<Model>_top{k}_nn_predictions.json     (NN)
    results_ens_val/<dataset>/<size>/<Model>_random{k}_predictions.json     (k = 2..5, Mean)
    results_ens_val/<dataset>/<size>/<Model>_random{k}_nn_predictions.json  (NN)

Each `run_XX.npz` holds, for one run, the validation configurations
(`config_idx`), their true values (`y_true`) and the predictions of the
ten variants (`y_pred`, in grid-search rank order). `<Model>_val_ranking.json`
lists the variants with their validation and test metrics and their
validation Borda rank. The ensemble files have the same format as the
ensemble files in `/results_paper`, plus the validation metrics
(`Val_Metrics`), the selector, `k` and the combination rule, and, per run,
the validation predictions and the members.

The scripts read the single variants from `/results_new` (the output folder
of `main.py`). To use the downloaded results instead, set `RESULTS_DIR` in
`val_predictions.py` and `RESULTS_OLD_DIR` in `analysis_val/config.py` to
`results_paper`.

### How to Run

From the repository root:

    python val_predictions.py
    python ensembles_val.py

Both accept `--dataset`, `--model`, `--size` and `--processes`. Then:

    cd analysis_val
    python run_all.py --all
    python paper_v2.py
    python paper_v2_extra.py
    cd ..
    python dataset_chars/dataset_chars.py

`run_all.py --load` reads the raw results into `output_artifacts/cache/`;
`--analyze` only needs the cache, which is included in the repository, so
`python run_all.py --analyze` followed by the two `paper_v2` scripts
reproduces the tables and figures without downloading the raw results.
`paper_v2.py` writes to `output_artifacts/paper_v2/` and stops if that
folder exists, so remove it first (`rm -r output_artifacts/paper_v2`).
The analysis needs the packages in `analysis_new/requirements_analysis.txt`.
If the GPU build of TensorFlow causes problems, install
`tensorflow-cpu==2.10.1` instead.

### Tables and Figures of the Paper

Paths relative to `analysis_val/output_artifacts/paper_v2/`:

    Table 6  -> latex/table6_singles_b.tex          Fig. 6   -> figures/fig06_d_forest_median.pdf
    Table 7  -> latex/table_sk_ranks_fig7_fig8.tex  Fig. 7   -> figures/fig07_singles_mre_all.pdf
    Table 8  -> csv/table7_single_vs_ensemble.csv   Fig. 8   -> figures/fig08_singles_mre_s1.pdf
    Table 9  -> latex/table_ablation.tex            Fig. 9   -> figures/fig09_singles_rank_s1.pdf
    Table 10 -> latex/table_diversity_summary.tex   Fig. 10  -> figures/fig10_gap_close.pdf
    Table 11 -> latex/table8_ensembles_b.tex        Fig. 11  -> figures/fig11_k_sk_rank_curve.pdf
                                                    Fig. 12  -> figures/fig12_rule_heatmap_all_median.pdf
                                                    Fig. 13  -> figures/fig13_rule_heatmap_s1_median.pdf
                                                    Fig. A.1 -> figures/fig11_k_sk_rank_curve_per_dataset.pdf
