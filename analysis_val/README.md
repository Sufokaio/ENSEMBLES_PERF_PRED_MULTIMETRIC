# analysis_val: validation-based analysis (main results)

Reads `results_new/` (single variants, old test-ranked ensembles, baseline) and
`results_ens_val/` (validation rankings, validation-ranked ensembles). Nothing is
retrained and no input folder is written to. `analysis_test/` runs the same code
with the selection made on test (oracle / upper bound).

## Selection

| | `analysis_val` (main) | `analysis_test` (oracle) |
|---|---|---|
| Best single | variant with `Val_Borda_Rank` 1 | variant with the lowest mean test MRE |
| Best ensemble | top-k (k, rule) with the best validation Borda | top-k (k, rule) with the lowest mean test MRE |
| Best k per rule | k with the best validation Borda, top-k only | k with the lowest mean test MRE, top-k only |

The validation Borda of the ensembles ranks the candidates of one learner and
scenario on the mean over the 30 runs of the validation MAE, MRE, MBRE and MIBRE
(`Val_Metrics`), sums the four ranks and breaks ties by MAE, as `ensembles_val.py`
does for the single variants. The candidates are the 27 top-k ensembles
(k = 2..10 x MEAN, IRWM, NN); random-k ensembles are used only in the ablation
(item 2). All reported values are test metrics.

## Run

    cd analysis_val
    python run_all.py --all            # writes analysis_val/output_artifacts/
    cd ../analysis_test
    python run_all.py --analyze        # writes analysis_test/output_artifacts/

`--load` reads the JSON files once into `analysis_val/output_artifacts/cache/`
(shared by both folders); `--analyze` writes `csv/`, `latex/` and `figures/`;
`--force` ignores the cache. Dependencies: `analysis_new/requirements_analysis.txt`.

## Outputs

| Item | Files (prefix) |
|---|---|
| Tables 6, 7, 8 | `table6_singles`, `table7_*`, `table8_ensembles` |
| Figures 6-13 | `fig06_d_forest`, `fig07_singles_mre_all`, `fig08_singles_mre_s1`, `fig09_singles_rank_s1`, `fig10_gap_close`, `fig11_k_sk_rank_curve`, `fig12_rule_heatmap_all`, `fig13_rule_heatmap_s1` |
| Chosen variants and ensembles | `selected_*` |
| Random-k first on validation (side note) | `side_note_random_k*` |
| 1. Old vs new | `item01_*` |
| 2. Ablation | `item02_*` |
| 3. Diversity | `item03_*` |
| 4. Improvement magnitude | `item04_*` |
| 5. Medians | `item05_*` (median metrics); the `Median` column of Tables 6 and 8 |
| 6. Combination rules | `item06_*` |
| 7. Saturation of k per dataset | `item07_*` |
| 8. S1 stability | `item08_*` |
| 9. Negative predictions | `item09_*` |

## Definitions

- Metrics of the single variants are recomputed from their saved test predictions
  with `perf_metrics.py`; ensemble metrics are read from the result files.
- Mean and Median in Tables 6 and 8: over the 40 scenario means (mean over the 30 runs).
- SK: Scott-Knott on the 30 runs of each competitor in a scenario (A12 threshold 0.60,
  bootstrap confidence 0.01), averaged over the scenarios. Borda: sum of the SK ranks
  over scenarios and the four metrics.
- SA = 1 - MAE / MAEp0 and D = (MAE - MAEp0) / Sp0 with the saved baseline.
- A12 (item 4): probability that a run of the best ensemble has a lower error than a
  run of the best single, per scenario; 0.56 / 0.64 / 0.71 = small / medium / large.
- Diversity (item 3): per run, from the members' test predictions. Error correlation
  and prediction correlation are the mean pairwise Pearson correlations; ambiguity is
  the mean squared deviation of the members from their unweighted mean (Krogh and
  Vedelsby), and the relative ambiguity divides it by the mean squared error of the members.
- Items 1, 6, 7 and 9 (all variants) do not depend on the selection and are identical
  in both folders.
