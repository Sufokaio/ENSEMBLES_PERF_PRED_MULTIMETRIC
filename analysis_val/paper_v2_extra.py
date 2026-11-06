# Median versions of Figures 12 and 13.

import json
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
import core
import loaders
import paper_v2 as pv
from out import bold, ds_label

pv.SCRIPT = "analysis_val/paper_v2_extra.py"
M = cfg.MODEL_TYPES
SCEN = core.SCEN
DAGGER_CASES = [("RT", "MRE"), ("RT", "MIBRE"), ("RF", "MIBRE"), ("KRR", "MBRE"), ("KRR", "SA"), ("LR", "MBRE")]
N_LIST = 3


def scenario_label(df):
    return df["dataset"].map(ds_label) + " S" + core.size_index(df).astype(str)


def figures12_13_median(w, data):
    best_rule = core.select_best_k_per_rule(data.ensembles, "val")
    sc = best_rule.groupby(cfg.KEYS + ["rule"], as_index=False)["test_MRE"].mean()
    out = {}
    for number, scope in [("12", "all"), ("13", "s1")]:
        src = f"fig{number}_rule_heatmap_{scope}.csv"
        old = pv.val_csv(src)
        s = sc if scope == "all" else core.s1_only(sc)
        n_scen = s.groupby(["rule", "model"]).size()
        assert (n_scen == (40 if scope == "all" else 8)).all()
        stats = s.groupby(["rule", "model"])["test_MRE"].agg(["mean", "median"]).reset_index()
        df = old.merge(stats, on=["rule", "model"])
        assert len(df) == len(old) == len(cfg.RULES) * len(M)
        assert np.allclose(df["mean"], df["mean_MRE"], rtol=1e-9)
        df = df.rename(columns={"median": "median_MRE"})[["rule", "model", "median_MRE", "mean_MRE", "mean_sk_rank"]]
        df["n_scenarios"] = int(n_scen.iloc[0])
        sources = ["analysis_val/output_artifacts/cache/ensembles.parquet (median MRE)",
                   f"analysis_val/output_artifacts/csv/{src} (mean SK rank)"]
        w.csv(df, f"fig{number}_rule_heatmap_{scope}_median.csv", sources, "figures12_13_median")

        sk = df.pivot(index="rule", columns="model", values="mean_sk_rank").reindex(index=cfg.RULES, columns=M)
        med = df.pivot(index="rule", columns="model", values="median_MRE").reindex(index=cfg.RULES, columns=M)
        sk_v, med_v = sk.values.astype(float), med.values.astype(float)
        fig, ax = plt.subplots(figsize=(max(5.5, len(M) * 0.75), max(1.6, len(cfg.RULES) * 0.52 + 0.6)))
        im = ax.imshow(sk_v, cmap="RdYlGn_r", vmin=np.nanmin(sk_v), vmax=np.nanmax(sk_v), aspect="auto")
        ax.set_xticks(range(len(M)))
        ax.set_xticklabels(M, fontsize=9)
        ax.set_yticks(range(len(cfg.RULES)))
        ax.set_yticklabels(cfg.RULES, fontsize=10)
        for j in range(len(cfg.RULES)):
            for i in range(len(M)):
                ax.text(i, j - 0.17, f"{med_v[j, i]:.2f}", ha="center", va="center", fontsize=8.0)
                ax.text(i, j + 0.17, f"({sk_v[j, i]:.2f})", ha="center", va="center", fontsize=7.0)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04).set_label("Mean SK rank", fontsize=8)
        fig.tight_layout(pad=0.4)
        w.fig(fig, f"fig{number}_rule_heatmap_{scope}_median.pdf", sources, "figures12_13_median")
        out[scope] = df
    return out


def borda_check(w, data):
    s_cols = SCEN + ["model", "run"] + [f"test_{m}" for m in cfg.METRICS]
    sk_single = core.run_sk_metrics(data.singles, SCEN, "model", cfg.METRICS)
    sk_ens = core.run_sk_metrics(data.ens, SCEN, "model", cfg.METRICS)
    s = data.singles[s_cols].copy()
    s["competitor"] = s["model"] + "_S"
    e = data.ens[s_cols].copy()
    e["competitor"] = e["model"] + "_E"
    sk_mixed = core.run_sk_metrics(pd.concat([s, e], ignore_index=True), SCEN, "competitor", cfg.METRICS)

    t6 = pv.val_csv("table6_singles.csv").groupby("model").first()
    t8 = pv.val_csv("table8_ensembles.csv").groupby("model").first()
    rows = []
    for table, sk, col, ref, ref_col in [
            ("Table 6", sk_single, "model", t6, "borda_rank"),
            ("Table 8 (8)", sk_ens, "model", t8, "borda_rank_ensembles"),
            ("Table 8 (16)", sk_mixed, "competitor", t8, "borda_rank_mixed")]:
        if table == "Table 6":
            assert (pv.val_csv("table6_singles.csv").groupby("model")["borda_total"].nunique() == 1).all()
        points = sk.groupby(SCEN + [col], as_index=False)["sk_rank"].sum()
        agg = points.groupby(col)["sk_rank"].agg(["sum", "mean", "median"])
        agg["rank_sum"] = agg["sum"].rank(method="min").astype(int)
        agg["rank_median"] = agg["median"].rank(method="min").astype(int)
        for c, r in agg.iterrows():
            model = c[:-2] if col == "competitor" else c
            if col == "competitor" and not c.endswith("_E"):
                kind = "single"
            else:
                kind = "ensemble" if table.startswith("Table 8") else "single"
            row = {"table": table, "competitor": c, "model": model, "kind": kind,
                   "borda_points_sum": float(r["sum"]), "borda_points_mean": float(r["mean"]),
                   "borda_points_median": float(r["median"]),
                   "rank_sum": int(r["rank_sum"]), "rank_median": int(r["rank_median"])}
            column = "borda_rank_mixed_single" if col == "competitor" and kind == "single" else ref_col
            row["rank_in_table"] = int(ref.loc[model, column])
            if table == "Table 6":
                assert int(ref.loc[model, "borda_total"]) == int(r["sum"])
            rows.append(row)
    df = pd.DataFrame(rows)
    assert (df["rank_sum"] == df["rank_in_table"]).all(), "Borda (sum) does not reproduce the tables"
    sources = ["analysis_val/output_artifacts/cache/variants.parquet",
               "analysis_val/output_artifacts/cache/ensembles.parquet",
               "analysis_val/output_artifacts/csv/table6_singles.csv",
               "analysis_val/output_artifacts/csv/table8_ensembles.csv"]
    w.csv(df, "check_a_borda_aggregation.csv", sources, "borda_check")
    return df


def ensemble_negatives(dataset, size, model, k, rule):
    path = os.path.join(cfg.RESULTS_ENS_DIR, dataset, str(size), loaders._ens_file(model, "top", k, rule))
    with open(path) as f:
        obj = json.load(f)
    preds = [np.asarray(r["y_pred"], dtype=float).ravel() for r in obj["Runs"]]
    return int(sum((p < 0).sum() for p in preds)), int(sum(p.size for p in preds)), int(sum((p < 0).any() for p in preds))


def dagger_check(w, data):
    imp = pv.val_csv("item04_improvement.csv")
    t7 = pv.val_csv("table7_single_vs_ensemble.csv")
    merged = data.single_sc.merge(data.ens_sc[cfg.KEYS + ["k", "rule"] + [c for c in data.ens_sc.columns
                                                                         if c.startswith("test_")]],
                                  on=cfg.KEYS, suffixes=("_single", "_ens"))
    merged["scenario"] = scenario_label(merged)
    neg_single = data.singles.groupby(cfg.KEYS).agg(single_n_negative=("n_negative", "sum"),
                                                   single_n_pred=("n_test", "sum"),
                                                   single_runs_with_negative=("n_negative", lambda x: int((x > 0).sum())))
    merged = merged.merge(neg_single.reset_index(), on=cfg.KEYS)

    rows, summary = [], []
    for model, metric in DAGGER_CASES:
        sub = merged[merged["model"] == model].copy()
        sv, ev = sub[f"test_{metric}_single"], sub[f"test_{metric}_ens"]
        sub["rel_change_pct"] = ((ev - sv) if metric == "SA" else (sv - ev)) / sv.abs() * 100
        ref = (t7 if metric == "SA" else imp)
        ref = ref[(ref["model"] == model) & (ref["metric"] == metric)].iloc[0]
        assert abs(sub["rel_change_pct"].mean() - ref["rel_improvement_mean_pct"]) < 1e-6
        assert abs(sub["rel_change_pct"].median() - ref["rel_improvement_median_pct"]) < 1e-6
        order = sub.sort_values("rel_change_pct")
        picked = pd.concat([order.head(N_LIST).assign(side="largest negative"),
                            order.tail(N_LIST).iloc[::-1].assign(side="largest positive")])
        worst = order["rel_change_pct"].iloc[0]
        summary.append({"model": model, "metric": metric, "n": len(sub),
                        "n_negative_change": int((sub["rel_change_pct"] < 0).sum()),
                        "n_positive_change": int((sub["rel_change_pct"] > 0).sum()),
                        "mean_pct": float(sub["rel_change_pct"].mean()),
                        "median_pct": float(sub["rel_change_pct"].median()),
                        "mean_pct_without_most_negative": float(order["rel_change_pct"].iloc[1:].mean()),
                        "most_negative_pct": float(worst),
                        "n_negative_value_single": int((sv < 0).sum()),
                        "n_negative_value_ens": int((ev < 0).sum())})
        for _, r in picked.iterrows():
            row = {"model": model, "metric": metric, "side": r["side"], "scenario": r["scenario"],
                   "dataset": r["dataset"], "sample_size": int(r["sample_size"]),
                   "single": float(r[f"test_{metric}_single"]), "ensemble": float(r[f"test_{metric}_ens"]),
                   "rel_change_pct": float(r["rel_change_pct"]),
                   "ens_k": int(r["k"]), "ens_rule": r["rule"],
                   "single_n_negative": int(r["single_n_negative"]), "single_n_pred": int(r["single_n_pred"]),
                   "single_runs_with_negative": int(r["single_runs_with_negative"])}
            n_neg, n_pred, runs_neg = ensemble_negatives(r["dataset"], int(r["sample_size"]), model,
                                                         int(r["k"]), r["rule"])
            row.update(ens_n_negative=n_neg, ens_n_pred=n_pred, ens_runs_with_negative=runs_neg)
            rows.append(row)
    detail, summary = pd.DataFrame(rows), pd.DataFrame(summary)
    sources = ["analysis_val/output_artifacts/cache/variants.parquet",
               "analysis_val/output_artifacts/cache/ensembles.parquet",
               "results_ens_val/<dataset>/<size>/<model>_top<k>[_rule]_predictions.json (y_pred)",
               "analysis_val/output_artifacts/csv/item04_improvement.csv"]
    w.csv(detail, "check_b_dagger_scenarios.csv", sources, "dagger_check")
    w.csv(summary, "check_b_dagger_summary.csv", sources, "dagger_check")
    return detail, summary


def ablation_restricted(w, data):
    sub = data.ensembles[(data.ensembles["selector"] == "top") & data.ensembles["k"].isin(cfg.K_RANDOM)
                         & data.ensembles["rule"].isin(cfg.SELECTOR_RULES["random"])]
    n_cand = sub.groupby(cfg.KEYS)[["k", "rule"]].apply(lambda g: len(g.drop_duplicates()))
    assert (n_cand == 8).all()
    top8 = core._select(sub, "val", cfg.KEYS, ["selector", "k", "rule"])
    assert len(top8) == 320 * cfg.NUM_RUNS

    cols = cfg.KEYS + ["run", "test_MRE"]
    parts = []
    for label, df in [("single", data.singles), ("top-k", top8), ("random-k", data.ens_random)]:
        p = df[cols].copy()
        p["version"] = label
        parts.append(p)
    both = pd.concat(parts, ignore_index=True)
    both["competitor"] = both["model"] + "|" + both["version"]
    sk = core.run_sk(both, SCEN, "competitor", "test_MRE")
    sk["model"] = sk["competitor"].str.split("|").str[0]
    sk["version"] = sk["competitor"].str.split("|").str[1]
    sources = ["analysis_val/output_artifacts/cache/variants.parquet",
               "analysis_val/output_artifacts/cache/ensembles.parquet"]
    w.csv(sk, "ablation_restricted_sk_per_scenario.csv", sources, "ablation_restricted")

    borda = core.borda_from_sk(sk, "competitor").set_index("competitor")
    means = both.groupby("competitor")["test_MRE"].mean()
    rows = []
    for model in M:
        for version in ["single", "top-k", "random-k"]:
            c = f"{model}|{version}"
            rows.append({"model": model, "version": version,
                         "mean_sk_rank": float(sk[sk["competitor"] == c]["sk_rank"].mean()),
                         "borda_total": int(borda.loc[c, "borda_total"]), "borda_rank": int(borda.loc[c, "borda_rank"]),
                         "mean_test_MRE": float(means.loc[c])})
    summary = pd.DataFrame(rows)
    w.csv(summary, "ablation_restricted_summary.csv", sources, "ablation_restricted")

    header_1 = r"Model & \multicolumn{3}{c}{Mean SK rank (MRE)} & \multicolumn{3}{c}{Borda rank (of 24)} \\"
    header_2 = r" & Single & Top-$k$ & Random-$k$ & Single & Top-$k$ & Random-$k$ \\"
    body = []
    for model in M:
        s = summary[summary["model"] == model].set_index("version")
        ranks = [s.loc[v, "mean_sk_rank"] for v in ["single", "top-k", "random-k"]]
        best = min(ranks)
        cells = [model] + [bold(f"{x:.2f}") if abs(x - best) < 1e-12 else f"{x:.2f}" for x in ranks]
        cells += [str(int(s.loc[v, "borda_rank"])) for v in ["single", "top-k", "random-k"]]
        body.append(cells)
    notes = [r"One Scott-Knott test per scenario over 24 competitors (8 learners $\times$ 3 versions), "
             r"30 runs each; mean over the 40 scenarios.",
             r"All three versions are selected on validation: single = Borda rank 1 of the 10 variants; "
             r"top-$k$ and random-$k$ each among the same 8 candidates ($k$ = 2..5 $\times$ MEAN, NN)."]
    w.tex(pv.latex_table("lrrrrrr", [header_1, header_2], body, notes), "table_ablation_restricted.tex",
          sources, "ablation_restricted")

    chosen = top8[cfg.KEYS + ["k", "rule"]].drop_duplicates()
    counts = chosen.groupby(["model", "k", "rule"]).size().rename("n_scenarios").reset_index()
    w.csv(counts, "ablation_restricted_choice_counts.csv", sources, "ablation_restricted")
    return summary, counts


def main(out_dir=pv.OUT_DIR):
    w = pv.Writer(out_dir)
    data = pv.Data()
    figures12_13_median(w, data)
    # borda_check(w, data)
    # dagger_check(w, data)
    # ablation_restricted(w, data)


if __name__ == "__main__":
    main()
