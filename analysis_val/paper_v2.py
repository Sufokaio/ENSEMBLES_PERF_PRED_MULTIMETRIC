# Tables and figures of the paper, built from the analysis_val outputs.

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import config as cfg
import core
import paper
from out import bold, ds_label, heatmap

_HERE = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.join(_HERE, "output_artifacts")
OUT_DIR = os.path.join(VAL_DIR, "paper_v2")
SCRIPT = "analysis_val/paper_v2.py"

M = cfg.MODEL_TYPES
SCEN = core.SCEN
TABLE_METRICS = cfg.METRICS
T7_METRICS = cfg.METRICS + ["SA"]


def val_csv(name):
    return pd.read_csv(os.path.join(VAL_DIR, "csv", name))


def cache(name):
    return pd.read_parquet(os.path.join(cfg.RAW_CACHE_DIR, f"{name}.parquet"))


class Writer:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.sources = []

    def _path(self, kind, name, sources, function):
        path = os.path.join(self.out_dir, kind, name)
        if os.path.exists(path):
            raise FileExistsError(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.sources.append((f"{kind}/{name}", "; ".join(sources), f"{SCRIPT}:{function}"))
        return path

    def csv(self, df, name, sources, function):
        df.to_csv(self._path("csv", name, sources, function), index=False)

    def tex(self, lines, name, sources, function):
        with open(self._path("latex", name, sources, function), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def fig(self, fig, name, sources, function):
        fig.savefig(self._path("figures", name, sources, function), format="pdf")
        plt.close(fig)


def p_fmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def iqr(values):
    v = np.asarray(values, dtype=float)
    return float(np.percentile(v, 75) - np.percentile(v, 25))


def latex_table(col_spec, header_rows, body, notes=None):
    lines = [r"\begin{tabular}{" + col_spec + "}", r"\toprule"] + header_rows + [r"\midrule"]
    for row in body:
        lines.append(row if isinstance(row, str) else " & ".join(str(c) for c in row) + r" \\")
    lines.append(r"\bottomrule")
    n_cols = len(col_spec.replace("|", ""))
    for note in notes or []:
        lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{" + note + r"} \\")
    lines.append(r"\end{tabular}")
    return lines


class Data:
    def __init__(self):
        self.baseline = cache("baseline")
        self.variants = core.add_sa_d(cache("variants"), self.baseline)
        self.ensembles = core.add_sa_d(cache("ensembles"), self.baseline)
        self.diversity = cache("diversity")
        self.singles = core.select_best_singles(self.variants, "val")
        self.ens = core.select_best_ensembles(self.ensembles, "val")
        self.ens_random = core.select_best_ensembles(self.ensembles, "val", selectors=["random"])
        cols = [f"test_{m}" for m in cfg.ALL_METRICS + ["SA", "D"]]
        self.single_sc = self.singles.groupby(cfg.KEYS, as_index=False)[cols].mean()
        self.ens_sc = self.ensembles_scenario_means(self.ens, cols)
        self.check_selection()

    @staticmethod
    def ensembles_scenario_means(df, cols):
        return df.groupby(cfg.KEYS + ["selector", "k", "rule"], as_index=False)[cols].mean()

    def check_selection(self):
        sel = val_csv("selected_single_and_ensemble.csv")
        s = self.singles[cfg.KEYS + ["variant"]].drop_duplicates()
        e = self.ens[cfg.KEYS + ["k", "rule"]].drop_duplicates()
        m = sel.merge(s, on=cfg.KEYS).merge(e, on=cfg.KEYS)
        assert len(m) == 320
        assert (m["single_variant"] == m["variant"]).all()
        assert ((m["ens_k"] == m["k"]) & (m["ens_rule"] == m["rule"])).all()


def rank_table(w, sc, sk_csv, name, versions, borda_cols, title_cols, sources, function):
    rows = []
    for version in versions:
        for model in M:
            sub = sc[sc["model"] == model]
            for metric in TABLE_METRICS + ["SA", "D"]:
                v = sub[f"test_{metric}"]
                center = v.mean() if version == "a" else v.median()
                spread = v.std(ddof=1) if version == "a" else iqr(v)
                sk = np.nan
                if metric in TABLE_METRICS:
                    sk = float(sk_csv[(sk_csv["model"] == model) & (sk_csv["metric"] == metric)]["mean_sk_rank"].iloc[0])
                rows.append({"version": version, "model": model, "metric": metric,
                             "statistic": "mean" if version == "a" else "median",
                             "value": float(center), "spread": float(spread),
                             "spread_kind": "std" if version == "a" else "IQR", "mean_sk_rank": sk})
    df = pd.DataFrame(rows)
    for col in borda_cols:
        df[col] = df["model"].map(sk_csv.groupby("model")[col].first())
    w.csv(df, f"{name}.csv", sources, function)

    for version in versions:
        d = df[df["version"] == version]
        best = {m: d[d["metric"] == m]["value"].min() for m in TABLE_METRICS}
        best_sa = d[d["metric"] == "SA"]["value"].max()
        label = "Mean (std)" if version == "a" else "Median (IQR)"
        header_1 = ("Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in TABLE_METRICS)
                    + r" & SA ($\Delta$) & " + " & ".join(title_cols) + r" \\")
        header_2 = (" & " + " & ".join([f"{label} & SK"] * len(TABLE_METRICS))
                    + " & " + label + " & " * len(title_cols) + r" \\")
        body = []
        for model in M:
            cells = [model]
            for m in TABLE_METRICS:
                r = d[(d["model"] == model) & (d["metric"] == m)].iloc[0]
                value = f"{r['value']:.2f} ({r['spread']:.2f})"
                if abs(r["value"] - best[m]) < 1e-12:
                    value = bold(value)
                cells += [value, f"{r['mean_sk_rank']:.2f}"]
            sa = d[(d["model"] == model) & (d["metric"] == "SA")].iloc[0]
            dd = d[(d["model"] == model) & (d["metric"] == "D")].iloc[0]
            sa_cell = f"{sa['value']:.3f} ({dd['value']:.2f})"
            cells.append(bold(sa_cell) if abs(sa["value"] - best_sa) < 1e-12 else sa_cell)
            for col in borda_cols:
                b = int(d[d["model"] == model][col].iloc[0])
                cells.append(bold(b) if b == 1 else str(b))
            body.append(cells)
        notes = [("Value: " + ("mean (standard deviation)" if version == "a" else "median (interquartile range)")
                  + r" across the 40 scenario means; SK: mean Scott-Knott rank; SA ($\Delta$): "
                  + ("mean" if version == "a" else "median") + " across the 40 scenarios.")]
        w.tex(latex_table("l" + "rr" * len(TABLE_METRICS) + "r" + "r" * len(borda_cols),
                          [header_1, header_2], body, notes),
              f"{name}_{version}.tex", sources, function)
    return df


def table6(w, data):
    sk = val_csv("table6_singles.csv")
    # return rank_table(w, data.single_sc, sk, "table6_singles", ["a", "b"], ["borda_rank"], ["Borda"],
    return rank_table(w, data.single_sc, sk, "table6_singles", ["b"], ["borda_rank"], ["Borda"],
                      ["analysis_val/output_artifacts/cache/variants.parquet",
                       "analysis_val/output_artifacts/csv/table6_singles.csv (SK, Borda)"], "table6")


def table8(w, data):
    sk = val_csv("table8_ensembles.csv")
    # return rank_table(w, data.ens_sc, sk, "table8_ensembles", ["a", "b"],
    return rank_table(w, data.ens_sc, sk, "table8_ensembles", ["b"],
                      ["borda_rank_ensembles", "borda_rank_mixed"], ["Borda (8)", "Borda (16)"],
                      ["analysis_val/output_artifacts/cache/ensembles.parquet",
                       "analysis_val/output_artifacts/csv/table8_ensembles.csv (SK, Borda)"], "table8")


def figure6(w, data):
    rows = []
    for model in M:
        s = data.single_sc[data.single_sc["model"] == model]["test_D"]
        e = data.ens_sc[data.ens_sc["model"] == model]["test_D"]
        rows.append({"model": model,
                     "D_single_mean": s.mean(), "D_single_sd": s.std(ddof=1),
                     "D_single_median": s.median(), "D_single_q25": s.quantile(0.25), "D_single_q75": s.quantile(0.75),
                     "D_ens_mean": e.mean(), "D_ens_sd": e.std(ddof=1),
                     "D_ens_median": e.median(), "D_ens_q25": e.quantile(0.25), "D_ens_q75": e.quantile(0.75)})
    res = pd.DataFrame(rows)
    sources = ["analysis_val/output_artifacts/cache/variants.parquet",
               "analysis_val/output_artifacts/cache/ensembles.parquet",
               "analysis_val/output_artifacts/cache/baseline.parquet"]
    w.csv(res, "fig06_d_forest.csv", sources, "figure6")

    # for version in ["mean", "median"]:
    for version in ["median"]:
        order = res.sort_values(f"D_single_{version}").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        off = 0.15
        for i, r in order.iterrows():
            color = cfg.MODEL_COLORS[r["model"]]
            xs, xe = r[f"D_single_{version}"], r[f"D_ens_{version}"]
            if version == "mean":
                err_s, err_e = r["D_single_sd"], r["D_ens_sd"]
            else:
                err_s = [[xs - r["D_single_q25"]], [r["D_single_q75"] - xs]]
                err_e = [[xe - r["D_ens_q25"]], [r["D_ens_q75"] - xe]]
            ax.plot([xs, xe], [i + off, i - off], color=color, linewidth=1.1, alpha=0.65, zorder=1)
            ax.errorbar(xs, i + off, xerr=err_s, fmt="none", color=color, capsize=2, linewidth=0.7, alpha=0.5)
            ax.scatter(xs, i + off, marker="o", s=48, facecolors="none", edgecolors=color, linewidth=1.5, zorder=3)
            ax.errorbar(xe, i - off, xerr=err_e, fmt="none", color=color, capsize=2, linewidth=0.7, alpha=0.5)
            ax.scatter(xe, i - off, marker="o", s=48, color=color, zorder=3)
        ax.axvline(0, color="red", linewidth=1.0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order["model"].tolist())
        ax.set_xlabel(r"$\Delta$ (" + ("mean, error bars = std" if version == "mean"
                                         else "median, error bars = IQR") + ")")
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="gray", markerfacecolor="none", markersize=7, linestyle="none",
                   label="Best single"),
            Line2D([0], [0], marker="o", color="gray", markersize=7, linestyle="none", label="Best ensemble"),
            Line2D([0], [0], color="red", linewidth=1.0, label=r"$\Delta$ = 0"),
        ], fontsize=7, loc="lower right")
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        w.fig(fig, f"fig06_d_forest_{version}.pdf", sources, "figure6")
    return res


def figures7_8(w):
    datasets = None
    tables = {}
    for number, scope in [("07", "all"), ("08", "s1")]:
        src = f"fig{number}_singles_mre_{scope}.csv"
        df = val_csv(src)
        datasets = sorted(df["dataset"].unique())
        mre = df.pivot(index="model", columns="dataset", values="mean_MRE").reindex(index=M, columns=datasets)
        tables[scope] = df.pivot(index="model", columns="dataset", values="mean_sk_rank").reindex(index=M, columns=datasets)
        values = mre.values.astype(float)
        fig, ax = plt.subplots(figsize=(6.4, 2.75))
        im = heatmap(ax, values, M, [ds_label(d) for d in datasets], fmt="{:.2f}",
                     vmin=0, vmax=float(np.nanpercentile(values, 95)))
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("Mean MRE")
        fig.tight_layout(pad=0.3)
        w.fig(fig, f"fig{number}_singles_mre_{scope}.pdf", [f"analysis_val/output_artifacts/csv/{src}"], "figures7_8")
    return tables, datasets


def table_sk_ranks(w, tables, datasets):
    rows = []
    for panel, scope in [("a", "all"), ("b", "s1")]:
        t = tables[scope]
        for model in M:
            for ds in datasets:
                rows.append({"panel": panel, "scope": "all sizes" if scope == "all" else "S1",
                             "model": model, "dataset": ds, "mean_sk_rank": float(t.loc[model, ds])})
    sources = ["analysis_val/output_artifacts/csv/fig07_singles_mre_all.csv",
               "analysis_val/output_artifacts/csv/fig08_singles_mre_s1.csv"]
    w.csv(pd.DataFrame(rows), "table_sk_ranks_fig7_fig8.csv", sources, "table_sk_ranks")
    n = len(datasets) + 1
    header = "Model & " + " & ".join(ds_label(d) for d in datasets) + r" \\"
    body = []
    for panel, scope, label in [("a", "all", "(a) All training sizes (Figure 7)"), ("b", "s1", "(b) S1 (Figure 8)")]:
        if panel == "b":
            body.append(r"\midrule")
        body.append(r"\multicolumn{" + str(n) + r"}{l}{\textit{" + label + r"}} \\")
        for model in M:
            body.append([model] + [f"{tables[scope].loc[model, ds]:.1f}" for ds in datasets])
    w.tex(latex_table("l" + "r" * len(datasets), [header], body,
                      ["Mean Scott-Knott rank of the best single variant (MRE) per learner and system."]),
          "table_sk_ranks_fig7_fig8.tex", sources, "table_sk_ranks")


def figure9(w):
    src = "fig09_singles_rank_s1.csv"
    borda = val_csv(src)
    datasets = sorted(borda["dataset"].unique())
    pivot = borda.pivot(index="dataset", columns="model", values="borda_rank").reindex(index=datasets, columns=M)
    n = len(M)
    fig, ax = plt.subplots(figsize=(max(5.5, n * 0.75), max(2.6, len(datasets) * 0.42)))
    im = heatmap(ax, pivot.values.astype(float), [ds_label(d) for d in datasets], M,
                 fmt="{:.0f}", cmap=plt.get_cmap("YlOrRd", n), vmin=0.5, vmax=n + 0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Borda rank")
    cbar.set_ticks(range(1, n + 1))
    fig.tight_layout(pad=0.4)
    w.fig(fig, "fig09_singles_rank_s1.pdf", [f"analysis_val/output_artifacts/csv/{src}"], "figure9")


def table7(w):
    imp = val_csv("item04_improvement.csv")
    t7 = val_csv("table7_single_vs_ensemble.csv")
    rows = []
    for model in M:
        for metric in T7_METRICS:
            if metric == "SA":
                r = t7[(t7["model"] == model) & (t7["metric"] == "SA")].iloc[0]
                a12 = imp[(imp["model"] == model) & (imp["metric"] == "MAE")].iloc[0]
            else:
                r = imp[(imp["model"] == model) & (imp["metric"] == metric)].iloc[0]
                a12 = r
            row = {"model": model, "metric": metric, "wins": int(r["W"]), "ties": int(r["T"]),
                   "losses": int(r["L"]), "n": int(r["N"]),
                   "rel_improvement_mean_pct": float(r["rel_improvement_mean_pct"]),
                   "rel_improvement_median_pct": float(r["rel_improvement_median_pct"]),
                   "a12_median": float(a12["a12_median"]), "a12_mean": float(a12["a12_mean"])}
            row["flag_negative_mean_despite_majority_wins"] = bool(
                row["rel_improvement_mean_pct"] < 0 and row["wins"] > row["n"] / 2)
            rows.append(row)
    df = pd.DataFrame(rows)
    sources = ["analysis_val/output_artifacts/csv/item04_improvement.csv",
               "analysis_val/output_artifacts/csv/table7_single_vs_ensemble.csv (SA)"]
    w.csv(df, "table7_single_vs_ensemble.csv", sources, "table7")

    header_1 = "Model & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{m}}}" for m in T7_METRICS) + r" \\"
    header_2 = " & " + " & ".join([r"Wins & Mean\,\% & Med.\,\% & $A_{12}$"] * len(T7_METRICS)) + r" \\"
    body = []
    for model in M:
        cells = [model]
        for metric in T7_METRICS:
            r = df[(df["model"] == model) & (df["metric"] == metric)].iloc[0]
            mean = f"{r['rel_improvement_mean_pct']:+.1f}"
            if r["flag_negative_mean_despite_majority_wins"]:
                mean += r"$^{\dagger}$"
            cells += [f"{r['wins']}/{r['n']}", mean, f"{r['rel_improvement_median_pct']:+.1f}",
                      f"{r['a12_median']:.2f}"]
        body.append(cells)
    notes = [r"Wins: scenarios (of 40) in which the best ensemble has the lower mean test value (higher for SA).",
             r"Mean/Med.\,\%: relative improvement of the ensemble over the single across the 40 scenarios.",
             r"$A_{12}$: median over scenarios of the probability that an ensemble run beats a single run "
             r"(SA uses the MAE value).",
             r"$^{\dagger}$ negative mean improvement despite a majority of wins."]
    w.tex(latex_table("l" + "rrrr" * len(T7_METRICS), [header_1, header_2], body, notes),
          "table7_single_vs_ensemble.tex", sources, "table7")
    return df


def figure10(w):
    src = "fig10_gap_close.csv"
    df = val_csv(src)
    datasets = sorted(df["dataset"].unique())
    delta = df.pivot(index="model", columns="dataset", values="delta").reindex(index=M, columns=datasets)
    mat = delta.values.astype(float)
    vmax = max(float(np.nanmax(np.abs(mat))), 0.5)
    fig, ax = plt.subplots(figsize=(max(6.0, len(datasets) * 0.82), max(2.4, len(M) * 0.38)))
    im = heatmap(ax, mat, M, [ds_label(d) for d in datasets], fmt="{:+.1f}", cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04).set_label(r"$\Delta$ SK rank")
    fig.tight_layout(pad=0.4)
    w.fig(fig, "fig10_gap_close.pdf", [f"analysis_val/output_artifacts/csv/{src}"], "figure10")


def ablation(w, data):
    cols = cfg.KEYS + ["run", "test_MRE"]
    parts = []
    for label, df in [("single", data.singles), ("top-k", data.ens), ("random-k", data.ens_random)]:
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
    w.csv(sk, "ablation_sk_per_scenario.csv", sources, "ablation")

    borda = core.borda_from_sk(sk, "competitor").set_index("competitor")
    means = both.groupby(["competitor"])["test_MRE"].mean()
    rows = []
    for model in M:
        for version in ["single", "top-k", "random-k"]:
            c = f"{model}|{version}"
            r = sk[sk["competitor"] == c]["sk_rank"]
            rows.append({"model": model, "version": version, "mean_sk_rank": float(r.mean()),
                         "borda_total": int(borda.loc[c, "borda_total"]), "borda_rank": int(borda.loc[c, "borda_rank"]),
                         "mean_test_MRE": float(means.loc[c])})
    summary = pd.DataFrame(rows)
    w.csv(summary, "ablation_summary.csv", sources, "ablation")

    header_1 = (r"Model & \multicolumn{3}{c}{Mean SK rank (MRE)} & \multicolumn{3}{c}{Borda rank (of 24)} \\")
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
             r"top-$k$ among 27, random-$k$ among 8 ensembles."]
    w.tex(latex_table("lrrrrrr", [header_1, header_2], body, notes), "table_ablation.tex", sources, "ablation")

    # counts = []
    # for label, df in [("top-k", data.ens), ("random-k", data.ens_random)]:
    #     chosen = df[cfg.KEYS + ["k", "rule"]].drop_duplicates()
    #     for (model, k, rule), g in chosen.groupby(["model", "k", "rule"]):
    #         counts.append({"selector": label, "model": model, "k": int(k), "rule": rule, "n_scenarios": len(g)})
    # counts = pd.DataFrame(counts)
    # w.csv(counts, "ablation_choice_counts.csv", sources, "ablation")

    # lines = []
    # for label, ks, rules in [("top-k", cfg.K_TOP, cfg.RULES), ("random-k", cfg.K_RANDOM, ["MEAN", "NN"])]:
    #     sub = counts[counts["selector"] == label]
    #     header = "Model & " + " & ".join(f"$k$={k}" for k in ks) + " & " + " & ".join(rules) + r" \\"
    #     body = []
    #     for model in M + ["All"]:
    #         s = sub if model == "All" else sub[sub["model"] == model]
    #         body.append([model] + [int(s[s["k"] == k]["n_scenarios"].sum()) for k in ks]
    #                     + [int(s[s["rule"] == r]["n_scenarios"].sum()) for r in rules])
    #     lines += latex_table("l" + "r" * (len(ks) + len(rules)), [header], body,
    #                          [f"Number of scenarios in which each $k$ and rule is selected on validation ({label})."])
    #     lines.append("")
    # w.tex(lines, "table_ablation_choice_counts.tex", sources, "ablation")
    # return summary, counts
    return summary


def member_spread(data):
    vm = data.variants.groupby(cfg.KEYS + ["variant", "val_borda_rank"], as_index=False)["test_MRE"].mean()
    rows = []
    for key, g in vm.groupby(cfg.KEYS):
        mre = g.sort_values("val_borda_rank")["test_MRE"].to_numpy()
        for k in cfg.K_TOP:
            mem = mre[:k]
            rows.append(dict(zip(cfg.KEYS, key), k=k, spread=float(mem.mean() / mem.min())))
    return pd.DataFrame(rows)


def diversity_tables(w, data):
    spread = member_spread(data)
    top = data.diversity[data.diversity["selector"] == "top"]
    per_ens = top.groupby(cfg.KEYS + ["k"], as_index=False)[["err_corr", "pred_corr", "rel_ambiguity"]].mean()
    per_ens = per_ens.merge(spread, on=cfg.KEYS + ["k"])
    best_k = data.ens[cfg.KEYS + ["k"]].drop_duplicates()
    best = per_ens.merge(best_k, on=cfg.KEYS + ["k"])
    imp = val_csv("item04_improvement.csv")
    imp = imp[imp["metric"] == "MRE"].set_index("model")

    rows = []
    for model in M:
        a = per_ens[per_ens["model"] == model]
        b = best[best["model"] == model]
        rows.append({"model": model,
                     "err_corr_mean": a["err_corr"].mean(), "pred_corr_mean": a["pred_corr"].mean(),
                     "rel_ambiguity_mean": a["rel_ambiguity"].mean(), "member_spread_mean": a["spread"].mean(),
                     "median_mre_gain_best_ensemble_pct": float(imp.loc[model, "rel_improvement_median_pct"]),
                     "best_ens_err_corr_mean": b["err_corr"].mean(), "best_ens_pred_corr_mean": b["pred_corr"].mean(),
                     "best_ens_rel_ambiguity_mean": b["rel_ambiguity"].mean(),
                     "best_ens_member_spread_mean": b["spread"].mean()})
    summary = pd.DataFrame(rows)
    sources = ["analysis_val/output_artifacts/cache/diversity.parquet",
               "analysis_val/output_artifacts/cache/variants.parquet",
               "analysis_val/output_artifacts/csv/item04_improvement.csv"]
    w.csv(summary, "table_diversity_summary.csv", sources, "diversity_tables")
    header = (r"Model & Error corr. & Prediction corr. & Rel. ambiguity & Accuracy spread & "
              r"Median MRE gain (\%) \\")
    body = [[r["model"], f"{r['err_corr_mean']:.3f}", f"{r['pred_corr_mean']:.3f}", f"{r['rel_ambiguity_mean']:.3f}",
             f"{r['member_spread_mean']:.3f}", f"{r['median_mre_gain_best_ensemble_pct']:+.1f}"]
            for _, r in summary.iterrows()]
    notes = [r"Diversity: mean over all top-$k$ ensembles ($k$ = 2..10) and the 40 scenarios, from the members' "
             r"test predictions. Relative ambiguity: Krogh--Vedelsby ambiguity / mean member MSE.",
             r"Accuracy spread: mean member MRE / MRE of the best member. "
             r"Median MRE gain: best ensemble vs best single, median over the 40 scenarios."]
    w.tex(latex_table("lrrrrr", [header], body, notes), "table_diversity_summary.tex", sources, "diversity_tables")

    # detail = val_csv("item03_diversity_detail.csv")
    # detail = detail[detail["selector"] == "top"].merge(spread, on=cfg.KEYS + ["k"])
    # rho_rows = []
    # for measure in ["err_corr", "spread"]:
    #     for rule in ["MEAN", "IRWM"]:
    #         for model in M:
    #             for k in cfg.K_TOP:
    #                 s = detail[(detail["model"] == model) & (detail["rule"] == rule) & (detail["k"] == k)]
    #                 rho, p = spearmanr(s[measure], s["gain_pct_MRE"])
    #                 rho_rows.append({"measure": measure, "rule": rule, "model": model, "k": k,
    #                                  "n": len(s), "spearman_rho": float(rho), "p_value": float(p)})
    # rho = pd.DataFrame(rho_rows)
    # sources_rho = ["analysis_val/output_artifacts/csv/item03_diversity_detail.csv",
    #                "analysis_val/output_artifacts/cache/variants.parquet (accuracy spread)"]
    # w.csv(rho, "table_diversity_spearman_per_k.csv", sources_rho, "diversity_tables")
    # assert (rho["n"] == 40).all()
    # for measure, title in [("err_corr", "error correlation"), ("spread", "member accuracy spread")]:
    #     header = "Model & " + " & ".join(f"$k$={k}" for k in cfg.K_TOP) + r" \\"
    #     body = []
    #     for panel, rule in [("a", "MEAN"), ("b", "IRWM")]:
    #         if panel == "b":
    #             body.append(r"\midrule")
    #         body.append(r"\multicolumn{" + str(len(cfg.K_TOP) + 1) + r"}{l}{\textit{(" + panel + ") " + rule + r"}} \\")
    #         for model in M:
    #             s = rho[(rho["measure"] == measure) & (rho["rule"] == rule) & (rho["model"] == model)].set_index("k")
    #             body.append([model] + [f"{s.loc[k, 'spearman_rho']:+.2f} ({p_fmt(s.loc[k, 'p_value'])})"
    #                                    for k in cfg.K_TOP])
    #     notes = [rf"Spearman $\rho$ (p-value) between {title} and the MRE gain of the top-$k$ ensemble over the "
    #              r"best single; uncorrected, $n$ = 40 per cell."]
    #     w.tex(latex_table("l" + "r" * len(cfg.K_TOP), [header], body, notes),
    #           f"table_appendix_spearman_{measure}.tex", sources_rho, "diversity_tables")
    return summary


def table_s1_stability(w):
    src = "item08_s1_stability.csv"
    st = val_csv(src)
    st = st[st["metric"] == "MRE"]
    rows = []
    for model in M:
        r = {"model": model}
        for kind in ["single", "ensemble"]:
            s = st[(st["model"] == model) & (st["kind"] == kind)]
            r[f"{kind}_median_std"] = float(s["std"].median())
            r[f"{kind}_median_iqr"] = float(s["iqr"].median())
        rows.append(r)
    df = pd.DataFrame(rows)
    w.csv(df, "table_s1_stability.csv", [f"analysis_val/output_artifacts/csv/{src}"], "table_s1_stability")
    cols = ["single_median_std", "single_median_iqr", "ensemble_median_std", "ensemble_median_iqr"]
    header_1 = r"Model & \multicolumn{2}{c}{Best single} & \multicolumn{2}{c}{Best ensemble} \\"
    header_2 = r" & Std & IQR & Std & IQR \\"
    body = [[r["model"]] + [f"{r[c]:.2f}" for c in cols] for _, r in df.iterrows()]
    body.append(r"\midrule")
    body.append(["Range"] + [f"{df[c].min():.2f}--{df[c].max():.2f}" for c in cols])
    notes = ["Median over the 8 systems at S1 of the standard deviation and IQR of the test MRE across the 30 runs."]
    w.tex(latex_table("lrrrr", [header_1, header_2], body, notes), "table_s1_stability.tex",
          [f"analysis_val/output_artifacts/csv/{src}"], "table_s1_stability")


def figure11(w):
    src = "fig11_k_sk_rank_curve.csv"
    mean_sk = val_csv(src)
    ks = sorted(mean_sk["k"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    paper._k_curve_axes(axes, mean_sk, ks)
    for col, rule in enumerate(cfg.RULES):
        axes[col].set_title(rule, fontsize=11, fontweight="bold")
        axes[col].set_xlabel("$k$")
    axes[0].set_ylabel("Mean SK rank")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="center left", bbox_to_anchor=(1.04, 0.5), title="Base type", ncol=2)
    fig.tight_layout()
    w.fig(fig, "fig11_k_sk_rank_curve.pdf", [f"analysis_val/output_artifacts/csv/{src}"], "figure11")

    src = "item07_k_saturation_per_dataset.csv"
    per_ds = val_csv(src)
    datasets = sorted(per_ds["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 3, figsize=(12.0, 2.7 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        paper._k_curve_axes(axes[i], per_ds[per_ds["dataset"] == ds], ks)
        axes[i][0].set_ylabel(f"{ds_label(ds)}\nMean SK rank")
        for col, rule in enumerate(cfg.RULES):
            if i == 0:
                axes[i][col].set_title(rule, fontsize=11, fontweight="bold")
            if i == len(datasets) - 1:
                axes[i][col].set_xlabel("$k$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, title="Base type", bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout()
    w.fig(fig, "fig11_k_sk_rank_curve_per_dataset.pdf", [f"analysis_val/output_artifacts/csv/{src}"], "figure11")


def figures12_13(w):
    for number, scope in [("12", "all"), ("13", "s1")]:
        src = f"fig{number}_rule_heatmap_{scope}.csv"
        df = val_csv(src)
        sk = df.pivot(index="rule", columns="model", values="mean_sk_rank").reindex(index=cfg.RULES, columns=M)
        mre = df.pivot(index="rule", columns="model", values="mean_MRE").reindex(index=cfg.RULES, columns=M)
        sk_v, mre_v = sk.values.astype(float), mre.values.astype(float)
        fig, ax = plt.subplots(figsize=(max(5.5, len(M) * 0.75), max(1.6, len(cfg.RULES) * 0.52 + 0.6)))
        im = ax.imshow(sk_v, cmap="RdYlGn_r", vmin=np.nanmin(sk_v), vmax=np.nanmax(sk_v), aspect="auto")
        ax.set_xticks(range(len(M)))
        ax.set_xticklabels(M, fontsize=9)
        ax.set_yticks(range(len(cfg.RULES)))
        ax.set_yticklabels(cfg.RULES, fontsize=10)
        for j in range(len(cfg.RULES)):
            for i in range(len(M)):
                ax.text(i, j - 0.17, f"{mre_v[j, i]:.2f}", ha="center", va="center", fontsize=8.0)
                ax.text(i, j + 0.17, f"({sk_v[j, i]:.2f})", ha="center", va="center", fontsize=7.0)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04).set_label("Mean SK rank", fontsize=8)
        fig.tight_layout(pad=0.4)
        w.fig(fig, f"fig{number}_rule_heatmap_{scope}.pdf", [f"analysis_val/output_artifacts/csv/{src}"],
              "figures12_13")


def table_rules(w):
    sk = val_csv("item06_rules_sk_summary.csv")
    wtl = val_csv("item06_rules_wtl.csv")
    sel = val_csv("selected_single_and_ensemble.csv")
    sel["is_s1"] = sel["sample_size"] == sel.groupby("dataset")["sample_size"].transform("min")
    pairs = [("MEAN", "IRWM"), ("MEAN", "NN"), ("IRWM", "NN")]
    rows = []
    for scope in ["all", "S1"]:
        s_sel = sel if scope == "all" else sel[sel["is_s1"]]
        for model in M + ["ALL"]:
            r = {"scope": scope, "model": model}
            for rule in cfg.RULES:
                x = sk[(sk["scope"] == scope) & (sk["model"] == model) & (sk["rule"] == rule)].iloc[0]
                r[f"mean_sk_rank_MRE_{rule}"] = float(x["mean_sk_rank_MRE"])
                r[f"borda_rank_{rule}"] = int(x["borda_rank"])
                chosen = s_sel if model == "ALL" else s_sel[s_sel["model"] == model]
                r[f"selected_{rule}"] = int((chosen["ens_rule"] == rule).sum())
            for a, b in pairs:
                x = wtl[(wtl["scope"] == scope) & (wtl["model"] == model) & (wtl["metric"] == "MRE")
                        & (wtl["rule_a"] == a) & (wtl["rule_b"] == b)].iloc[0]
                r[f"{a}_vs_{b}_W"], r[f"{a}_vs_{b}_T"], r[f"{a}_vs_{b}_L"] = (
                    int(x["a_wins"]), int(x["ties"]), int(x["b_wins"]))
            rows.append(r)
    df = pd.DataFrame(rows)
    sources = ["analysis_val/output_artifacts/csv/item06_rules_sk_summary.csv",
               "analysis_val/output_artifacts/csv/item06_rules_wtl.csv",
               "analysis_val/output_artifacts/csv/selected_single_and_ensemble.csv"]
    w.csv(df, "table_rules_head_to_head.csv", sources, "table_rules")
    lines = []
    for scope, label in [("all", "all training sizes (360 learner-scenario-$k$ units per learner)"),
                         ("S1", "S1 (72 units per learner)")]:
        header_1 = (r"Model & \multicolumn{3}{c}{Mean SK rank (MRE)} & "
                    + " & ".join(rf"\multicolumn{{3}}{{c}}{{{a} vs {b}}}" for a, b in pairs)
                    + r" & \multicolumn{3}{c}{Selected on validation} \\")
        header_2 = " & MEAN & IRWM & NN & " + " & ".join(["W & T & L"] * len(pairs)) + r" & MEAN & IRWM & NN \\"
        body = []
        for _, r in df[df["scope"] == scope].iterrows():
            cells = [r["model"].replace("ALL", "All")] + [f"{r[f'mean_sk_rank_MRE_{x}']:.2f}" for x in cfg.RULES]
            for a, b in pairs:
                cells += [r[f"{a}_vs_{b}_W"], r[f"{a}_vs_{b}_T"], r[f"{a}_vs_{b}_L"]]
            cells += [r[f"selected_{x}"] for x in cfg.RULES]
            body.append(cells)
        notes = [f"Top-$k$ ensembles, {label}. One Scott-Knott test over the three rules per learner, scenario and $k$;",
                 r"W/T/L from the SK ranks (MRE). Selected: scenarios in which the rule belongs to the "
                 r"validation-selected best ensemble."]
        lines += latex_table("lrrr" + "rrr" * len(pairs) + "rrr", [header_1, header_2], body, notes)
        lines.append("")
    w.tex(lines, "table_rules_head_to_head.tex", sources, "table_rules")
    return df


def main():
    if os.path.exists(OUT_DIR):
        raise FileExistsError(OUT_DIR)
    w = Writer(OUT_DIR)
    data = Data()

    table6(w, data)
    figure6(w, data)
    tables, datasets = figures7_8(w)
    table_sk_ranks(w, tables, datasets)
    figure9(w)
    table7(w)
    figure10(w)
    ablation(w, data)
    diversity_tables(w, data)
    # table_s1_stability(w)
    table8(w, data)
    figure11(w)
    # figures12_13(w)
    # table_rules(w)


if __name__ == "__main__":
    main()
