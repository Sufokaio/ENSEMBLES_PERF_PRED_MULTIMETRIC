# Analysis items 1-4, 6, 8 and 9 (items 5, 7 and 10 are in paper.py).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import config as cfg
import core
import paper
from out import ds_label, heatmap, tabular

SCEN = core.SCEN
MODELS_ALL = cfg.MODEL_TYPES + ["ALL"]


def _of_model(df, model):
    return df if model == "ALL" else df[df["model"] == model]


# Item 1: test-ranked ensembles (results_new) against validation-ranked ensembles (results_ens_val)
def item01_old_vs_new(ctx):
    cols = [f"test_{m}" for m in cfg.METRICS]
    by = cfg.KEYS + ["k", "rule"]
    new = ctx.ensembles[ctx.ensembles["selector"] == "top"].groupby(by, as_index=False)[cols].mean()
    old = ctx.old.groupby(by, as_index=False)[cols].mean()
    detail = old.merge(new, on=by, suffixes=("_old", "_new"))
    for m in cfg.METRICS:
        detail[f"change_pct_{m}"] = ((detail[f"test_{m}_new"] - detail[f"test_{m}_old"])
                                     / detail[f"test_{m}_old"].abs() * 100)
    ctx.w.csv(detail, "item01_old_vs_new_detail.csv")

    rows = []
    for model in MODELS_ALL:
        for rule in cfg.RULES + ["ALL"]:
            sub = _of_model(detail, model)
            if rule != "ALL":
                sub = sub[sub["rule"] == rule]
            for m in cfg.METRICS:
                o, n, ch = sub[f"test_{m}_old"], sub[f"test_{m}_new"], sub[f"change_pct_{m}"]
                same = np.isclose(o, n, rtol=1e-9, atol=0.0)
                rows.append({
                    "model": model, "rule": rule, "metric": m, "n": len(sub),
                    "mean_old": float(o.mean()), "mean_new": float(n.mean()),
                    "median_old": float(o.median()), "median_new": float(n.median()),
                    "change_pct_mean": float(ch.mean()), "change_pct_median": float(ch.median()),
                    "new_better": int(((n < o) & ~same).sum()), "equal": int(same.sum()),
                    "new_worse": int(((n > o) & ~same).sum()),
                })
    summary = pd.DataFrame(rows)
    ctx.w.csv(summary, "item01_old_vs_new_summary.csv")

    ranks = ctx.variants.groupby(cfg.KEYS + ["variant"], as_index=False)[["val_borda_rank", "test_borda_rank"]].first()
    agree_rows = []
    for key, g in ranks.groupby(cfg.KEYS):
        val_order = g.sort_values("val_borda_rank")["variant"].tolist()
        test_order = g.sort_values("test_borda_rank")["variant"].tolist()
        row = dict(zip(cfg.KEYS, key),
                   spearman=float(spearmanr(g["val_borda_rank"], g["test_borda_rank"])[0]),
                   same_best=val_order[0] == test_order[0],
                   test_rank_of_val_best=test_order.index(val_order[0]) + 1)
        for k in cfg.K_TOP:
            row[f"overlap_top{k}"] = len(set(val_order[:k]) & set(test_order[:k])) / k
        agree_rows.append(row)
    agree = pd.DataFrame(agree_rows)
    ctx.w.csv(agree, "item01_ranking_agreement_detail.csv")

    agree_summary = []
    for model in MODELS_ALL:
        sub = _of_model(agree, model)
        agree_summary.append({
            "model": model, "n_scenarios": len(sub),
            "spearman_median": float(sub["spearman"].median()), "spearman_mean": float(sub["spearman"].mean()),
            "spearman_min": float(sub["spearman"].min()), "n_negative_spearman": int((sub["spearman"] < 0).sum()),
            "same_best": int(sub["same_best"].sum()),
            "test_rank_of_val_best_mean": float(sub["test_rank_of_val_best"].mean()),
            "overlap_top3_mean": float(sub["overlap_top3"].mean()),
            "overlap_top5_mean": float(sub["overlap_top5"].mean()),
        })
    agree_summary = pd.DataFrame(agree_summary)
    ctx.w.csv(agree_summary, "item01_ranking_agreement_summary.csv")

    header_1 = (r"Model & \multicolumn{3}{c}{Ranking (validation vs test)} & "
                + " & ".join(rf"\multicolumn{{4}}{{c}}{{{r}}}" for r in cfg.RULES) + r" \\")
    header_2 = (r" & $\rho$ & Same best & Top-5 overlap & "
                + " & ".join([r"Old & New & Med.\,$\Delta$\% & B/E/W"] * len(cfg.RULES)) + r" \\")
    body = []
    for model in MODELS_ALL:
        a = agree_summary[agree_summary["model"] == model].iloc[0]
        cells = [model, f"{a['spearman_median']:.2f}", f"{a['same_best']}/{a['n_scenarios']}",
                 f"{a['overlap_top5_mean']:.2f}"]
        for rule in cfg.RULES:
            r = summary[(summary["model"] == model) & (summary["rule"] == rule)
                        & (summary["metric"] == "MRE")].iloc[0]
            cells += [f"{r['mean_old']:.2f}", f"{r['mean_new']:.2f}", f"{r['change_pct_median']:+.2f}",
                      f"{r['new_better']}/{r['equal']}/{r['new_worse']}"]
        body.append(cells)
    ctx.w.tex(tabular([header_1, header_2], body, "lrrr" + "rrrr" * len(cfg.RULES)), "item01_old_vs_new.tex")

    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    width = 0.26
    for j, rule in enumerate(cfg.RULES):
        data = [detail[(detail["model"] == m) & (detail["rule"] == rule)]["change_pct_MRE"].values
                for m in cfg.MODEL_TYPES]
        pos = np.arange(len(cfg.MODEL_TYPES)) + (j - 1) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=True,
                        flierprops={"markersize": 2, "alpha": 0.5}, medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor(cfg.RULE_COLORS[rule])
            patch.set_alpha(0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(range(len(cfg.MODEL_TYPES)))
    ax.set_xticklabels(cfg.MODEL_TYPES)
    ax.set_ylabel("Change in test MRE (%), new vs old")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cfg.RULE_COLORS[r], alpha=0.75) for r in cfg.RULES],
              labels=cfg.RULES, title="Rule", loc="upper left", ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    ctx.w.fig(fig, "item01_old_vs_new.pdf")


# Item 2: best single against top-k and random-k (k = 2..5) with MEAN and NN
def item02_ablation(ctx):
    cols = cfg.KEYS + ["run"] + [f"test_{m}" for m in cfg.METRICS]
    single = ctx.singles_best[cols].copy()
    single["competitor"] = "single"
    ens = ctx.ensembles[(ctx.ensembles["k"] <= max(cfg.K_RANDOM)) & ctx.ensembles["rule"].isin(["MEAN", "NN"])]
    ens = ens[cols + ["selector", "k", "rule"]].copy()
    ens["competitor"] = ens["selector"] + "-" + ens["k"].astype(str) + " " + ens["rule"]
    both = pd.concat([single, ens[cols + ["competitor"]]], ignore_index=True)

    order = ["single"] + [f"{s}-{k} {r}" for s in ["top", "random"] for r in ["MEAN", "NN"] for k in cfg.K_RANDOM]

    sk = core.run_sk_metrics(both, cfg.KEYS, "competitor", cfg.METRICS)
    ctx.w.csv(sk, "item02_ablation_sk_detail.csv")

    means = both.groupby(["model", "competitor"], as_index=False)[[f"test_{m}" for m in cfg.METRICS]].mean()
    rows = []
    for model in MODELS_ALL:
        sk_m = _of_model(sk, model)
        borda = core.borda_from_sk(sk_m, "competitor").set_index("competitor")
        for comp in order:
            r = sk_m[sk_m["competitor"] == comp]
            row = {"model": model, "competitor": comp,
                   "mean_sk_rank": float(r["sk_rank"].mean()),
                   "borda_total": int(borda.loc[comp, "borda_total"]),
                   "borda_rank": int(borda.loc[comp, "borda_rank"])}
            for m in cfg.METRICS:
                row[f"mean_sk_rank_{m}"] = float(r[r["metric"] == m]["sk_rank"].mean())
                if model == "ALL":
                    row[f"mean_test_{m}"] = float(means[means["competitor"] == comp][f"test_{m}"].mean())
                else:
                    row[f"mean_test_{m}"] = float(means[(means["model"] == model)
                                                        & (means["competitor"] == comp)][f"test_{m}"].iloc[0])
            rows.append(row)
    summary = pd.DataFrame(rows)
    ctx.w.csv(summary, "item02_ablation_summary.csv")

    strategy = summary.copy()
    strategy["strategy"] = strategy["competitor"].map(
        lambda c: "single" if c == "single" else c.split("-")[0] + " " + c.split(" ")[1])
    by_strategy = (strategy.groupby(["model", "strategy"], as_index=False)
                   [["mean_sk_rank"] + [f"mean_sk_rank_{m}" for m in cfg.METRICS]
                    + [f"mean_test_{m}" for m in cfg.METRICS]].mean())
    ctx.w.csv(by_strategy, "item02_ablation_by_strategy.csv")

    rank_mat = summary.pivot(index="competitor", columns="model", values="mean_sk_rank").reindex(
        index=order, columns=MODELS_ALL)
    borda_mat = summary.pivot(index="competitor", columns="model", values="borda_rank").reindex(
        index=order, columns=MODELS_ALL)
    body = []
    for comp in order:
        cells = [comp] + [f"{rank_mat.loc[comp, m]:.2f}" for m in MODELS_ALL]
        cells.append(str(int(borda_mat.loc[comp, "ALL"])))
        body.append(cells)
    header = "Competitor & " + " & ".join(MODELS_ALL) + r" & Borda \\"
    ctx.w.tex(tabular([header], body, "l" + "r" * (len(MODELS_ALL) + 1)), "item02_ablation.tex")

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    im = heatmap(ax, rank_mat.values.astype(float), order, MODELS_ALL, fmt="{:.2f}", cmap="RdYlGn_r", rotate=0)
    fig.colorbar(im, ax=ax, fraction=0.03).set_label("Mean SK rank (4 metrics, 40 scenarios)")
    fig.tight_layout(pad=0.4)
    ctx.w.fig(fig, "item02_ablation.pdf")


# Item 3: diversity of each ensemble against its gain over the best single
def item03_diversity(ctx):
    measures = ["err_corr", "pred_corr", "ambiguity", "rel_ambiguity", "member_mse", "ens_mse"]
    by = cfg.KEYS + ["selector", "k"]
    div = ctx.diversity.groupby(by, as_index=False)[measures].mean()

    cols = [f"test_{m}" for m in cfg.METRICS]
    single = ctx.singles_best.groupby(cfg.KEYS, as_index=False)[cols].mean()
    ens = ctx.ensembles.groupby(by + ["rule"], as_index=False)[cols].mean()
    detail = ens.merge(single, on=cfg.KEYS, suffixes=("", "_single")).merge(div, on=by)
    for m in cfg.METRICS:
        detail[f"gain_pct_{m}"] = ((detail[f"test_{m}_single"] - detail[f"test_{m}"])
                                   / detail[f"test_{m}_single"].abs() * 100)
    ctx.w.csv(detail, "item03_diversity_detail.csv")

    rows = []
    for model in MODELS_ALL:
        for selector in ["top", "random"]:
            for rule in cfg.SELECTOR_RULES[selector]:
                sub = _of_model(detail, model)
                sub = sub[(sub["selector"] == selector) & (sub["rule"] == rule)]
                for measure in ["err_corr", "pred_corr", "rel_ambiguity", "ambiguity"]:
                    for m in cfg.METRICS:
                        ok = sub[[measure, f"gain_pct_{m}"]].dropna()
                        rho, p = spearmanr(ok[measure], ok[f"gain_pct_{m}"]) if len(ok) > 2 else (np.nan, np.nan)
                        rows.append({"model": model, "selector": selector, "rule": rule, "measure": measure,
                                     "gain_metric": m, "n": len(ok), "spearman_rho": float(rho), "p_value": float(p)})
    spearman = pd.DataFrame(rows)
    ctx.w.csv(spearman, "item03_diversity_spearman.csv")

    top = div[div["selector"] == "top"]
    level_rows = []
    for model in MODELS_ALL:
        sub = _of_model(top, model)
        level_rows.append({"model": model,
                           "err_corr_mean": float(sub["err_corr"].mean()),
                           "err_corr_median": float(sub["err_corr"].median()),
                           "pred_corr_mean": float(sub["pred_corr"].mean()),
                           "rel_ambiguity_mean": float(sub["rel_ambiguity"].mean()),
                           "rel_ambiguity_median": float(sub["rel_ambiguity"].median())})
    levels = pd.DataFrame(level_rows)
    ctx.w.csv(levels, "item03_diversity_levels.csv")

    def rho_of(model, rule, measure):
        r = spearman[(spearman["model"] == model) & (spearman["selector"] == "top") & (spearman["rule"] == rule)
                     & (spearman["measure"] == measure) & (spearman["gain_metric"] == "MRE")].iloc[0]
        return f"{r['spearman_rho']:+.2f}"

    body = []
    for model in MODELS_ALL:
        lv = levels[levels["model"] == model].iloc[0]
        body.append([model, f"{lv['err_corr_mean']:.3f}", f"{lv['pred_corr_mean']:.3f}",
                     f"{lv['rel_ambiguity_mean']:.3f}"]
                    + [rho_of(model, rule, "err_corr") for rule in cfg.RULES]
                    + [rho_of(model, "MEAN", "pred_corr"), rho_of(model, "MEAN", "rel_ambiguity")])
    header_1 = (r"Model & \multicolumn{3}{c}{Mean over top-$k$ ensembles} & "
                r"\multicolumn{3}{c}{$\rho$(error corr., gain)} & \multicolumn{2}{c}{$\rho$(., gain), MEAN} \\")
    header_2 = (r" & Error corr. & Pred. corr. & Rel. ambiguity & MEAN & IRWM & NN & Pred. corr. & Rel. ambiguity \\")
    ctx.w.tex(tabular([header_1, header_2], body, "lrrrrrrrr"), "item03_diversity.tex")

    points = detail[(detail["selector"] == "top") & (detail["rule"] == "MEAN")]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0), squeeze=False)
    for idx, model in enumerate(cfg.MODEL_TYPES):
        ax = axes[idx // 4][idx % 4]
        sub = points[points["model"] == model]
        sc = ax.scatter(sub["err_corr"], sub["gain_pct_MRE"], c=sub["k"], cmap="viridis", s=12, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_yscale("symlog", linthresh=1.0)
        r = spearman[(spearman["model"] == model) & (spearman["selector"] == "top") & (spearman["rule"] == "MEAN")
                     & (spearman["measure"] == "err_corr") & (spearman["gain_metric"] == "MRE")].iloc[0]
        ax.set_title(f"{model} ($\\rho$ = {r['spearman_rho']:+.2f}, n = {int(r['n'])})")
        ax.grid(True, alpha=0.2)
        if idx // 4 == 1:
            ax.set_xlabel("Mean pairwise error correlation")
        if idx % 4 == 0:
            ax.set_ylabel("Gain in MRE over best single (%)")
    fig.tight_layout()
    fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01).set_label("$k$")
    ctx.w.fig(fig, "item03_diversity_vs_gain.pdf")


# Item 4: size of the improvement of the best ensemble over the best single
def item04_improvement(ctx):
    metrics = cfg.ALL_METRICS
    summary = paper.wtl(ctx, tuple(metrics)).copy()

    rows = []
    s_groups = {k: g for k, g in ctx.singles_best.groupby(cfg.KEYS)}
    for key, e in ctx.ens_best.groupby(cfg.KEYS):
        s = s_groups[key]
        for m in metrics:
            value = core.a12_lower(e[f"test_{m}"].to_numpy(), s[f"test_{m}"].to_numpy())
            rows.append(dict(zip(cfg.KEYS, key), metric=m, a12=value, magnitude=core.a12_magnitude(value),
                             favours="ensemble" if value > 0.5 else "single" if value < 0.5 else "none"))
    a12 = pd.DataFrame(rows)
    ctx.w.csv(a12, "item04_a12_per_scenario.csv")

    extra = []
    for (model, metric), g in a12.groupby(["model", "metric"]):
        row = {"model": model, "metric": metric, "a12_mean": float(g["a12"].mean()),
               "a12_median": float(g["a12"].median())}
        for side in ["ensemble", "single"]:
            for mag in ["large", "medium", "small"]:
                row[f"n_{mag}_{side}"] = int(((g["favours"] == side) & (g["magnitude"] == mag)).sum())
        row["n_negligible"] = int((g["magnitude"] == "negligible").sum())
        extra.append(row)
    summary = summary.merge(pd.DataFrame(extra), on=["model", "metric"])
    ctx.w.csv(summary, "item04_improvement.csv")

    for name, group in [("item04_improvement.tex", cfg.METRICS),
                        ("item04_improvement_median_metrics.tex", cfg.MEDIAN_METRICS)]:
        header_1 = "Model & " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{m}}}" for m in group) + r" \\"
        header_2 = " & " + " & ".join([r"$\Delta$ mean & $\Delta$ med. & \% mean & \% med. & $A_{12}$"] * len(group)) + r" \\"
        body = []
        for model in cfg.MODEL_TYPES:
            cells = [model]
            for m in group:
                r = summary[(summary["model"] == model) & (summary["metric"] == m)].iloc[0]
                cells += [f"{r['abs_improvement_mean']:.3g}", f"{r['abs_improvement_median']:.3g}",
                          f"{r['rel_improvement_mean_pct']:+.1f}", f"{r['rel_improvement_median_pct']:+.1f}",
                          f"{r['a12_median']:.2f}"]
            body.append(cells)
        ctx.w.tex(tabular([header_1, header_2], body, "l" + "rrrrr" * len(group)), name)

    mat = summary.pivot(index="model", columns="metric", values="a12_median").reindex(
        index=cfg.MODEL_TYPES, columns=metrics)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    im = heatmap(ax, mat.values.astype(float), cfg.MODEL_TYPES, metrics, fmt="{:.2f}", cmap="RdYlGn",
                 vmin=0.2, vmax=0.8, rotate=0)
    fig.colorbar(im, ax=ax, fraction=0.03).set_label("Median $A_{12}$ (> 0.5: ensemble lower error)")
    fig.tight_layout(pad=0.4)
    ctx.w.fig(fig, "item04_a12.pdf")


def _rule_sk(ctx):
    if "rule_head_to_head" not in ctx.memo:
        top = ctx.ensembles[ctx.ensembles["selector"] == "top"]
        ctx.memo["rule_head_to_head"] = core.run_sk_metrics(top, cfg.KEYS + ["k"], "rule", cfg.METRICS)
    return ctx.memo["rule_head_to_head"]


# Item 6: MEAN, IRWM and NN compared on the same top-k members
def item06_rules(ctx):
    sk = _rule_sk(ctx)
    ctx.w.csv(sk, "item06_rules_sk_detail.csv")
    pairs = [("MEAN", "IRWM"), ("MEAN", "NN"), ("IRWM", "NN")]

    sk_rows, wtl_rows = [], []
    for scope, sk_scope in [("all", sk), ("S1", core.s1_only(sk))]:
        for model in MODELS_ALL:
            sub = _of_model(sk_scope, model)
            borda = core.borda_from_sk(sub, "rule").set_index("rule")
            for rule in cfg.RULES:
                r = sub[sub["rule"] == rule]
                row = {"scope": scope, "model": model, "rule": rule,
                       "mean_sk_rank": float(r["sk_rank"].mean()),
                       "borda_total": int(borda.loc[rule, "borda_total"]),
                       "borda_rank": int(borda.loc[rule, "borda_rank"])}
                for m in cfg.METRICS:
                    row[f"mean_sk_rank_{m}"] = float(r[r["metric"] == m]["sk_rank"].mean())
                sk_rows.append(row)
            for m in cfg.METRICS:
                wide = sub[sub["metric"] == m].pivot_table(index=cfg.KEYS + ["k"], columns="rule", values="sk_rank")
                for a, b in pairs:
                    wtl_rows.append({"scope": scope, "model": model, "metric": m, "rule_a": a, "rule_b": b,
                                     "a_wins": int((wide[a] < wide[b]).sum()), "ties": int((wide[a] == wide[b]).sum()),
                                     "b_wins": int((wide[a] > wide[b]).sum()), "n": len(wide)})
    sk_summary, wtl = pd.DataFrame(sk_rows), pd.DataFrame(wtl_rows)
    ctx.w.csv(sk_summary, "item06_rules_sk_summary.csv")
    ctx.w.csv(wtl, "item06_rules_wtl.csv")

    for scope in ["all", "S1"]:
        header_1 = (r"Model & \multicolumn{3}{c}{Mean SK rank (MRE)} & "
                    + " & ".join(rf"\multicolumn{{3}}{{c}}{{{a} vs {b}}}" for a, b in pairs) + r" \\")
        header_2 = " & MEAN & IRWM & NN & " + " & ".join(["W & T & L"] * len(pairs)) + r" \\"
        body = []
        for model in MODELS_ALL:
            cells = [model]
            for rule in cfg.RULES:
                r = sk_summary[(sk_summary["scope"] == scope) & (sk_summary["model"] == model)
                               & (sk_summary["rule"] == rule)].iloc[0]
                cells.append(f"{r['mean_sk_rank_MRE']:.2f}")
            for a, b in pairs:
                r = wtl[(wtl["scope"] == scope) & (wtl["model"] == model) & (wtl["metric"] == "MRE")
                        & (wtl["rule_a"] == a) & (wtl["rule_b"] == b)].iloc[0]
                cells += [int(r["a_wins"]), int(r["ties"]), int(r["b_wins"])]
            body.append(cells)
        ctx.w.tex(tabular([header_1, header_2], body, "lrrr" + "rrr" * len(pairs)),
                  f"item06_rules_{scope.lower()}.tex")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 2.4))
    for ax, scope in zip(axes, ["all", "S1"]):
        mat = (sk_summary[sk_summary["scope"] == scope]
               .pivot(index="rule", columns="model", values="mean_sk_rank_MRE")
               .reindex(index=cfg.RULES, columns=MODELS_ALL))
        im = heatmap(ax, mat.values.astype(float), cfg.RULES, MODELS_ALL, fmt="{:.2f}", cmap="RdYlGn_r",
                     vmin=1.0, vmax=max(2.0, float(np.nanmax(mat.values))), rotate=0)
        ax.set_title("All sizes" if scope == "all" else "S1")
        fig.colorbar(im, ax=ax, fraction=0.03).set_label("Mean SK rank (MRE)")
    fig.tight_layout(pad=0.4)
    ctx.w.fig(fig, "item06_rules.pdf")


# Item 8: spread across the 30 runs at the smallest training size
def item08_s1_stability(ctx):
    rows = []
    for kind, df in [("single", core.s1_only(ctx.singles_best)), ("ensemble", core.s1_only(ctx.ens_best))]:
        for (model, dataset, size), g in df.groupby(["model", "dataset", "sample_size"]):
            for m in cfg.ALL_METRICS:
                v = g[f"test_{m}"].to_numpy()
                mean = float(v.mean())
                rows.append({"kind": kind, "model": model, "dataset": dataset, "sample_size": size, "metric": m,
                             "mean": mean, "std": float(v.std(ddof=1)),
                             "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                             "cv": float(v.std(ddof=1) / abs(mean)) if mean else np.nan,
                             "min": float(v.min()), "max": float(v.max())})
    stats = pd.DataFrame(rows)
    ctx.w.csv(stats, "item08_s1_stability.csv")

    datasets = sorted(stats["dataset"].unique())
    mre = stats[stats["metric"] == "MRE"]
    for kind in ["single", "ensemble"]:
        sub = mre[mre["kind"] == kind]
        body = []
        for model in cfg.MODEL_TYPES:
            cells = [model]
            for ds in datasets:
                r = sub[(sub["model"] == model) & (sub["dataset"] == ds)].iloc[0]
                cells += [f"{r['std']:.2f}", f"{r['iqr']:.2f}"]
            body.append(cells)
        header_1 = "Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{ds_label(d)}}}" for d in datasets) + r" \\"
        header_2 = " & " + " & ".join(["Std & IQR"] * len(datasets)) + r" \\"
        ctx.w.tex(tabular([header_1, header_2], body, "l" + "rr" * len(datasets)), f"item08_s1_stability_{kind}.tex")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 6.2))
    for i, stat in enumerate(["std", "iqr"]):
        for j, kind in enumerate(["single", "ensemble"]):
            mat = (mre[mre["kind"] == kind].pivot(index="model", columns="dataset", values=stat)
                   .reindex(index=cfg.MODEL_TYPES, columns=datasets))
            values = mat.values.astype(float)
            im = heatmap(axes[i][j], values, cfg.MODEL_TYPES, [ds_label(d) for d in datasets], fmt="{:.1f}",
                         vmin=0, vmax=float(np.nanpercentile(values, 90)))
            axes[i][j].set_title(f"{'Std' if stat == 'std' else 'IQR'} of MRE over 30 runs at S1, best {kind}")
            fig.colorbar(im, ax=axes[i][j], fraction=0.03)
    fig.tight_layout(pad=0.4)
    ctx.w.fig(fig, "item08_s1_stability.pdf")


# Item 9: share of negative test predictions of the single variants
def item09_negative_predictions(ctx):
    v = ctx.variants.copy()
    v["has_negative"] = v["n_negative"] > 0
    all_variants = v.groupby(cfg.KEYS).agg(
        n_predictions=("n_test", "sum"), n_negative=("n_negative", "sum"),
        pct_variant_runs_with_negative=("has_negative", lambda s: s.mean() * 100)).reset_index()
    all_variants["pct_negative_all_variants"] = all_variants["n_negative"] / all_variants["n_predictions"] * 100

    b = ctx.singles_best.copy()
    b["has_negative"] = b["n_negative"] > 0
    best = b.groupby(cfg.KEYS).agg(
        n_predictions_best=("n_test", "sum"), n_negative_best=("n_negative", "sum"),
        pct_runs_with_negative_best=("has_negative", lambda s: s.mean() * 100)).reset_index()
    best["pct_negative_best_single"] = best["n_negative_best"] / best["n_predictions_best"] * 100

    detail = all_variants.merge(best, on=cfg.KEYS)
    detail["size_index"] = core.size_index(detail)
    ctx.w.csv(detail, "item09_negative_predictions.csv")

    by_system = detail.groupby(["model", "dataset"], as_index=False).agg(
        n_predictions=("n_predictions", "sum"), n_negative=("n_negative", "sum"),
        n_predictions_best=("n_predictions_best", "sum"), n_negative_best=("n_negative_best", "sum"))
    by_system["pct_negative_all_variants"] = by_system["n_negative"] / by_system["n_predictions"] * 100
    by_system["pct_negative_best_single"] = by_system["n_negative_best"] / by_system["n_predictions_best"] * 100
    ctx.w.csv(by_system, "item09_negative_predictions_by_system.csv")

    datasets = sorted(detail["dataset"].unique())
    body = []
    for model in cfg.MODEL_TYPES:
        cells = [model]
        for ds in datasets:
            r = by_system[(by_system["model"] == model) & (by_system["dataset"] == ds)].iloc[0]
            cells.append(f"{r['pct_negative_all_variants']:.2f}")
        sub = by_system[by_system["model"] == model]
        cells.append(f"{sub['n_negative'].sum() / sub['n_predictions'].sum() * 100:.2f}")
        body.append(cells)
    header = "Model & " + " & ".join(ds_label(d) for d in datasets) + r" & All \\"
    ctx.w.tex(tabular([header], body, "l" + "r" * (len(datasets) + 1)), "item09_negative_predictions.tex")

    detail["scenario"] = detail["dataset"].map(ds_label) + " S" + detail["size_index"].astype(str)
    scenarios = (detail.sort_values(["dataset", "size_index"])["scenario"].drop_duplicates().tolist())
    fig, axes = plt.subplots(2, 1, figsize=(15.0, 6.4))
    for ax, col, title in [(axes[0], "pct_negative_all_variants", "all 10 variants"),
                           (axes[1], "pct_negative_best_single", "best single")]:
        mat = detail.pivot(index="model", columns="scenario", values=col).reindex(
            index=cfg.MODEL_TYPES, columns=scenarios)
        values = mat.values.astype(float)
        im = heatmap(ax, values, cfg.MODEL_TYPES, scenarios, fmt="{:.1f}", fontsize=5.5, rotate=60,
                     vmin=0, vmax=max(1.0, float(np.nanpercentile(values, 99))))
        ax.set_title(f"Negative test predictions (%), {title}")
        fig.colorbar(im, ax=ax, fraction=0.015)
    fig.tight_layout(pad=0.4)
    ctx.w.fig(fig, "item09_negative_predictions.pdf")


def run(ctx):
    item01_old_vs_new(ctx)
    item02_ablation(ctx)
    item03_diversity(ctx)
    item04_improvement(ctx)
    item06_rules(ctx)
    item08_s1_stability(ctx)
    item09_negative_predictions(ctx)
