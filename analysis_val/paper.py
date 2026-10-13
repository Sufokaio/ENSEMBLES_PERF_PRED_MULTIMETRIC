# Tables 6-8 (and their median-metric versions) and Figures 6-13 of the paper.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config as cfg
import core
from out import bold, ds_label, heatmap, tabular

SCEN = core.SCEN


def _rank_table_rows(df, sk, metrics, competitor_col="model"):
    rows = []
    for model in cfg.MODEL_TYPES:
        sub = df[df[competitor_col] == model]
        for m in metrics:
            per_scenario = sub.groupby(SCEN)[f"test_{m}"].mean()
            ranks = sk[(sk[competitor_col] == model) & (sk["metric"] == m)]["sk_rank"]
            rows.append({
                "model": model, "metric": m,
                "mean": float(per_scenario.mean()),
                "median_scenarios": float(per_scenario.median()),
                "std_runs": float(sub[f"test_{m}"].std(ddof=1)),
                "std_scenarios": float(per_scenario.std(ddof=1)),
                "mean_sk_rank": float(ranks.mean()),
            })
    return pd.DataFrame(rows)


def _rank_table_tex(stats, metrics, extra, extra_headers):
    best = stats.groupby("metric")["mean"].min().to_dict()
    header_1 = "Model & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{m}}}" for m in metrics)
    header_2 = " & " + " & ".join(["Mean & Median & SK"] * len(metrics))
    if extra_headers:
        header_1 += " & " + " & ".join(extra_headers)
        header_2 += " & " * len(extra_headers)
    body = []
    for model in cfg.MODEL_TYPES:
        cells = [model]
        for m in metrics:
            r = stats[(stats["model"] == model) & (stats["metric"] == m)].iloc[0]
            mean = f"{r['mean']:.2f}"
            if abs(r["mean"] - best[m]) < 1e-12:
                mean = bold(mean)
            cells += [mean, f"{r['median_scenarios']:.2f}", f"{r['mean_sk_rank']:.1f}"]
        cells += extra.get(model, [])
        body.append(cells)
    col_spec = "l" + "rrr" * len(metrics) + "r" * len(extra_headers)
    return tabular([header_1 + r" \\", header_2 + r" \\"], body, col_spec)


def _sa_d(df, competitor_col="model"):
    out = {}
    for model in cfg.MODEL_TYPES:
        sub = df[df[competitor_col] == model]
        out[model] = (float(sub["test_SA"].mean()), float(sub["test_D"].mean()))
    return out


def singles_sk(ctx, metrics):
    key = ("singles_sk", tuple(metrics))
    if key not in ctx.memo:
        ctx.memo[key] = core.run_sk_metrics(ctx.singles_best, SCEN, "model", metrics)
    return ctx.memo[key]


def ensembles_sk(ctx, metrics):
    key = ("ens_sk", tuple(metrics))
    if key not in ctx.memo:
        ctx.memo[key] = core.run_sk_metrics(ctx.ens_best, SCEN, "model", metrics)
    return ctx.memo[key]


# Singles and best ensembles ranked together: 16 competitors per scenario
def mixed_sk(ctx, metrics):
    key = ("mixed_sk", tuple(metrics))
    if key not in ctx.memo:
        cols = SCEN + ["model", "run"] + [f"test_{m}" for m in metrics]
        s = ctx.singles_best[cols].copy()
        s["kind"] = "single"
        e = ctx.ens_best[cols].copy()
        e["kind"] = "ensemble"
        both = pd.concat([s, e], ignore_index=True)
        both["competitor"] = both["model"] + np.where(both["kind"] == "single", "_S", "_E")
        sk = core.run_sk_metrics(both, SCEN, "competitor", metrics)
        sk["model"] = sk["competitor"].str[:-2]
        sk["kind"] = np.where(sk["competitor"].str.endswith("_S"), "single", "ensemble")
        ctx.memo[key] = sk
    return ctx.memo[key]


def table6(ctx, metrics=None, name="table6_singles", with_sa=True):
    metrics = metrics or cfg.METRICS
    sk = singles_sk(ctx, metrics)
    stats = _rank_table_rows(ctx.singles_best, sk, metrics)
    borda = core.borda_from_sk(sk, "model").set_index("model")
    stats["borda_total"] = stats["model"].map(borda["borda_total"])
    stats["borda_rank"] = stats["model"].map(borda["borda_rank"])
    extra, headers = {m: [] for m in cfg.MODEL_TYPES}, []
    if with_sa:
        sa = _sa_d(ctx.singles_best)
        stats["mean_SA"] = stats["model"].map(lambda m: sa[m][0])
        stats["mean_D"] = stats["model"].map(lambda m: sa[m][1])
        best_sa = max(v[0] for v in sa.values())
        for m in cfg.MODEL_TYPES:
            cell = f"{sa[m][0]:.3f}"
            extra[m] += [bold(cell) if sa[m][0] == best_sa else cell, f"{sa[m][1]:.2f}"]
        headers += ["SA", "D"]
    for m in cfg.MODEL_TYPES:
        rank = int(borda.loc[m, "borda_rank"])
        extra[m].append(bold(rank) if rank == 1 else str(rank))
    headers.append("Borda")
    ctx.w.csv(stats, f"{name}.csv")
    ctx.w.tex(_rank_table_tex(stats, metrics, extra, headers), f"{name}.tex")
    return stats


def table8(ctx, metrics=None, name="table8_ensembles", with_sa=True):
    metrics = metrics or cfg.METRICS
    sk = ensembles_sk(ctx, metrics)
    stats = _rank_table_rows(ctx.ens_best, sk, metrics)
    borda = core.borda_from_sk(sk, "model").set_index("model")
    mixed = core.borda_from_sk(mixed_sk(ctx, metrics), "competitor").set_index("competitor")
    stats["borda_rank_ensembles"] = stats["model"].map(borda["borda_rank"])
    stats["borda_rank_mixed"] = stats["model"].map(lambda m: int(mixed.loc[f"{m}_E", "borda_rank"]))
    stats["borda_rank_mixed_single"] = stats["model"].map(lambda m: int(mixed.loc[f"{m}_S", "borda_rank"]))
    extra, headers = {m: [] for m in cfg.MODEL_TYPES}, []
    if with_sa:
        sa = _sa_d(ctx.ens_best)
        stats["mean_SA"] = stats["model"].map(lambda m: sa[m][0])
        stats["mean_D"] = stats["model"].map(lambda m: sa[m][1])
        best_sa = max(v[0] for v in sa.values())
        for m in cfg.MODEL_TYPES:
            cell = f"{sa[m][0]:.3f}"
            extra[m] += [bold(cell) if sa[m][0] == best_sa else cell, f"{sa[m][1]:.2f}"]
        headers += ["SA", "D"]
    for m in cfg.MODEL_TYPES:
        extra[m] += [str(int(borda.loc[m, "borda_rank"])), str(int(mixed.loc[f"{m}_E", "borda_rank"]))]
    headers += ["Borda (8)", "Borda (16)"]
    ctx.w.csv(stats, f"{name}.csv")
    ctx.w.tex(_rank_table_tex(stats, metrics, extra, headers), f"{name}.tex")
    return stats


# Best ensemble against best single on the scenario means: wins, ties, losses and improvement
def wtl(ctx, metrics):
    key = ("wtl", tuple(metrics))
    if key in ctx.memo:
        return ctx.memo[key]
    cols = [f"test_{m}" for m in metrics]
    s = ctx.singles_best.groupby(cfg.KEYS, as_index=False)[cols].mean()
    e = ctx.ens_best.groupby(cfg.KEYS, as_index=False)[cols].mean()
    merged = s.merge(e, on=cfg.KEYS, suffixes=("_single", "_ens"))
    rows = []
    for model in cfg.MODEL_TYPES:
        sub = merged[merged["model"] == model]
        for m in metrics:
            sv, ev = sub[f"test_{m}_single"], sub[f"test_{m}_ens"]
            if m == "SA":
                wins, losses = int((ev > sv).sum()), int((ev < sv).sum())
                imp = (ev - sv) / sv.abs() * 100
                diff = ev - sv
            else:
                wins, losses = int((ev < sv).sum()), int((ev > sv).sum())
                imp = (sv - ev) / sv.abs() * 100
                diff = sv - ev
            n = len(sub)
            rate, lo, hi = core.wilson_ci(wins, n)
            rows.append({
                "model": model, "metric": m, "W": wins, "T": n - wins - losses, "L": losses, "N": n,
                "win_rate": rate, "win_rate_lo": lo, "win_rate_hi": hi,
                "abs_improvement_mean": float(diff.mean()), "abs_improvement_median": float(diff.median()),
                "rel_improvement_mean_pct": float(imp.mean()), "rel_improvement_median_pct": float(imp.median()),
            })
    ctx.memo[key] = pd.DataFrame(rows)
    return ctx.memo[key]


def table7(ctx):
    metrics = cfg.METRICS + ["SA"]
    df = wtl(ctx, metrics)
    ctx.w.csv(df, "table7_single_vs_ensemble.csv")

    body = []
    for model in cfg.MODEL_TYPES:
        cells = [model]
        for m in metrics:
            r = df[(df["model"] == model) & (df["metric"] == m)].iloc[0]
            cells.append(f"{r['win_rate'] * 100:.0f}\\% [{r['win_rate_lo'] * 100:.0f}--{r['win_rate_hi'] * 100:.0f}\\%] "
                         f"({r['rel_improvement_mean_pct']:+.1f}\\%)")
        body.append(cells)
    ctx.w.tex(tabular(["Model & " + " & ".join(metrics) + r" \\"], body, "l" + "c" * len(metrics)),
              "table7_winrate.tex")

    header_1 = "Model & " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{m}}}" for m in metrics) + r" \\"
    header_2 = " & " + " & ".join([r"W & T & L & mean\% & med.\%"] * len(metrics)) + r" \\"
    body = []
    for model in cfg.MODEL_TYPES:
        cells = [model]
        for m in metrics:
            r = df[(df["model"] == model) & (df["metric"] == m)].iloc[0]
            cells += [int(r["W"]), int(r["T"]), int(r["L"]),
                      f"{r['rel_improvement_mean_pct']:+.1f}", f"{r['rel_improvement_median_pct']:+.1f}"]
        body.append(cells)
    ctx.w.tex(tabular([header_1, header_2], body, "l" + "rrrrr" * len(metrics)), "table7_wtl.tex")


# Chosen variant / ensemble per learner and scenario
def selection_report(ctx):
    s = ctx.singles_best.groupby(cfg.KEYS + ["variant", "val_borda_rank", "test_borda_rank"],
                                 as_index=False)[["test_MRE", "val_MRE"]].mean()
    s = s.rename(columns={"variant": "single_variant", "val_borda_rank": "single_val_borda_rank",
                          "test_borda_rank": "single_test_borda_rank",
                          "test_MRE": "single_test_MRE", "val_MRE": "single_val_MRE"})
    e = ctx.ens_best.groupby(cfg.KEYS + ["selector", "k", "rule"], as_index=False)[["test_MRE", "val_MRE"]].mean()
    e = e.rename(columns={"selector": "ens_selector", "k": "ens_k", "rule": "ens_rule",
                          "test_MRE": "ens_test_MRE", "val_MRE": "ens_val_MRE"})
    report = s.merge(e, on=cfg.KEYS)
    ctx.w.csv(report, "selected_single_and_ensemble.csv")

    counts = (report.groupby(["model", "ens_selector", "ens_rule"]).size()
              .rename("n_scenarios").reset_index())
    ctx.w.csv(counts, "selected_ensemble_counts.csv")
    k_counts = report.groupby(["model", "ens_k"]).size().rename("n_scenarios").reset_index()
    ctx.w.csv(k_counts, "selected_ensemble_k_counts.csv")
    random_k_side_note(ctx)
    return report


# Side note: scenarios where a random-k ensemble has a better validation Borda than every top-k ensemble
def random_k_side_note(ctx):
    cols = [f"val_{m}" for m in cfg.METRICS] + ["test_MRE"]
    ids = ["selector", "k", "rule"]
    means = ctx.ensembles.groupby(cfg.KEYS + ids, as_index=False)[cols].mean()
    ranked = core.val_borda(means, cfg.KEYS, ids)
    first = ranked[ranked["val_borda_rank"] == 1]
    best_top = (ranked[ranked["selector"] == "top"].sort_values(cfg.KEYS + ["val_borda_rank"])
                .groupby(cfg.KEYS, as_index=False).first())
    detail = first[cfg.KEYS + ids + ["test_MRE"]].merge(
        best_top[cfg.KEYS + ids + ["test_MRE"]], on=cfg.KEYS, suffixes=("_all35", "_top27"))
    detail["random_k_first"] = detail["selector_all35"] == "random"
    ctx.w.csv(detail, "side_note_random_k_detail.csv")
    rows = []
    for model in cfg.MODEL_TYPES + ["ALL"]:
        sub = detail if model == "ALL" else detail[detail["model"] == model]
        won = sub[sub["random_k_first"]]
        rows.append({
            "model": model, "n_scenarios": len(sub), "n_random_k_first_on_validation": len(won),
            "of_which_random_k_lower_test_MRE": int((won["test_MRE_all35"] < won["test_MRE_top27"]).sum()),
            "of_which_random_k_higher_test_MRE": int((won["test_MRE_all35"] > won["test_MRE_top27"]).sum()),
        })
    ctx.w.csv(pd.DataFrame(rows), "side_note_random_k.csv")


def _two_panel(values, ranks, row_labels, col_labels, value_label, rotate=38, fmt="{:.2f}", rank_fmt="{:.1f}"):
    n_rows, n_cols = values.shape
    fig, axes = plt.subplots(1, 2, figsize=(max(11.0, n_cols * 1.7), max(2.6, n_rows * 0.42)))
    finite = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(finite, 95)) if len(finite) else 1.0
    im = heatmap(axes[0], values, row_labels, col_labels, fmt=fmt, vmin=0, vmax=vmax, rotate=rotate)
    fig.colorbar(im, ax=axes[0], fraction=0.03).set_label(value_label)
    axes[0].set_title(f"(a) {value_label}")
    im = heatmap(axes[1], ranks, row_labels, col_labels, fmt=rank_fmt, cmap="RdYlGn_r", rotate=rotate)
    fig.colorbar(im, ax=axes[1], fraction=0.03).set_label("Mean SK rank")
    axes[1].set_title("(b) Mean SK rank")
    fig.tight_layout(pad=0.4)
    return fig


def fig_singles_mre(ctx):
    sk = singles_sk(ctx, cfg.METRICS)
    sk = sk[sk["metric"] == "MRE"]
    datasets = sorted(ctx.singles_best["dataset"].unique())
    for number, scope, df, sk_scope in [("07", "all", ctx.singles_best, sk),
                                        ("08", "s1", core.s1_only(ctx.singles_best), core.s1_only(sk))]:
        mre = df.groupby(["model", "dataset"])["test_MRE"].mean().unstack("dataset")
        rank = sk_scope.groupby(["model", "dataset"])["sk_rank"].mean().unstack("dataset")
        mre, rank = mre.reindex(cfg.MODEL_TYPES)[datasets], rank.reindex(cfg.MODEL_TYPES)[datasets]
        table = (mre.stack().rename("mean_MRE").reset_index()
                 .merge(rank.stack().rename("mean_sk_rank").reset_index(), on=["model", "dataset"]))
        ctx.w.csv(table, f"fig{number}_singles_mre_{scope}.csv")
        fig = _two_panel(mre.values, rank.values, cfg.MODEL_TYPES, [ds_label(d) for d in datasets], "Mean MRE")
        ctx.w.fig(fig, f"fig{number}_singles_mre_{scope}.pdf")


def fig_singles_rank_s1(ctx):
    sk = core.s1_only(singles_sk(ctx, cfg.METRICS))
    borda = core.borda_from_sk(sk, "model", by=["dataset"])
    datasets = sorted(borda["dataset"].unique())
    pivot = borda.pivot(index="dataset", columns="model", values="borda_rank").reindex(
        index=datasets, columns=cfg.MODEL_TYPES)
    ctx.w.csv(borda, "fig09_singles_rank_s1.csv")
    n = len(cfg.MODEL_TYPES)
    fig, ax = plt.subplots(figsize=(max(5.5, n * 0.75), max(2.6, len(datasets) * 0.42)))
    im = heatmap(ax, pivot.values.astype(float), [ds_label(d) for d in datasets], cfg.MODEL_TYPES,
                 fmt="{:.0f}", cmap=plt.get_cmap("YlOrRd", n), vmin=0.5, vmax=n + 0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Borda rank")
    cbar.set_ticks(range(1, n + 1))
    fig.tight_layout(pad=0.4)
    ctx.w.fig(fig, "fig09_singles_rank_s1.pdf")


def fig_d_forest(ctx):
    rows = []
    for model in cfg.MODEL_TYPES:
        row = {"model": model}
        for label, df in [("single", ctx.singles_best), ("ens", ctx.ens_best)]:
            per_scenario = df[df["model"] == model].groupby(SCEN)["test_D"].mean()
            row[f"D_{label}"] = float(per_scenario.mean())
            row[f"D_{label}_sd"] = float(per_scenario.std(ddof=1))
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("D_single").reset_index(drop=True)
    ctx.w.csv(res, "fig06_d_forest.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    off = 0.15
    for i, row in res.iterrows():
        color = cfg.MODEL_COLORS[row["model"]]
        ax.plot([row["D_single"], row["D_ens"]], [i + off, i - off], color=color, linewidth=1.1, alpha=0.65, zorder=1)
        ax.errorbar(row["D_single"], i + off, xerr=row["D_single_sd"], fmt="none", color=color,
                    capsize=2, linewidth=0.7, alpha=0.5, zorder=2)
        ax.scatter(row["D_single"], i + off, marker="o", s=48, facecolors="none", edgecolors=color,
                   linewidth=1.5, zorder=3)
        ax.errorbar(row["D_ens"], i - off, xerr=row["D_ens_sd"], fmt="none", color=color,
                    capsize=2, linewidth=0.7, alpha=0.5, zorder=2)
        ax.scatter(row["D_ens"], i - off, marker="o", s=48, color=color, zorder=3)
    ax.axvline(0, color="red", linewidth=1.0)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res["model"].tolist())
    ax.set_xlabel(r"$\Delta$")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="none", markersize=7, linestyle="none",
               label="Best single"),
        Line2D([0], [0], marker="o", color="gray", markersize=7, linestyle="none", label="Best ensemble"),
        Line2D([0], [0], color="red", linewidth=1.0, label=r"$\Delta$ = 0"),
    ], fontsize=7, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    ctx.w.fig(fig, "fig06_d_forest.pdf")


def fig_gap_close(ctx):
    sk = mixed_sk(ctx, cfg.METRICS)
    sk = sk[sk["metric"] == "MRE"]
    datasets = sorted(sk["dataset"].unique())
    mean_rank = sk.groupby(["model", "kind", "dataset"])["sk_rank"].mean().unstack("kind")
    delta = (mean_rank["ensemble"] - mean_rank["single"]).unstack("dataset").reindex(cfg.MODEL_TYPES)[datasets]
    ctx.w.csv(mean_rank.reset_index().assign(delta=lambda d: d["ensemble"] - d["single"]), "fig10_gap_close.csv")
    mat = delta.values.astype(float)
    vmax = max(float(np.nanmax(np.abs(mat))), 0.5)
    fig, ax = plt.subplots(figsize=(max(6.0, len(datasets) * 0.82), max(2.4, len(cfg.MODEL_TYPES) * 0.38)))
    im = heatmap(ax, mat, cfg.MODEL_TYPES, [ds_label(d) for d in datasets], fmt="{:+.1f}",
                 cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04).set_label(r"$\Delta$ SK rank")
    fig.tight_layout(pad=0.4)
    ctx.w.fig(fig, "fig10_gap_close.pdf")


def _k_curve_axes(axes, mean_sk, ks):
    for col, rule in enumerate(cfg.RULES):
        ax = axes[col]
        rule_data = mean_sk[mean_sk["rule"] == rule]
        if not rule_data.empty:
            ax.axhline(rule_data["sk_rank"].min(), color="#888888", linewidth=0.9, linestyle="--", zorder=0)
        for model in cfg.MODEL_TYPES:
            d = rule_data[rule_data["model"] == model].sort_values("k")
            ax.plot(d["k"], d["sk_rank"], color=cfg.MODEL_COLORS[model], marker=cfg.BASE_MARKERS[model],
                    markersize=5, linewidth=1.5, label=model)
        ax.set_xticks(ks)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)


# Figure 11: mean SK rank against k per rule (72 competitors = 8 learners x 9 k per scenario and rule)
def fig_k_curve(ctx):
    sk = ctx.k_sk
    ks = sorted(sk["k"].unique())
    mean_sk = sk.groupby(["model", "rule", "k"], as_index=False)["sk_rank"].mean()
    ctx.w.csv(mean_sk, "fig11_k_sk_rank_curve.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    _k_curve_axes(axes, mean_sk, ks)
    for col, rule in enumerate(cfg.RULES):
        axes[col].set_title(rule, fontsize=11, fontweight="bold")
        axes[col].set_xlabel("$k$")
    axes[0].set_ylabel("Mean SK rank")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="center left", bbox_to_anchor=(1.04, 0.5), title="Base type", ncol=2)
    fig.tight_layout()
    ctx.w.fig(fig, "fig11_k_sk_rank_curve.pdf")


# Item 7: the same curve for each dataset (mean over its five training sizes)
def fig_k_curve_per_dataset(ctx):
    sk = ctx.k_sk
    ks = sorted(sk["k"].unique())
    datasets = sorted(sk["dataset"].unique())
    mean_sk = sk.groupby(["dataset", "model", "rule", "k"], as_index=False)["sk_rank"].mean()
    ctx.w.csv(mean_sk, "item07_k_saturation_per_dataset.csv")

    best = (mean_sk.sort_values(["dataset", "model", "rule", "sk_rank", "k"])
            .groupby(["dataset", "model", "rule"], as_index=False).first()
            .rename(columns={"k": "smallest_k_with_best_mean_sk_rank", "sk_rank": "best_mean_sk_rank"}))
    ctx.w.csv(best, "item07_k_saturation_best_k.csv")

    fig, axes = plt.subplots(len(datasets), 3, figsize=(12.0, 2.7 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        _k_curve_axes(axes[i], mean_sk[mean_sk["dataset"] == ds], ks)
        axes[i][0].set_ylabel(f"{ds_label(ds)}\nMean SK rank")
        for col, rule in enumerate(cfg.RULES):
            if i == 0:
                axes[i][col].set_title(rule, fontsize=11, fontweight="bold")
            if i == len(datasets) - 1:
                axes[i][col].set_xlabel("$k$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, title="Base type", bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout()
    ctx.w.fig(fig, "item07_k_saturation_per_dataset.pdf")

    for ds in datasets:
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
        _k_curve_axes(axes, mean_sk[mean_sk["dataset"] == ds], ks)
        for col, rule in enumerate(cfg.RULES):
            axes[col].set_title(rule, fontsize=11, fontweight="bold")
            axes[col].set_xlabel("$k$")
        axes[0].set_ylabel("Mean SK rank")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[2].legend(handles, labels, loc="center left", bbox_to_anchor=(1.04, 0.5), title="Base type", ncol=2)
        fig.tight_layout()
        ctx.w.fig(fig, f"item07_k_saturation_{ds}.pdf")


# Figures 12-13: best k per rule, learners compared within each rule
def fig_rule_heatmap(ctx):
    key = "rule_sk"
    if key not in ctx.memo:
        parts = []
        for rule in cfg.RULES:
            sub = ctx.ens_best_rule[ctx.ens_best_rule["rule"] == rule]
            sk = core.run_sk_metrics(sub, SCEN, "model", cfg.METRICS)
            sk["rule"] = rule
            parts.append(sk)
        ctx.memo[key] = pd.concat(parts, ignore_index=True)
    sk = ctx.memo[key]
    for number, scope, sk_scope, df in [("12", "all", sk, ctx.ens_best_rule),
                                        ("13", "s1", core.s1_only(sk), core.s1_only(ctx.ens_best_rule))]:
        rank = sk_scope.groupby(["rule", "model"])["sk_rank"].mean().unstack("model")
        mre = df.groupby(["rule", "model"])["test_MRE"].mean().unstack("model")
        rank, mre = rank.reindex(cfg.RULES)[cfg.MODEL_TYPES], mre.reindex(cfg.RULES)[cfg.MODEL_TYPES]
        table = (mre.stack().rename("mean_MRE").reset_index()
                 .merge(rank.stack().rename("mean_sk_rank").reset_index(), on=["rule", "model"]))
        ctx.w.csv(table, f"fig{number}_rule_heatmap_{scope}.csv")
        fig = _two_panel(mre.values, rank.values, cfg.RULES, cfg.MODEL_TYPES, "Mean MRE", rotate=0,
                         rank_fmt="{:.2f}")
        fig.set_size_inches(13.0, 2.4)
        ctx.w.fig(fig, f"fig{number}_rule_heatmap_{scope}.pdf")


def run(ctx):
    selection_report(ctx)
    table6(ctx)
    table7(ctx)
    table8(ctx)
    table6(ctx, metrics=cfg.MEDIAN_METRICS, name="item05_table6_singles_median_metrics", with_sa=False)
    table8(ctx, metrics=cfg.MEDIAN_METRICS, name="item05_table8_ensembles_median_metrics", with_sa=False)
    fig_singles_rank_s1(ctx)
    fig_singles_mre(ctx)
    fig_d_forest(ctx)
    fig_gap_close(ctx)
    fig_k_curve(ctx)
    fig_k_curve_per_dataset(ctx)
    fig_rule_heatmap(ctx)
