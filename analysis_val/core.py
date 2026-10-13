# Selection of the best single / best ensemble, Scott-Knott and Borda helpers, effect sizes.

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as cfg
from sk_impl import a12, scott_knott

SCEN = ["dataset", "sample_size"]


def add_sa_d(df, baseline):
    out = df.merge(baseline[SCEN + ["MAEp0", "Sp0"]], on=SCEN)
    out["test_SA"] = 1.0 - out["test_MAE"] / out["MAEp0"]
    out["test_D"] = (out["test_MAE"] - out["MAEp0"]) / out["Sp0"]
    return out.drop(columns=["MAEp0", "Sp0"])


def s1_only(df):
    return df[df["sample_size"] == df.groupby("dataset")["sample_size"].transform("min")]


def size_index(df):
    return df.groupby("dataset")["sample_size"].rank(method="dense").astype(int)


# Borda over the validation means of MAE, MRE, MBRE, MIBRE, ties broken by MAE (as in ensembles_val.py)
def val_borda(means, group_cols, id_cols):
    df = means.sort_values(group_cols + ["val_MAE"] + id_cols).reset_index(drop=True)
    score = 0
    for m in cfg.BORDA_METRICS:
        df[f"val_rank_{m}"] = df.groupby(group_cols)[f"val_{m}"].rank(method="first")
        score = score + df[f"val_rank_{m}"]
    df["val_borda_score"] = score
    df = df.sort_values(group_cols + ["val_borda_score", "val_MAE"] + id_cols).reset_index(drop=True)
    df["val_borda_rank"] = df.groupby(group_cols).cumcount() + 1
    return df


def select_best_singles(variants, selection):
    if selection == "val":
        chosen = variants.loc[variants["val_borda_rank"] == 1, cfg.KEYS + ["variant"]].drop_duplicates()
    else:
        means = variants.groupby(cfg.KEYS + ["variant"], as_index=False)["test_MRE"].mean()
        chosen = means.loc[means.groupby(cfg.KEYS)["test_MRE"].idxmin(), cfg.KEYS + ["variant"]]
    return variants.merge(chosen, on=cfg.KEYS + ["variant"]).reset_index(drop=True)


def _candidate_means(ensembles, id_cols):
    cols = [f"val_{m}" for m in cfg.METRICS] + ["test_MRE"]
    return ensembles.groupby(cfg.KEYS + id_cols, as_index=False)[cols].mean()


def _select(ensembles, selection, group_cols, id_cols):
    means = _candidate_means(ensembles, id_cols)
    if selection == "val":
        ranked = val_borda(means, group_cols, id_cols)
        chosen = ranked.loc[ranked["val_borda_rank"] == 1, cfg.KEYS + id_cols]
    else:
        means = means.sort_values(cfg.KEYS + id_cols).reset_index(drop=True)
        chosen = means.loc[means.groupby(group_cols)["test_MRE"].idxmin(), cfg.KEYS + id_cols]
    chosen = chosen.drop_duplicates()
    return ensembles.merge(chosen, on=cfg.KEYS + id_cols).reset_index(drop=True)


# Best (selector, k, rule) per learner and scenario
def select_best_ensembles(ensembles, selection, selectors=None):
    selectors = selectors or cfg.BEST_ENS_SELECTORS
    sub = ensembles[ensembles["selector"].isin(selectors)]
    return _select(sub, selection, cfg.KEYS, ["selector", "k", "rule"])


# Best k per rule (top-k) per learner and scenario
def select_best_k_per_rule(ensembles, selection):
    sub = ensembles[ensembles["selector"] == "top"]
    out = []
    for rule in cfg.RULES:
        out.append(_select(sub[sub["rule"] == rule], selection, cfg.KEYS, ["selector", "k", "rule"]))
    return pd.concat(out, ignore_index=True)


def _sk_one(key, groups):
    names = list(groups)
    if len(names) == 1:
        return key, {names[0]: 1}, False
    try:
        result = scott_knott([(n, list(v)) for n, v in groups.items()], **cfg.SK_KWARGS)
        return key, {name: int(rank) for rank, name, *_ in result}, False
    except Exception:
        return key, {n: 1 for n in names}, True


# Scott-Knott over the runs of each competitor, one test per group of `by`
def run_sk(df, by, competitor_col, value_col, n_jobs=None):
    n_jobs = n_jobs or cfg.N_JOBS
    tasks = []
    for key, grp in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        groups = {name: g[value_col].to_numpy() for name, g in grp.groupby(competitor_col, sort=True)}
        tasks.append((key, groups))
    results = Parallel(n_jobs=n_jobs)(delayed(_sk_one)(k, g) for k, g in tasks)
    rows, failed = [], 0
    for key, ranks, fail in results:
        failed += int(fail)
        for name, rank in ranks.items():
            rows.append(dict(zip(by, key), **{competitor_col: name, "sk_rank": rank}))
    if failed:
        print(f"  Scott-Knott failed in {failed} of {len(tasks)} groups (all competitors set to rank 1)")
    return pd.DataFrame(rows)


# Scott-Knott for several metrics; `df` is wide with columns test_<metric>
def run_sk_metrics(df, by, competitor_col, metrics, n_jobs=None):
    out = []
    for m in metrics:
        sk = run_sk(df, by, competitor_col, f"test_{m}", n_jobs)
        sk["metric"] = m
        out.append(sk)
    return pd.concat(out, ignore_index=True)


def borda_from_sk(sk, competitor_col, by=None):
    by = by or []
    total = sk.groupby(by + [competitor_col], as_index=False)["sk_rank"].sum()
    total = total.rename(columns={"sk_rank": "borda_total"})
    if by:
        total["borda_rank"] = total.groupby(by)["borda_total"].rank(method="min").astype(int)
    else:
        total["borda_rank"] = total["borda_total"].rank(method="min").astype(int)
    return total


def wilson_ci(successes, total, alpha=0.05):
    from scipy.stats import norm
    if total == 0:
        return 0.0, 0.0, 1.0
    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = z / denom * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
    return p, float(max(0.0, center - margin)), float(min(1.0, center + margin))


# Probability that a run of `better` has a lower value than a run of `reference`
def a12_lower(better, reference):
    return float(a12(reference, better))


def a12_magnitude(value):
    d = abs(value - 0.5)
    if d >= 0.21:
        return "large"
    if d >= 0.14:
        return "medium"
    if d >= 0.06:
        return "small"
    return "negligible"


def scenario_means(df, group_cols, metrics, prefix="test_"):
    cols = [f"{prefix}{m}" for m in metrics]
    return df.groupby(group_cols, as_index=False)[cols].mean()
