import os

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_DIR = "dataset_chars"
ETA_THRESHOLD = 0.05

SIZES = {
    "apache": [9, 18, 27, 36, 45],
    "bdbc": [18, 36, 54, 72, 90],
    "kanzi": [31, 62, 93, 124, 155],
    "x264": [16, 32, 48, 64, 80],
    "lrzip": [127, 295, 386, 485, 907],
    "dune": [224, 692, 1000, 1365, 1612],
    "hipacc": [261, 528, 736, 1281, 2631],
    "hsmgp": [77, 173, 384, 480, 864],
}


def eta_squared(x, y):
    grand = y.mean()
    ss_total = ((y - grand) ** 2).sum()
    groups = pd.Series(y).groupby(np.asarray(x))
    ss_between = (groups.size() * (groups.mean() - grand) ** 2).sum()
    return float(ss_between / ss_total)


def skewness(y):
    d = y - y.mean()
    return float((d ** 3).mean() / (d ** 2).mean() ** 1.5)


rows, option_rows = [], []
for name in sorted(SIZES):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    X, y = df.iloc[:, :-1], df.iloc[:, -1].to_numpy(dtype=float)
    distinct = X.nunique()

    constant = [c for c in X.columns if distinct[c] == 1]
    binary = [c for c in X.columns if distinct[c] == 2]
    numeric = [c for c in X.columns if distinct[c] > 2]
    n_options = len(binary) + len(numeric)

    eta = {c: eta_squared(X[c], y) for c in binary + numeric}
    ranked = sorted(eta, key=eta.get, reverse=True)
    for rank, c in enumerate(ranked, start=1):
        option_rows.append({
            "dataset": name,
            "option": c,
            "type": "binary" if c in binary else "numeric",
            "distinct_values": int(distinct[c]),
            "eta_squared": eta[c],
            "rank": rank,
        })

    row = {
        "dataset": name,
        "n_configurations": len(df),
        "n_columns": X.shape[1],
        "n_constant_columns": len(constant),
        "n_options": n_options,
        "n_binary": len(binary),
        "n_numeric": len(numeric),
        "share_numeric": len(numeric) / n_options,
        "numeric_distinct_values": "; ".join(f"{c}: {int(distinct[c])}" for c in numeric),
    }
    for i, s in enumerate(SIZES[name], start=1):
        row[f"S{i}"] = s
    for i, s in enumerate(SIZES[name], start=1):
        row[f"S{i}_per_option"] = s / n_options
    row.update({
        "skewness": skewness(y),
        "cv": float(y.std(ddof=1) / y.mean()),
        "eta2_max": eta[ranked[0]],
        "eta2_max_option": ranked[0],
        "eta2_top3_sum": float(sum(eta[c] for c in ranked[:3])),
        "eta2_top3_options": "; ".join(ranked[:3]),
        "n_options_eta2_above_0.05": int(sum(v > ETA_THRESHOLD for v in eta.values())),
    })
    rows.append(row)

table = pd.DataFrame(rows)
options = pd.DataFrame(option_rows)
table.to_csv(os.path.join(OUT_DIR, "dataset_characteristics.csv"), index=False)
options.to_csv(os.path.join(OUT_DIR, "option_eta_squared.csv"), index=False)


def tex_escape(s):
    return str(s).replace("_", r"\_")


lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Characteristics of the subject systems. $p$: number of non-constant options; "
    r"$S_i/p$: training size divided by $p$; CV: coefficient of variation of the performance value; "
    r"$\eta^2$: share of the variance of the performance value explained by a single option.}",
    r"\label{tab:dataset_chars}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lrrrrlrrrrrrrrrr}",
    r"\toprule",
    r"System & \#Conf. & \#Bin. & \#Num. & \%Num. & Distinct values (numeric) & $S_1/p$ & $S_2/p$ & $S_3/p$ "
    r"& $S_4/p$ & $S_5/p$ & Skew. & CV & max $\eta^2$ & top-3 $\sum\eta^2$ & \#$\eta^2{>}0.05$ \\",
    r"\midrule",
]
for r in rows:
    numeric_values = ", ".join(v.split(": ")[1] for v in r["numeric_distinct_values"].split("; ")) \
        if r["numeric_distinct_values"] else "--"
    cells = [
        r["dataset"].upper(),
        f"{r['n_configurations']:,}".replace(",", "{,}"),
        str(r["n_binary"]),
        str(r["n_numeric"]),
        f"{100 * r['share_numeric']:.1f}",
        numeric_values,
    ]
    cells += [f"{r[f'S{i}_per_option']:.1f}" for i in range(1, 6)]
    cells += [
        f"{r['skewness']:.2f}",
        f"{r['cv']:.2f}",
        f"{r['eta2_max']:.2f} ({tex_escape(r['eta2_max_option'])})",
        f"{r['eta2_top3_sum']:.2f}",
        str(r["n_options_eta2_above_0.05"]),
    ]
    lines.append(" & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
with open(os.path.join(OUT_DIR, "dataset_characteristics.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(table.drop(columns=["numeric_distinct_values", "eta2_top3_options"]).round(3).to_string(index=False))
print(table[["dataset", "numeric_distinct_values", "eta2_top3_options"]].to_string(index=False))
