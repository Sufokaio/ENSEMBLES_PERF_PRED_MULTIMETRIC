# Output helpers: CSV, LaTeX and figure writers shared by all analysis items.

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

DS_DISPLAY = {
    "apache": "APACHE", "bdbc": "BDBC", "dune": "DUNE", "hipacc": "HIPACC",
    "hsmgp": "HSMGP", "kanzi": "KANZI", "lrzip": "LRZIP", "x264": "X264",
}


def ds_label(name):
    return DS_DISPLAY.get(str(name).lower(), str(name).upper())


class Writer:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.written = []

    def _path(self, kind, name):
        path = os.path.join(self.out_dir, kind, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.written.append(os.path.join(kind, name))
        return path

    def csv(self, df, name):
        df.to_csv(self._path("csv", name), index=False)
        print(f"  wrote csv/{name} ({len(df):,} rows)")

    def tex(self, lines, name):
        with open(self._path("latex", name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  wrote latex/{name}")

    def fig(self, fig, name):
        fig.savefig(self._path("figures", name), format="pdf")
        plt.close(fig)
        print(f"  wrote figures/{name}")


def bold(s):
    return r"\textbf{" + str(s) + "}"


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def tabular(header_rows, body_rows, col_spec, footer=None):
    lines = [r"\begin{tabular}{" + col_spec + "}", r"\toprule"]
    lines += header_rows
    lines.append(r"\midrule")
    for row in body_rows:
        lines.append(row if isinstance(row, str) else " & ".join(str(c) for c in row) + r" \\")
    lines.append(r"\bottomrule")
    if footer:
        lines += footer
    lines.append(r"\end{tabular}")
    return lines


def heatmap(ax, mat, row_labels, col_labels, fmt="{:.2f}", cmap="YlOrRd", vmin=None, vmax=None,
            fontsize=8, rotate=38):
    import numpy as np
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize, color="black")
    return im
