"""Re-draw Figures 1 and 2 at print size from the saved results (program v2).

Run: python -B replot_figures.py
Reads results/test_results.csv and results/selected_config.csv (read-only).
Writes figures/pca_accuracy_print.png and figures/lda_accuracy_print.png only.
No experiment is rerun and no value is changed. The series, markers, line
styles, colours and labels are those of plot_pca and plot_lda in main_v2.py
(metric = test accuracy). Only the size changes, so that the figures can be
printed 8.0 cm wide side by side: figsize 3.15 x 2.5 in, axis labels 8 pt,
ticks 7 pt, legend 6.5 pt, lines about 1 pt, 300 dpi.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")

PCA_GRID = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]   # as in main_v2.py
LDA_DIMS = list(range(1, 10))
C_KNN, C_K5, C_MAHA = "#0072B2", "#E69F00", "#009E73"   # Okabe-Ito, as in main_v2.py

FIGSIZE = (3.15, 2.5)        # inches; 3.15 in = 8.0 cm = print width in the pair
LABEL_PT, TICK_PT, LEGEND_PT = 8, 7, 6.5
LW, LW_RAW, MS, RING_S, RING_LW = 1.0, 0.9, 3.0, 45, 1.0
DPI = 300


def _num(v, cast):
    return None if v == "NA" else cast(v)


def read_test_rows(path):
    """test_results.csv -> list of dicts with the types main_v2.py used."""
    with open(path, newline="") as fh:
        return [dict(pipeline=r["pipeline"], classifier=r["classifier"],
                     pca_dim=_num(r["pca_dim"], int), lda_dim=_num(r["lda_dim"], int),
                     test_acc_pct=float(r["test_acc_pct"]))
                for r in csv.DictReader(fh)]


def read_selected(path):
    """selected_config.csv -> {(pipeline, classifier): row} as in main_v2.py."""
    with open(path, newline="") as fh:
        return {(r["pipeline"], r["classifier"]):
                dict(pca_dim=_num(r["pca_dim"], int), lda_dim=_num(r["lda_dim"], int))
                for r in csv.DictReader(fh)}


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=LABEL_PT)
    ax.set_ylabel(ylabel, fontsize=LABEL_PT)
    ax.tick_params(labelsize=TICK_PT, width=0.6, length=2.5)
    ax.grid(alpha=0.3, lw=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def plot_pca(rows, selected, path, metric="test_acc_pct", ylabel="Test accuracy (%)"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    spec = [("kNN", "kNN (CV-chosen k)", C_KNN, "o", "-"),
            ("kNN_k5", "kNN (k = 5)", C_K5, "s", "--"),
            ("Maha", "Min. Mahalanobis (CV-chosen ridge)", C_MAHA, "^", "-")]
    for clf, label, col, mk, ls in spec:
        pts = sorted((r["pca_dim"], r[metric]) for r in rows
                     if r["pipeline"] == "PCA" and r["classifier"] == clf)
        ax.plot(*zip(*pts), color=col, marker=mk, ls=ls, lw=LW, ms=MS, label=label)
    for clf, col in [("kNN", C_KNN), ("Maha", C_MAHA)]:
        raw = next(r for r in rows if r["pipeline"] == "Raw" and r["classifier"] == clf)
        ax.axhline(raw[metric], color=col, ls=":", lw=LW_RAW,
                   label=f"Raw 784-D, {'kNN' if clf == 'kNN' else 'Mahalanobis'}")
    for clf in ("kNN", "Maha"):
        m = selected[("PCA", clf)]["pca_dim"]
        r = next(r for r in rows if r["pipeline"] == "PCA" and r["classifier"] == clf
                 and r["pca_dim"] == m)
        ax.scatter([m], [r[metric]], s=RING_S, facecolors="none", edgecolors="black",
                   linewidths=RING_LW, zorder=5,
                   label="CV-selected" if clf == "kNN" else None)
    ax.set_xscale("log")
    ax.set_xticks(PCA_GRID)
    ax.set_xticklabels([str(m) for m in PCA_GRID], fontsize=TICK_PT, rotation=90)
    ax.minorticks_off()
    style_axes(ax, "PCA dimension m", ylabel)
    # Lower right is empty (all curves are above 86 % for m >= 20).
    ax.legend(fontsize=LEGEND_PT, loc="lower right", frameon=False, handlelength=2.2,
              borderaxespad=0.3, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=DPI)
    return fig


def plot_lda(rows, selected, path, metric="test_acc_pct", ylabel="Test accuracy (%)"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for clf, col, mk, nm in [("kNN", C_KNN, "o", "kNN"), ("Maha", C_MAHA, "^", "Mahalanobis")]:
        pts = sorted((r["lda_dim"], r[metric]) for r in rows
                     if r["pipeline"] == "LDA" and r["classifier"] == clf)
        ax.plot(*zip(*pts), color=col, marker=mk, ls="-", lw=LW, ms=MS, label=f"LDA, {nm}")
        ms = selected[("PCA_LDA", clf)]["pca_dim"]
        pts = sorted((r["lda_dim"], r[metric]) for r in rows
                     if r["pipeline"] == "PCA_LDA" and r["classifier"] == clf
                     and r["pca_dim"] == ms)
        ax.plot(*zip(*pts), color=col, marker=mk, ls="--", lw=LW, ms=MS, mfc="white",
                label=f"PCA({ms}) -> LDA, {nm}")
    ax.set_xticks(LDA_DIMS)
    style_axes(ax, "LDA dimension d (at most c - 1 = 9)", ylabel)
    # Lower right is empty (all curves are above 80 % for d >= 5).
    ax.legend(fontsize=LEGEND_PT, loc="lower right", frameon=False, handlelength=2.2,
              borderaxespad=0.3, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=DPI)
    return fig


def main():
    rows = read_test_rows(os.path.join(RES_DIR, "test_results.csv"))
    selected = read_selected(os.path.join(RES_DIR, "selected_config.csv"))
    for name, fn in [("pca_accuracy_print.png", plot_pca), ("lda_accuracy_print.png", plot_lda)]:
        path = os.path.join(FIG_DIR, name)
        plt.close(fn(rows, selected, path))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
