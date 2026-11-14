"""Dimensionality reduction for classification on MNIST (program v2).

Pipelines : Raw (784-D), PCA(m), LDA(d <= 9), PCA(m) -> LDA(d).
Classifiers: kNN (Euclidean, k chosen by CV) and minimum-Mahalanobis distance
             (pooled within-class covariance with relative ridge, lambda by CV).
Protocol  : every hyper-parameter and the selected dimensionality come from
            stratified 5-fold cross-validation on (a subsample of) the TRAINING
            set only. The test set is loaded after all selection is finished and
            is used for descriptive curves and for the final numbers.

Run: python -B main_v2.py [--cv-subsample N] [--folds 5] [--seed 0] [--pilot]
Reads ../data/mnist.npz (read-only); writes only into v2/results and v2/figures
(sub-folders when --pilot or --out-subdir is used).
"""
import argparse
import csv
import os
import platform
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import matplotlib
import numpy as np
import scipy
import sklearn
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# Experiment definition (DECISION_LOG D-002, D-003, D-005)
# ----------------------------------------------------------------------------
PCA_GRID = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]
PCA_LDA_GRID = [20, 30, 40, 50, 75, 100, 150, 200, 300]
LDA_DIMS = list(range(1, 10))            # c - 1 = 9 discriminant axes
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21]
LAMBDA_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0]   # 3, 10 added (D-H03): pilot chose the old upper edge 1.0
N_CLASSES = 10
KNN_THREADS = max(1, (os.cpu_count() or 1) // 2)   # worker threads of the neighbour search
PCA_MAX = max(PCA_GRID)
DEFAULT_CV_SUBSAMPLE = 20000             # set from the pilot extrapolation (D-005.3)
LAMBDA0_EIG_RATIO = 1e-10                # lambda = 0 admissible only above this ratio
V1_REF = {("Raw", None): 96.88, ("PCA", 30): 97.55}   # v1 k = 5 test accuracy (CTX-0114, CTX-0105)
CTX0003_TRAIN = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
CTX0003_TEST = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(HERE), "data", "mnist.npz")

C_KNN, C_K5, C_MAHA = "#0072B2", "#E69F00", "#009E73"   # Okabe-Ito (colour-blind safe)


# ----------------------------------------------------------------------------
# Logging and timing
# ----------------------------------------------------------------------------
class Log:
    def __init__(self):
        self.fh = None

    def open(self, path):
        self.fh = open(path, "w")

    def __call__(self, msg=""):
        print(msg, flush=True)
        if self.fh:
            self.fh.write(msg + "\n")
            self.fh.flush()


log = Log()
TIMINGS = {}


@contextmanager
def stage(name):
    t0 = time.perf_counter()
    log(f"[stage start] {name}")
    yield
    dt = time.perf_counter() - t0
    TIMINGS[name] = TIMINGS.get(name, 0.0) + dt
    log(f"[stage done ] {name}: {dt:.1f} s")


# ----------------------------------------------------------------------------
# Data (test arrays are only read by load_test, called after selection)
# ----------------------------------------------------------------------------
def load_train():
    with np.load(DATA_PATH, allow_pickle=False) as d:
        return d["x_train"], d["y_train"].astype(np.int64)


def load_test():
    with np.load(DATA_PATH, allow_pickle=False) as d:
        return d["x_test"], d["y_test"].astype(np.int64)


def flatten(x):
    return np.ascontiguousarray(x.reshape(len(x), -1).astype(np.float64) / 255.0)


def centre_of_mass(x_img):
    img = x_img.astype(np.float64)
    tot = img.sum(axis=(1, 2))
    idx = np.arange(img.shape[1], dtype=np.float64)
    rows = (img.sum(axis=2) * idx).sum(axis=1) / tot
    cols = (img.sum(axis=1) * idx).sum(axis=1) / tot
    return rows, cols


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
def pca_fit(X):
    """PCA by eigh of the 784x784 covariance of the training data (one solver everywhere).
    Returns mean, eigenvectors (columns, descending variance) and eigenvalues."""
    mu = X.mean(axis=0)
    Xc = X - mu
    C = Xc.T @ Xc / (len(X) - 1)
    w, V = np.linalg.eigh(C)
    return mu, np.ascontiguousarray(V[:, ::-1]), w[::-1]


LDA_WARNINGS = defaultdict(int)


def lda_fit_transform(Ztr, ytr, Zq):
    """Fisher LDA (scikit-learn, solver='svd', no shrinkage), 9 axes; returns the
    projected training and query data plus fit and transform times."""
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lda = LinearDiscriminantAnalysis(solver="svd", n_components=N_CLASSES - 1)
        lda.fit(Ztr, ytr)
    for w in caught:
        LDA_WARNINGS[f"{w.category.__name__}: {w.message}"] += 1
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    Ttr = np.ascontiguousarray(lda.transform(Ztr))
    Tq = np.ascontiguousarray(lda.transform(Zq))
    return Ttr, Tq, t_fit, time.perf_counter() - t0


def iter_spaces(Xtr, ytr, Xq):
    """Yield (pipeline, pca_dim, lda_dim, Ztr, Zq, fit_time, transform_time) for every
    grid point. PCA is fitted once per call (nested components, columns are sliced)."""
    yield "Raw", None, None, Xtr, Xq, 0.0, 0.0

    t0 = time.perf_counter()
    mu, V, _ = pca_fit(Xtr)
    t_pca = time.perf_counter() - t0
    t0 = time.perf_counter()
    Ptr = np.ascontiguousarray((Xtr - mu) @ V[:, :PCA_MAX])
    Pq = np.ascontiguousarray((Xq - mu) @ V[:, :PCA_MAX])
    t_proj = time.perf_counter() - t0
    for m in PCA_GRID:
        yield ("PCA", m, None, np.ascontiguousarray(Ptr[:, :m]),
               np.ascontiguousarray(Pq[:, :m]), t_pca, t_proj)

    Ttr, Tq, t_fit, t_tr = lda_fit_transform(Xtr, ytr, Xq)
    for d in LDA_DIMS:
        yield ("LDA", None, d, np.ascontiguousarray(Ttr[:, :d]),
               np.ascontiguousarray(Tq[:, :d]), t_fit, t_tr)

    for m in PCA_LDA_GRID:
        Ttr, Tq, t_fit, t_tr = lda_fit_transform(
            np.ascontiguousarray(Ptr[:, :m]), ytr, np.ascontiguousarray(Pq[:, :m]))
        for d in LDA_DIMS:
            yield ("PCA_LDA", m, d, np.ascontiguousarray(Ttr[:, :d]),
                   np.ascontiguousarray(Tq[:, :d]), t_pca + t_fit, t_proj + t_tr)


# ----------------------------------------------------------------------------
# Classifiers
# ----------------------------------------------------------------------------
def knn_sorted_labels(Ztr, ytr, Zq, kmax):
    """Labels of the kmax nearest training points of every query, sorted by Euclidean
    distance (squared distance without the per-query constant |q|^2). Tie rule: among
    the kmax candidates, equal distances are ordered by training index; votes with equal
    counts go to the smallest class label (argmax)."""
    n = len(Ztr)
    kmax = min(kmax, n)
    sq = np.einsum("ij,ij->i", Ztr, Ztr)
    chunk = max(1, min(len(Zq), int(5e6 // n)))
    out = np.empty((len(Zq), kmax), dtype=np.int64)

    def work(s):
        D = Zq[s:s + chunk] @ Ztr.T
        D *= -2.0
        D += sq
        idx = np.argpartition(D, kmax - 1, axis=1)[:, :kmax]
        idx.sort(axis=1)
        order = np.argsort(np.take_along_axis(D, idx, axis=1), axis=1, kind="stable")
        out[s:s + chunk] = ytr[np.take_along_axis(idx, order, axis=1)]

    with ThreadPoolExecutor(max_workers=KNN_THREADS) as ex:   # chunks are independent
        list(ex.map(work, range(0, len(Zq), chunk)))
    return out


def knn_votes(lab, ks):
    """Predictions for every k in ks from one sorted-neighbour label matrix."""
    n = lab.shape[0]
    counts = np.zeros((n, N_CLASSES), dtype=np.int32)
    rows = np.arange(n)
    out = {}
    for j in range(lab.shape[1]):
        counts[rows, lab[:, j]] += 1
        if (j + 1) in ks:
            out[j + 1] = counts.argmax(axis=1)
    return out


def pooled_stats(Z, y):
    """Class means and pooled within-class covariance S_w = sum_i (n_i-1) S_i / (N-c)."""
    means = np.stack([Z[y == c].mean(axis=0) for c in range(N_CLASSES)])
    Zc = Z - means[y]
    return means, Zc.T @ Zc / (len(Z) - N_CLASSES)


def maha_chol(S, lam):
    """Cholesky factor of S + lam*tr(S)/q*I; None if lambda = 0 is inadmissible
    (smallest eigenvalue <= 1e-10 * largest) or the factorisation fails."""
    q = S.shape[0]
    if lam == 0.0:
        ev = np.linalg.eigvalsh(S)
        if ev[0] <= LAMBDA0_EIG_RATIO * ev[-1]:
            return None
        A = S
    else:
        A = S + lam * np.trace(S) / q * np.eye(q)
    try:
        return cholesky(A, lower=True)
    except LinAlgError:
        return None


def maha_predict(Zq, means, L):
    """argmin_i (x-mu_i)' A^-1 (x-mu_i) = argmin_i |L^-1 x - L^-1 mu_i|^2 (equal priors,
    no log-determinant); the per-query term |L^-1 x|^2 is constant and dropped."""
    Wq = solve_triangular(L, Zq.T, lower=True)
    Wm = solve_triangular(L, means.T, lower=True)
    score = (Wm * Wm).sum(axis=0)[None, :] - 2.0 * (Wq.T @ Wm)
    return score.argmin(axis=1)


# ----------------------------------------------------------------------------
# Cross-validation on training data only
# ----------------------------------------------------------------------------
def run_cv(Xs, ys, folds, seed):
    acc = defaultdict(list)          # (pipeline, classifier, m, d) -> [array over hyper per fold]
    skips = defaultdict(int)         # (pipeline, m, d) -> folds in which lambda = 0 was skipped
    cv_time = defaultdict(float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (tr, va) in enumerate(skf.split(Xs, ys)):
        t_fold = last = time.perf_counter()
        ytr, yva = ys[tr], ys[va]
        for pipe, m, d, Ztr, Zq, _, _ in iter_spaces(Xs[tr], ytr, Xs[va]):
            lab = knn_sorted_labels(Ztr, ytr, Zq, max(K_GRID))
            votes = knn_votes(lab, set(K_GRID))
            acc[(pipe, "kNN", m, d)].append(
                np.array([np.mean(votes[k] == yva) for k in K_GRID]))
            means, S = pooled_stats(Ztr, ytr)
            a = []
            for lam in LAMBDA_GRID:
                L = maha_chol(S, lam)
                if L is None:
                    a.append(np.nan)
                    if lam == 0.0:
                        skips[(pipe, m, d)] += 1
                else:
                    a.append(np.mean(maha_predict(Zq, means, L) == yva))
            acc[(pipe, "Maha", m, d)].append(np.array(a))
            now = time.perf_counter()
            cv_time[pipe] += now - last
            last = now
        log(f"  fold {f + 1}/{folds} done in {time.perf_counter() - t_fold:.1f} s "
            f"(train {len(tr)}, validation {len(va)})")
    return acc, skips, cv_time


HYPER = {"kNN": ("k", K_GRID), "Maha": ("lambda", LAMBDA_GRID)}


def cv_table(acc, n_cv_train):
    """Rows of cv_results.csv; a hyper value skipped in any fold is omitted."""
    rows = []
    for (pipe, clf, m, d), folds in acc.items():
        A = np.stack(folds)                      # (n_folds, n_hyper)
        hname, hgrid = HYPER[clf]
        for j, h in enumerate(hgrid):
            if np.isnan(A[:, j]).any():
                continue
            rows.append(dict(pipeline=pipe, classifier=clf, pca_dim=m, lda_dim=d,
                             hyper_name=hname, hyper_value=h,
                             cv_mean_acc=float(A[:, j].mean()),
                             cv_std_acc=float(A[:, j].std(ddof=1)),
                             n_folds=A.shape[0], n_cv_train=n_cv_train))
    return rows


def select(cv_rows):
    """Selection from CV rows only (no test data is passed to this function).
    Per grid point: best hyper-parameter (ties -> smaller value). Per (pipeline,
    classifier): best grid point (ties -> smaller feature dimension, then smaller m,
    then smaller d)."""
    per_point = defaultdict(list)
    for r in cv_rows:
        per_point[(r["pipeline"], r["classifier"], r["pca_dim"], r["lda_dim"])].append(r)
    best_point = {}
    for key, rs in per_point.items():
        top = max(r["cv_mean_acc"] for r in rs)
        best_point[key] = next(r for r in rs if r["cv_mean_acc"] >= top - 1e-12)
    selected = {}
    by_cfg = defaultdict(list)
    for (pipe, clf, m, d), r in best_point.items():
        by_cfg[(pipe, clf)].append(((pipe, clf, m, d), r))
    for cfg, items in by_cfg.items():
        top = max(r["cv_mean_acc"] for _, r in items)
        cands = [(key, r) for key, r in items if r["cv_mean_acc"] >= top - 1e-12]
        cands.sort(key=lambda kr: (kr[0][3] if kr[0][3] is not None else (kr[0][2] or 784),
                                   kr[0][2] or 0, kr[0][3] or 0))
        selected[cfg] = cands[0][1]
    return best_point, selected


# ----------------------------------------------------------------------------
# Test evaluation (after selection)
# ----------------------------------------------------------------------------
def run_test(Xtr, ytr, Xte, yte, best_point, selected):
    rows, preds = [], {}
    sel_keys = {(c[0], c[1], r["pca_dim"], r["lda_dim"]) for c, r in selected.items()}
    t_run = time.perf_counter()
    for i, (pipe, m, d, Ztr, Zq, t_fit, t_xf) in enumerate(iter_spaces(Xtr, ytr, Xte)):
        if pipe in ("Raw", "PCA") or d == 1:
            log(f"  test grid point {i + 1}: {pipe} m={m} d={d} "
                f"({time.perf_counter() - t_run:.0f} s elapsed)")
        if pipe == "PCA":
            assert np.abs(Ztr.mean(axis=0)).max() < 1e-8, "PCA scores not centred on training data"
        # --- kNN (CV-chosen k; the k = 5 curve of v1 for Raw and PCA) ---
        k = best_point[(pipe, "kNN", m, d)]["hyper_value"]
        want5 = pipe in ("Raw", "PCA")
        t0 = time.perf_counter()
        lab = knn_sorted_labels(Ztr, ytr, Zq, max(k, 5) if want5 else k)
        votes = knn_votes(lab, {k, 5})
        t_pred = time.perf_counter() - t0 + t_xf
        for clf, kk in [("kNN", k)] + ([("kNN_k5", 5)] if want5 else []):
            p = votes[kk]
            acc = 100.0 * float(np.mean(p == yte))
            if clf == "kNN":
                preds[(pipe, "kNN", m, d)] = p
            rows.append(dict(pipeline=pipe, classifier=clf, pca_dim=m, lda_dim=d,
                             hyper_name="k", hyper_value=kk, test_acc_pct=acc,
                             test_err_pct=100.0 - acc, fit_time_s=t_fit,
                             predict_time_s=t_pred,
                             is_selected=int(clf == "kNN" and (pipe, clf, m, d) in sel_keys)))
        # --- minimum-Mahalanobis (CV-chosen ridge) ---
        lam = best_point[(pipe, "Maha", m, d)]["hyper_value"]
        t0 = time.perf_counter()
        means, S = pooled_stats(Ztr, ytr)
        L = maha_chol(S, lam)
        while L is None:                      # lambda inadmissible on the full fit: next larger
            nxt = [v for v in LAMBDA_GRID if v > lam]
            log(f"  DEVIATION: lambda={lam} unusable on full training fit of "
                f"{(pipe, m, d)}; using {nxt[0]}")
            lam = nxt[0]
            L = maha_chol(S, lam)
        t_cfit = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = maha_predict(Zq, means, L)
        t_pred = time.perf_counter() - t0 + t_xf
        acc = 100.0 * float(np.mean(p == yte))
        preds[(pipe, "Maha", m, d)] = p
        rows.append(dict(pipeline=pipe, classifier="Maha", pca_dim=m, lda_dim=d,
                         hyper_name="lambda", hyper_value=lam, test_acc_pct=acc,
                         test_err_pct=100.0 - acc, fit_time_s=t_fit + t_cfit,
                         predict_time_s=t_pred,
                         is_selected=int((pipe, "Maha", m, d) in sel_keys)))
    return rows, preds


# ----------------------------------------------------------------------------
# McNemar (exact, two-sided) with explicit Holm adjustment (D-006)
# ----------------------------------------------------------------------------
MCNEMAR_PAIRS = [("M1", ("PCA", "kNN"), ("Raw", "kNN")),
                 ("M2", ("LDA", "kNN"), ("Raw", "kNN")),
                 ("M3", ("PCA_LDA", "kNN"), ("Raw", "kNN")),
                 ("M4", ("PCA", "Maha"), ("Raw", "Maha")),
                 ("M5", ("PCA", "kNN"), ("PCA", "Maha")),
                 ("M6", ("PCA_LDA", "kNN"), ("PCA", "kNN")),
                 ("M7", ("LDA", "Maha"), ("LDA", "kNN"))]


def cfg_name(cfg, r):
    pipe, clf = cfg
    dims = {"Raw": "784", "PCA": f"m={r['pca_dim']}", "LDA": f"d={r['lda_dim']}",
            "PCA_LDA": f"m={r['pca_dim']},d={r['lda_dim']}"}[pipe]
    return f"{pipe}({dims})+{clf}({r['hyper_name']}={r['hyper_value']:g})"


def mcnemar_table(preds, yte, selected, test_rows):
    out = []
    for pid, a, b in MCNEMAR_PAIRS:
        ra, rb = selected[a], selected[b]
        pa = preds[(a[0], a[1], ra["pca_dim"], ra["lda_dim"])]
        pb = preds[(b[0], b[1], rb["pca_dim"], rb["lda_dim"])]
        ca, cb = pa == yte, pb == yte
        n_a, n_b = int(np.sum(ca & ~cb)), int(np.sum(cb & ~ca))
        p = 1.0 if n_a + n_b == 0 else float(binomtest(min(n_a, n_b), n_a + n_b, 0.5).pvalue)
        out.append(dict(pair_id=pid, config_a=cfg_name(a, ra), config_b=cfg_name(b, rb),
                        acc_a=100.0 * float(ca.mean()), acc_b=100.0 * float(cb.mean()),
                        n_a_only_correct=n_a, n_b_only_correct=n_b, p_value=p))
    # Holm: sort ascending, multiply by (m - rank), enforce monotonicity, cap at 1
    order = sorted(range(len(out)), key=lambda i: out[i]["p_value"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(out) - rank) * out[i]["p_value"]))
        out[i]["p_holm"] = running
    return out


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("NA" if r[k] is None else r[k]) for k in fieldnames})


def plot_pca(rows, selected, metric, ylabel, path):
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    spec = [("kNN", "kNN (CV-chosen k)", C_KNN, "o", "-"),
            ("kNN_k5", "kNN (k = 5)", C_K5, "s", "--"),
            ("Maha", "Min. Mahalanobis (CV-chosen ridge)", C_MAHA, "^", "-")]
    for clf, label, col, mk, ls in spec:
        pts = sorted((r["pca_dim"], r[metric]) for r in rows
                     if r["pipeline"] == "PCA" and r["classifier"] == clf)
        ax.plot(*zip(*pts), color=col, marker=mk, ls=ls, lw=1.6, ms=4.5, label=label)
    for clf, col in [("kNN", C_KNN), ("Maha", C_MAHA)]:
        raw = next(r for r in rows if r["pipeline"] == "Raw" and r["classifier"] == clf)
        ax.axhline(raw[metric], color=col, ls=":", lw=1.2,
                   label=f"Raw 784-D, {'kNN' if clf == 'kNN' else 'Mahalanobis'}")
    for clf, col in [("kNN", C_KNN), ("Maha", C_MAHA)]:
        m = selected[("PCA", clf)]["pca_dim"]
        r = next(r for r in rows if r["pipeline"] == "PCA" and r["classifier"] == clf
                 and r["pca_dim"] == m)
        ax.scatter([m], [r[metric]], s=130, facecolors="none", edgecolors="black",
                   linewidths=1.5, zorder=5,
                   label="CV-selected" if clf == "kNN" else None)
    ax.set_xscale("log")
    ax.set_xticks(PCA_GRID)
    ax.set_xticklabels([str(m) for m in PCA_GRID], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("PCA dimension m")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_lda(rows, selected, metric, ylabel, path):
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for clf, col, mk, nm in [("kNN", C_KNN, "o", "kNN"), ("Maha", C_MAHA, "^", "Mahalanobis")]:
        pts = sorted((r["lda_dim"], r[metric]) for r in rows
                     if r["pipeline"] == "LDA" and r["classifier"] == clf)
        ax.plot(*zip(*pts), color=col, marker=mk, ls="-", lw=1.6, ms=4.5, label=f"LDA, {nm}")
        ms = selected[("PCA_LDA", clf)]["pca_dim"]
        pts = sorted((r["lda_dim"], r[metric]) for r in rows
                     if r["pipeline"] == "PCA_LDA" and r["classifier"] == clf
                     and r["pca_dim"] == ms)
        ax.plot(*zip(*pts), color=col, marker=mk, ls="--", lw=1.4, ms=4.5, mfc="white",
                label=f"PCA({ms}) -> LDA, {nm}")
    ax.set_xticks(LDA_DIMS)
    ax.set_xlabel("LDA dimension d (at most c - 1 = 9)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pca_lda_m(rows, selected, path):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for clf, col, mk, nm in [("kNN", C_KNN, "o", "kNN"), ("Maha", C_MAHA, "^", "Mahalanobis")]:
        pts = sorted((r["pca_dim"], r["test_acc_pct"]) for r in rows
                     if r["pipeline"] == "PCA_LDA" and r["classifier"] == clf
                     and r["lda_dim"] == 9)
        ax.plot(*zip(*pts), color=col, marker=mk, lw=1.6, ms=4.5, label=nm)
        ms = selected[("PCA_LDA", clf)]["pca_dim"]
        y = next(v for x, v in pts if x == ms)
        ax.scatter([ms], [y], s=130, facecolors="none", edgecolors="black", linewidths=1.5,
                   zorder=5, label="CV-selected m" if clf == "kNN" else None)
    ax.set_xlabel("PCA dimension m before LDA (d = 9)")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_variance(cum, n90, n95, path):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(1, len(cum) + 1)
    ax.plot(x, 100.0 * cum, color=C_KNN, lw=1.8)
    for n, lvl in [(n90, 90), (n95, 95)]:
        ax.plot([n, n], [0, lvl], color="grey", ls=":", lw=1)
        ax.plot([0, n], [lvl, lvl], color="grey", ls=":", lw=1)
        ax.scatter([n], [lvl], color="black", s=18, zorder=5)
        ax.annotate(f"{lvl}%: {n} comp.", (n, lvl), textcoords="offset points",
                    xytext=(8, -12), fontsize=7)
    ax.set_xlim(0, len(cum))
    ax.set_ylim(0, 101)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cv-subsample", type=int, default=None,
                    help=f"stratified training subsample used for CV (default {DEFAULT_CV_SUBSAMPLE}; "
                         "5000 with --pilot)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot", action="store_true",
                    help="timing pilot: CV subsample 5000, outputs in results/pilot, figures/pilot")
    ap.add_argument("--out-subdir", default=None, help="write into results/<sub>, figures/<sub>")
    ap.add_argument("--max-train", type=int, default=None, help="smoke test only: use a stratified "
                    "subset of the training data for CV AND final fits")
    ap.add_argument("--max-test", type=int, default=None, help="smoke test only: first N test images")
    args = ap.parse_args()
    cv_n = args.cv_subsample or (5000 if args.pilot else DEFAULT_CV_SUBSAMPLE)
    sub = args.out_subdir or ("pilot" if args.pilot else "")
    res_dir = os.path.join(HERE, "results", sub)
    fig_dir = os.path.join(HERE, "figures", sub)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    log.open(os.path.join(res_dir, "run_log.txt"))
    t_start = time.perf_counter()

    log("== main_v2.py: dimensionality reduction for classification (MNIST) ==")
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, "
        f"scikit-learn {sklearn.__version__}, matplotlib {matplotlib.__version__}")
    log(f"platform {platform.platform()}, machine {platform.machine()}, cpu count {os.cpu_count()}")
    log(f"arguments: {vars(args)}")
    log(f"seed {args.seed}; folds {args.folds}; CV subsample {cv_n}; output {res_dir}")
    if args.max_train or args.max_test:
        log("WARNING: --max-train/--max-test are smoke-test options; results are not the experiment.")

    # ---- training data only --------------------------------------------------
    with stage("data load + checks (train)"):
        x_img, y_all = load_train()
        if args.max_train:
            keep, _ = train_test_split(np.arange(len(y_all)), train_size=args.max_train,
                                       stratify=y_all, random_state=args.seed)
            keep = np.sort(keep)
            x_img, y_all = x_img[keep], y_all[keep]
        counts = np.bincount(y_all, minlength=N_CLASSES).tolist()
        log(f"train images {x_img.shape}, class counts {counts}")
        if not args.max_train:
            log(f"class counts equal CTX-0003 (train): {counts == CTX0003_TRAIN}")
        rows_c, cols_c = centre_of_mass(x_img)
        log(f"centre of mass (train, row/col in pixels): row mean {rows_c.mean():.3f} "
            f"std {rows_c.std():.3f}; col mean {cols_c.mean():.3f} std {cols_c.std():.3f}")
        X = flatten(x_img)
        n_const = int(np.sum(X.std(axis=0) == 0))
        log(f"zero-variance pixels in training set: {n_const} of {X.shape[1]}")
        _, _, ev = pca_fit(X)
        cum = np.cumsum(ev) / ev.sum()
        n90, n95, n99 = (int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99))
        log(f"components for 90% / 95% / 99% cumulative variance (train): {n90} / {n95} / {n99}")
        write_csv(os.path.join(res_dir, "explained_variance.csv"),
                  ["n_components", "cumulative_explained_variance_ratio"],
                  [dict(n_components=i + 1, cumulative_explained_variance_ratio=float(c))
                   for i, c in enumerate(cum)])
        n_cv = min(cv_n, len(y_all))
        if n_cv < len(y_all):
            sub_idx, _ = train_test_split(np.arange(len(y_all)), train_size=n_cv,
                                          stratify=y_all, random_state=args.seed)
            sub_idx = np.sort(sub_idx)
        else:
            sub_idx = np.arange(len(y_all))
        Xs, ys = X[sub_idx], y_all[sub_idx]
        log(f"CV subsample {len(ys)} images, class counts {np.bincount(ys).tolist()}")
        n_cv_train = len(ys) - len(ys) // args.folds

    # ---- cross-validation (training data only) -------------------------------
    with stage("cross-validation (all pipelines)"):
        acc, skips, cv_time = run_cv(Xs, ys, args.folds, args.seed)
    for pipe, t in cv_time.items():
        TIMINGS[f"  CV pipeline {pipe}"] = t
        log(f"  CV time, pipeline {pipe}: {t:.1f} s")
    cv_rows = cv_table(acc, n_cv_train)
    write_csv(os.path.join(res_dir, "cv_results.csv"),
              ["pipeline", "classifier", "pca_dim", "lda_dim", "hyper_name", "hyper_value",
               "cv_mean_acc", "cv_std_acc", "n_folds", "n_cv_train"], cv_rows)
    for pipe in ("Raw", "PCA", "LDA", "PCA_LDA"):
        n_pts = len({(k[2], k[3]) for k in acc if k[0] == pipe})
        n_skip = sum(1 for k, v in skips.items() if k[0] == pipe)
        log(f"lambda = 0 skipped (singular pooled covariance) in {n_skip} of {n_pts} "
            f"grid points of {pipe} (in at least one fold)")
    log(f"LDA warnings (scikit-learn): {dict(LDA_WARNINGS) or 'none'}")

    # ---- selection (no test data exists in memory yet) -----------------------
    best_point, selected = select(cv_rows)
    sel_rows = []
    for (pipe, clf), r in sorted(selected.items()):
        sel_rows.append(dict(pipeline=pipe, classifier=clf, pca_dim=r["pca_dim"],
                             lda_dim=r["lda_dim"], hyper_name=r["hyper_name"],
                             hyper_value=r["hyper_value"], cv_mean_acc=r["cv_mean_acc"],
                             cv_std_acc=r["cv_std_acc"]))
        log(f"selected {pipe}+{clf}: pca_dim={r['pca_dim']} lda_dim={r['lda_dim']} "
            f"{r['hyper_name']}={r['hyper_value']:g} CV acc {100 * r['cv_mean_acc']:.2f}% "
            f"(sd {100 * r['cv_std_acc']:.2f})")
    write_csv(os.path.join(res_dir, "selected_config.csv"),
              ["pipeline", "classifier", "pca_dim", "lda_dim", "hyper_name", "hyper_value",
               "cv_mean_acc", "cv_std_acc"], sel_rows)

    # ---- test set: loaded only now -------------------------------------------
    with stage("data load (test)"):
        x_te, y_te = load_test()
        if args.max_test:
            x_te, y_te = x_te[:args.max_test], y_te[:args.max_test]
        if not args.max_test:
            log(f"class counts equal CTX-0003 (test): "
                f"{np.bincount(y_te, minlength=N_CLASSES).tolist() == CTX0003_TEST}")
        Xte = flatten(x_te)
    with stage("final fits on all training images + test evaluation"):
        test_rows, preds = run_test(X, y_all, Xte, y_te, best_point, selected)
    write_csv(os.path.join(res_dir, "test_results.csv"),
              ["pipeline", "classifier", "pca_dim", "lda_dim", "hyper_name", "hyper_value",
               "test_acc_pct", "test_err_pct", "fit_time_s", "predict_time_s", "is_selected"],
              test_rows)
    for r in test_rows:
        if r["is_selected"]:
            log(f"TEST {r['pipeline']}+{r['classifier']} (selected by CV): pca_dim={r['pca_dim']} "
                f"lda_dim={r['lda_dim']} {r['hyper_name']}={r['hyper_value']:g} "
                f"acc {r['test_acc_pct']:.2f}% err {r['test_err_pct']:.2f}%")

    # sanity check against v1 (D-007.5)
    if not (args.max_train or args.max_test):
        for (pipe, m), ref in V1_REF.items():
            r = next(r for r in test_rows if r["pipeline"] == pipe and r["pca_dim"] == m
                     and r["classifier"] == "kNN_k5")
            diff = r["test_acc_pct"] - ref
            log(f"sanity v1: {pipe}{'' if m is None else f'({m})'} + kNN k=5 test acc "
                f"{r['test_acc_pct']:.2f}% vs v1 {ref:.2f}% (diff {diff:+.2f} pp)"
                + ("  INVESTIGATE (> 0.1 pp)" if abs(diff) > 0.1 else ""))

    with stage("McNemar"):
        mc = mcnemar_table(preds, y_te, selected, test_rows)
        write_csv(os.path.join(res_dir, "mcnemar.csv"),
                  ["pair_id", "config_a", "config_b", "acc_a", "acc_b", "n_a_only_correct",
                   "n_b_only_correct", "p_value", "p_holm"], mc)
        for r in mc:
            log(f"{r['pair_id']}: {r['config_a']} vs {r['config_b']}: "
                f"{r['n_a_only_correct']}/{r['n_b_only_correct']} p={r['p_value']:.3g} "
                f"holm={r['p_holm']:.3g}")

    with stage("plots"):
        plot_pca(test_rows, selected, "test_acc_pct", "Test accuracy (%)",
                 os.path.join(fig_dir, "pca_accuracy.png"))
        plot_pca(test_rows, selected, "test_err_pct", "Test error (%)",
                 os.path.join(fig_dir, "pca_error.png"))
        plot_lda(test_rows, selected, "test_acc_pct", "Test accuracy (%)",
                 os.path.join(fig_dir, "lda_accuracy.png"))
        plot_lda(test_rows, selected, "test_err_pct", "Test error (%)",
                 os.path.join(fig_dir, "lda_error.png"))
        plot_pca_lda_m(test_rows, selected, os.path.join(fig_dir, "pca_lda_vs_m.png"))
        plot_variance(cum, n90, n95, os.path.join(fig_dir, "explained_variance.png"))

    total = time.perf_counter() - t_start
    log("== stage timings (s) ==")
    for k, v in TIMINGS.items():
        log(f"{k}: {v:.1f}")
    log(f"total: {total:.1f} s ({total / 60:.1f} min)")
    log(f"end: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
