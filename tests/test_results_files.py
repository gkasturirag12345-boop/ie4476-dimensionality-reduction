"""The committed v2 results are internally consistent, and the README quotes them correctly."""
import csv
import os
import re
import sys

import pytest
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "v2", "results")
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def read(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def test_selected_configs_one_per_pipeline_and_classifier():
    sel = [r for r in read("test_results.csv") if r["is_selected"] == "1"]
    assert sorted((r["pipeline"], r["classifier"]) for r in sel) == sorted(
        (p, c) for p in ("Raw", "PCA", "LDA", "PCA_LDA") for c in ("kNN", "Maha"))
    cv = {(r["pipeline"], r["classifier"]): r for r in read("selected_config.csv")}
    for r in sel:                                   # the test row uses exactly the CV-selected setting
        s = cv[(r["pipeline"], r["classifier"])]
        assert (r["pca_dim"], r["lda_dim"], r["hyper_value"]) == (s["pca_dim"], s["lda_dim"], s["hyper_value"])


def test_v1_reference_reproduced():
    rows = {(r["pipeline"], r["pca_dim"]): float(r["test_acc_pct"])
            for r in read("test_results.csv") if r["classifier"] == "kNN_k5"}
    assert rows[("Raw", "NA")] == pytest.approx(96.88) and rows[("PCA", "30")] == pytest.approx(97.55)


def test_mcnemar_p_values_recomputed_from_counts():
    rows = read("mcnemar.csv")
    for r in rows:
        b, c = int(r["n_a_only_correct"]), int(r["n_b_only_correct"])
        p = binomtest(b, b + c, 0.5).pvalue                      # exact McNemar test
        assert float(r["p_value"]) == pytest.approx(p, rel=1e-9, abs=1e-300), r["pair_id"]
    ordered = sorted(rows, key=lambda r: float(r["p_value"]))    # Holm step-down, with monotonicity
    m, running = len(rows), 0.0
    for i, r in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * float(r["p_value"])))
        assert float(r["p_holm"]) == pytest.approx(running, rel=1e-9), r["pair_id"]


def test_readme_table_matches_results():
    """Every accuracy in the README's results table appears in test_results.csv for the same configuration."""
    readme = open(os.path.join(ROOT, "README.md")).read()
    table = readme.split("<!-- results-table -->")[1]
    acc = {(r["pipeline"], r["classifier"]): round(float(r["test_acc_pct"]), 2)
           for r in read("test_results.csv") if r["is_selected"] == "1"}
    quoted = re.findall(r"^\| (Raw|PCA|LDA|PCA_LDA) \|.*?\| ([\d.]+)% \|.*?\| ([\d.]+)% \|", table, re.M)
    assert len(quoted) == 4
    for pipe, knn, maha in quoted:
        assert float(knn) == acc[(pipe, "kNN")] and float(maha) == acc[(pipe, "Maha")]


def test_fetch_rejects_a_wrong_file(tmp_path):
    import fetch_mnist
    f = tmp_path / "mnist.npz"
    f.write_bytes(b"not mnist")
    with pytest.raises(ValueError):
        fetch_mnist.verify(str(f))
