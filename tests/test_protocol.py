"""End-to-end checks of the v2 experiment on synthetic data: the protocol, not the MNIST numbers."""
import csv
import sys

import pytest

import main_v2


def run_main(monkeypatch, tmp_path, fake, *args):
    """Run main_v2.main() with synthetic data, writing into tmp_path instead of the repo."""
    events = []
    real_select = main_v2.select

    def select(rows):
        events.append("select")
        return real_select(rows)

    def load_test():
        events.append("load_test")
        return fake["test"]

    monkeypatch.setattr(main_v2, "HERE", str(tmp_path))
    monkeypatch.setattr(main_v2, "load_train", lambda: fake["train"])
    monkeypatch.setattr(main_v2, "load_test", load_test)
    monkeypatch.setattr(main_v2, "select", select)
    monkeypatch.setattr(sys, "argv", ["main_v2.py", "--cv-subsample", "400", "--folds", "3", *args])
    main_v2.main()
    return events


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_test_set_is_loaded_only_after_selection(monkeypatch, tmp_path, fake_mnist):
    events = run_main(monkeypatch, tmp_path, fake_mnist)
    assert "load_test" in events and "select" in events
    assert events.index("select") < events.index("load_test")      # selection never sees the test set
    assert events.count("load_test") == 1


def test_outputs_and_selected_configs_are_sane(monkeypatch, tmp_path, fake_mnist):
    run_main(monkeypatch, tmp_path, fake_mnist)
    res = tmp_path / "results"
    for name in ("cv_results.csv", "selected_config.csv", "test_results.csv", "mcnemar.csv", "run_log.txt"):
        assert (res / name).exists(), name
    cv = read(res / "cv_results.csv")
    assert {r["pipeline"] for r in cv} == {"Raw", "PCA", "LDA", "PCA_LDA"}
    assert {"kNN", "Maha"} <= {r["classifier"] for r in cv}
    assert all(int(r["lda_dim"]) <= 9 for r in cv if r["lda_dim"] != "NA")      # at most c - 1 = 9 LDA axes
    test_rows = read(res / "test_results.csv")
    selected = [r for r in test_rows if r["is_selected"] == "1"]
    assert {r["pipeline"] for r in selected} == {"Raw", "PCA", "LDA", "PCA_LDA"}
    for r in test_rows:
        assert abs(float(r["test_acc_pct"]) + float(r["test_err_pct"]) - 100) < 1e-9
    assert max(float(r["test_acc_pct"]) for r in selected) > 85        # the synthetic classes are separable

def test_same_seed_same_cv_table(monkeypatch, tmp_path, fake_mnist):
    run_main(monkeypatch, tmp_path / "a", fake_mnist)
    run_main(monkeypatch, tmp_path / "b", fake_mnist)
    assert read(tmp_path / "a/results/cv_results.csv") == read(tmp_path / "b/results/cv_results.csv")
