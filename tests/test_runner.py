# tests/test_runner.py
import json
import numpy as np
import pandas as pd
import pathlib

import xaicompare.runner as runner_mod
from xaicompare.runner import XAICompareRunner


# -----------------------------
# Test doubles (stubs/fakes)
# -----------------------------

class FakeArtifactStore:
    """Write DataFrames as JSON (not parquet) to keep the test environment simple."""
    def __init__(self, base_dir):
        self.base = pathlib.Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def write_parquet(self, filename, df: pd.DataFrame):
        p = self.base / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        # Store as JSON so we can easily read/validate content in tests.
        p.write_text(df.to_json(orient="records"), encoding="utf-8")


class FakeModelAdapterProba:
    """Model adapter that exposes predict + predict_proba + supporting methods."""
    def __init__(self, model, class_names=None):
        self._model = model
        self._class_names = list(class_names) if class_names is not None else ["A", "B", "C"]

    def class_names(self):
        return self._class_names

    def feature_names(self):
        # Keep simple: two features
        return ["f0", "f1"]

    def predict(self, X):
        # Deterministic labels cycling over classes
        n = len(X)
        return np.array([i % len(self._class_names) for i in range(n)])

    def predict_proba(self, X):
        # Produce a distribution with a clear top-1 for each row
        n = len(X)
        c = len(self._class_names)
        proba = np.full((n, c), 0.1, dtype=float)
        for i in range(n):
            proba[i, i % c] = 0.7
            # Normalize just in case:
            proba[i] = proba[i] / proba[i].sum()
        return proba

    def build_text_index(self, X_test, y_test=None, raw_text=None, class_names=None):
        n = len(X_test)
        df = pd.DataFrame({"sample_id": np.arange(n)})
        if y_test is not None:
            df["y_true"] = list(y_test)
        if raw_text is not None:
            df["raw_text"] = list(raw_text)
        return df


class FakeModelAdapterNoProba(FakeModelAdapterProba):
    def predict_proba(self, X):
        return None


class FakeXAIAdapter:
    """XAI adapter that returns stable global and local explanations."""
    def __init__(self, model_adapter, cfg):
        self.m = model_adapter
        self.cfg = cfg

    def global_importance(self, X, rows_limit=200):
        feats = self.m.feature_names()
        # Make f1 more important than f0 to test sorting
        mean_abs = np.array([0.4, 0.6], dtype=float)
        return mean_abs, feats

    def local_explanations(self, X_row):
        # Signed values; abs determines top-k
        return np.array([0.2, -0.5], dtype=float)


class FakeXAIAdapterTop1(FakeXAIAdapter):
    """Variant to test top_k_local=1 selection deterministically."""
    def local_explanations(self, X_row):
        # Make f1 the top by absolute value
        return np.array([0.1, 0.9], dtype=float)


# -----------------------------
# Helpers
# -----------------------------

def read_json_records(path: pathlib.Path):
    """Read a JSON array (written by FakeArtifactStore) from a .parquet path."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return json.loads(text)


# -----------------------------
# Tests
# -----------------------------

def test_publish_run_with_proba_creates_expected_artifacts(tmp_path, monkeypatch):
    # Arrange: patch all externals used by runner
    autodiscover_calls = {"n": 0}
    def fake_autodiscover():
        autodiscover_calls["n"] += 1

    monkeypatch.setattr(runner_mod, "autodiscover_adapters", fake_autodiscover)
    monkeypatch.setattr(runner_mod, "ArtifactStore", FakeArtifactStore)
    monkeypatch.setattr(runner_mod, "make_json_safe", lambda x: x)
    monkeypatch.setattr(runner_mod, "__version__", "0.0.0-test")

    # Provide model + XAI registry returns
    monkeypatch.setattr(runner_mod, "get_model_adapter", lambda model_type: FakeModelAdapterProba)
    monkeypatch.setattr(runner_mod, "get_xai_adapter", lambda name: FakeXAIAdapter)

    # Avoid writing a real joblib file (but still verify call in a different test)
    save_calls = []
    def fake_dump(model, path):
        save_calls.append(path)
    monkeypatch.setattr(runner_mod.joblib, "dump", fake_dump)

    # Inputs
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    y = [0, 1, 2]
    raw = ["t0", "t1", "t2"]
    run_dir = tmp_path / "runs" / "latest"

    config = {
        "progress": {"enabled": False},
        "rows_limit_global": 200,
        "rows_limit_local": 200,
    }

    # Act
    runner = XAICompareRunner(
        model=object(),
        X_test=X,
        y_test=y,
        raw_text=raw,
        class_names=["A", "B", "C"],
        run_dir=str(run_dir),
        config=config,
        save_model=True,
        model_type="sklearn",
        xai_methods=["fake_xai"],
        top_k_local=15,
    )
    runner.run()

    # Assert: autodiscover was called once
    assert autodiscover_calls["n"] == 1

    # Assert: meta.json created with expected content
    meta_path = run_dir / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["model_type"] == "sklearn"
    assert meta["methods"] == ["fake_xai"]
    assert meta["class_names"] == ["A", "B", "C"]
    assert meta["feature_count"] == 2
    assert meta["xaicompare_version"] == "0.0.0-test"

    # Assert: predictions artifact exists and has expected fields
    preds_path = run_dir / "predictions.parquet"
    assert preds_path.exists()
    preds = read_json_records(preds_path)
    assert len(preds) == 3
    # Basic columns
    for i, row in enumerate(preds):
        assert row["sample_id"] == i
        assert "y_pred" in row
        assert row.get("y_true") == y[i]
        # proba_topk_json should exist and be a valid JSON string
        topk = json.loads(row["proba_topk_json"])
        assert set(topk.keys()).issubset({"A", "B", "C"})
        # Top-probability should be reasonable
        assert max(topk.values()) <= 1.0

    # Assert: global explanations written and sorted by importance desc (f1 before f0)
    global_path = run_dir / "fake_xai_global.parquet"
    assert global_path.exists()
    glob = read_json_records(global_path)
    assert [r["feature"] for r in glob] == ["f1", "f0"]
    assert glob[0]["mean_abs_importance"] >= glob[1]["mean_abs_importance"]

    # Assert: local explanations written for each sample with 2 features each (since top_k_local=15 and 2 features total)
    local_path = run_dir / "fake_xai_local.parquet"
    assert local_path.exists()
    local = read_json_records(local_path)
    # 3 samples * 2 features = 6 rows
    assert len(local) == 6
    # Check schema and values
    for row in local:
        assert row["feature"] in {"f0", "f1"}
        assert abs(row["abs_value"] - abs(row["value"])) < 1e-12

    # Assert: text index written and includes raw_text + y_true if provided
    text_path = run_dir / "text_index.parquet"
    assert text_path.exists()
    trows = read_json_records(text_path)
    assert len(trows) == 3
    assert trows[0]["sample_id"] == 0
    assert "raw_text" in trows[0]
    assert "y_true" in trows[0]

    # Assert: config_used.yaml written since config was passed
    cfg_path = run_dir / "config_used.yaml"
    assert cfg_path.exists()

    # Assert: model saved (we patched joblib.dump)
    assert any(str(p).endswith("model.joblib") for p in save_calls)


def test_publish_run_without_proba_drops_proba_column(tmp_path, monkeypatch):
    # Patch externals
    monkeypatch.setattr(runner_mod, "autodiscover_adapters", lambda: None)
    monkeypatch.setattr(runner_mod, "ArtifactStore", FakeArtifactStore)
    monkeypatch.setattr(runner_mod, "make_json_safe", lambda x: x)
    monkeypatch.setattr(runner_mod, "__version__", "0.0.0-test")

    # No proba adapter
    monkeypatch.setattr(runner_mod, "get_model_adapter", lambda model_type: FakeModelAdapterNoProba)
    monkeypatch.setattr(runner_mod, "get_xai_adapter", lambda name: FakeXAIAdapter)

    # Avoid real joblib writes
    monkeypatch.setattr(runner_mod.joblib, "dump", lambda model, path: None)

    X = np.array([[1, 2], [3, 4]], dtype=float)
    run_dir = tmp_path / "run_noproba"
    config = {"progress": {"enabled": False}}

    runner = XAICompareRunner(
        model=object(),
        X_test=X,
        y_test=None,
        raw_text=None,
        class_names=["A", "B", "C"],
        run_dir=str(run_dir),
        config=config,
        save_model=False,
        model_type="sklearn",
        xai_methods=["fake_xai"],
        top_k_local=5,
    )
    runner.run()

    preds_path = run_dir / "predictions.parquet"
    assert preds_path.exists()
    preds = read_json_records(preds_path)
    assert len(preds) == 2
    # y_true shouldn't exist; proba_topk_json shouldn't exist
    assert "y_true" not in preds[0]
    assert "proba_topk_json" not in preds[0]


def test_publish_run_respects_top_k_and_row_limits(tmp_path, monkeypatch):
    # Patch externals
    monkeypatch.setattr(runner_mod, "autodiscover_adapters", lambda: None)
    monkeypatch.setattr(runner_mod, "ArtifactStore", FakeArtifactStore)
    monkeypatch.setattr(runner_mod, "make_json_safe", lambda x: x)
    monkeypatch.setattr(runner_mod, "__version__", "0.0.0-test")

    # Use proba adapter (doesn't matter for local) + top-1 local xai adapter
    monkeypatch.setattr(runner_mod, "get_model_adapter", lambda model_type: FakeModelAdapterProba)
    monkeypatch.setattr(runner_mod, "get_xai_adapter", lambda name: FakeXAIAdapterTop1)

    # Avoid real joblib writes
    monkeypatch.setattr(runner_mod.joblib, "dump", lambda model, path: None)

    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    run_dir = tmp_path / "run_limits"
    config = {
        "progress": {"enabled": False},
        "rows_limit_local": 1,  # Only explain first sample
    }

    runner = XAICompareRunner(
        model=object(),
        X_test=X,
        y_test=None,
        raw_text=None,
        class_names=["A", "B", "C"],
        run_dir=str(run_dir),
        config=config,
        save_model=False,
        model_type="sklearn",
        xai_methods=["fake_xai"],
        top_k_local=1,
    )
    runner.run()

    local_path = run_dir / "fake_xai_local.parquet"
    assert local_path.exists()
    local = read_json_records(local_path)
    # rows_limit_local=1 and top_k_local=1 -> exactly 1 row
    assert len(local) == 1
    # The top feature by abs value is f1 in our FakeXAIAdapterTop1
    assert local[0]["feature"] == "f1"
    assert abs(local[0]["abs_value"] - abs(local[0]["value"])) < 1e-12


def test_publish_run_does_not_save_model_when_flag_false(tmp_path, monkeypatch):
    # Patch externals
    monkeypatch.setattr(runner_mod, "autodiscover_adapters", lambda: None)
    monkeypatch.setattr(runner_mod, "ArtifactStore", FakeArtifactStore)
    monkeypatch.setattr(runner_mod, "make_json_safe", lambda x: x)
    monkeypatch.setattr(runner_mod, "__version__", "0.0.0-test")
    monkeypatch.setattr(runner_mod, "get_model_adapter", lambda model_type: FakeModelAdapterProba)
    monkeypatch.setattr(runner_mod, "get_xai_adapter", lambda name: FakeXAIAdapter)

    # Spy on joblib.dump
    calls = {"n": 0}
    def fake_dump(model, path):
        calls["n"] += 1
    monkeypatch.setattr(runner_mod.joblib, "dump", fake_dump)

    X = np.array([[0, 1]], dtype=float)
    run_dir = tmp_path / "run_no_save"
    config = {"progress": {"enabled": False}}

    runner = XAICompareRunner(
        model=object(),
        X_test=X,
        y_test=None,
        raw_text=None,
        class_names=["A", "B", "C"],
        run_dir=str(run_dir),
        config=config,
        save_model=False,  # <-- ensure no save
        model_type="sklearn",
        xai_methods=["fake_xai"],
        top_k_local=5,
    )
    runner.run()


    assert calls["n"] == 0, "joblib.dump should not be called when save_model=False"