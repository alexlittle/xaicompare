# xai_kit/runner.py
from typing import Optional, Dict, Any, Sequence
import numpy as np
import pandas as pd
import json, pathlib, time, uuid
import joblib
from .artifacts import ArtifactStore
from .adapters.models.model_sklearn import SklearnPipelineAdapter
from .adapters.explainers.explainer_shap_tree import ShapTreeExplainerAdapter
from .helpers import make_json_safe
from ._version import __version__

def publish_run(
    model,                         # trained pipeline/model
    X_test,                        # test features (text or vectorized)
    y_test: Optional[Sequence]=None,
    raw_text: Optional[Sequence]=None,
    class_names: Optional[Sequence[str]]=None,
    run_dir: str="runs/_latest",
    config: Optional[Dict[str, Any]]=None,
    save_model: bool=True,
    method: str="shap_tree",
    top_k_local: int=15,
):
    run_path = pathlib.Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # 1) Wrap model
    m = SklearnPipelineAdapter(model, class_names=class_names)

    # 2) Save model (optional) + meta
    if save_model:
        joblib.dump(model, run_path / "model.joblib")
    meta = {
        "run_id": str(uuid.uuid4()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": method,
        "top_k_local": top_k_local,
        "class_names": m.class_names(),
        "feature_count": len(m.feature_names()),
        "expailens_version": __version__,
    }
    meta_safe = make_json_safe(meta)
    (run_path / "meta.json").write_text(json.dumps(meta_safe, indent=2))

    # 3) Predictions
    y_pred = m.predict(X_test)
    proba  = m.predict_proba(X_test)
    df_pred = pd.DataFrame({
        "sample_id": np.arange(len(y_pred)),
        "y_pred": y_pred
    })
    if y_test is not None:
        df_pred["y_true"] = y_test
    if proba is not None:
        # keep top-5 probs as JSON to keep file size small
        cls_names = [str(c) for c in m.class_names()]  # ensure pure Python str

        topk = 5
        top_idx = np.argsort(-proba, axis=1)[:, :topk]  # (n_samples, topk)

        top_cols = []
        for i in range(proba.shape[0]):
            row = {cls_names[c]: float(proba[i, c]) for c in top_idx[i]}
            top_cols.append(json.dumps(row))  # keys are str, values are float

        df_pred["proba_topk_json"] = top_cols

    # 4) Global importance (streamed)
    expl = ShapTreeExplainerAdapter(m, config or {})
    mean_abs, feats = expl.global_importance(
        X_test, rows_limit=(config or {}).get("rows_limit_global", 200)
    )
    df_global = pd.DataFrame({"feature": feats, "mean_abs_importance": mean_abs}).sort_values("mean_abs_importance", ascending=False)

    # 5) Local explanations (top-k per row)
    rows_limit_local = (config or {}).get("rows_limit_local", 200)
    k = min(rows_limit_local, len(df_pred))
    records = []
    for i in range(k):
        vals = expl.local_explanations(X_test[i:i+1])  # signed vector
        idx = np.argsort(np.abs(vals))[-top_k_local:][::-1]
        for j in idx:
            records.append({
                "sample_id": i,
                "feature": feats[j],
                "shap_value": float(vals[j]),
                "abs_value": float(abs(vals[j]))
            })
    df_local = pd.DataFrame.from_records(records)

    # 6) Text index (optional)
    if raw_text is not None:
        df_text = pd.DataFrame({"sample_id": np.arange(len(raw_text)), "text": raw_text})
    else:
        df_text = pd.DataFrame({"sample_id": df_pred["sample_id"]})

    # 7) Write artifacts
    store = ArtifactStore(run_path)
    store.write_parquet("predictions.parquet", df_pred)
    store.write_parquet("shap_global.parquet", df_global)
    store.write_parquet("shap_local.parquet", df_local)
    store.write_parquet("text_index.parquet", df_text)
    if config:
        (run_path / "config_used.yaml").write_text(_to_yaml(config))

def _to_yaml(d):
    import yaml
    return yaml.safe_dump(d, sort_keys=False)