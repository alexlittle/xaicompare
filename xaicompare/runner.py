# xai_kit/runner.py
from typing import Optional, Dict, Any, Sequence, List
import numpy as np
import pandas as pd
import json, pathlib, time, uuid
import joblib

from .artifacts import ArtifactStore
from .helpers import make_json_safe
from ._version import __version__

# NEW: use registries instead of hard imports
from .registry.model_registry import get_model_adapter
from .registry.xai_registry import get_xai_adapter


def publish_run(
    model,                         # trained pipeline/model
    X_test,                        # test features (text or vectorized)
    y_test: Optional[Sequence] = None,
    raw_text: Optional[Sequence] = None,
    class_names: Optional[Sequence[str]] = None,
    run_dir: str = "runs/_latest",
    config: Optional[Dict[str, Any]] = None,
    save_model: bool = True,

    # ---- API updates (with backward-compat) ----
    model_type: str = "sklearn",         # new: registry key for model adapter
    methods: Optional[List[str]] = None, # new: list of explainer keys
    method: str = "shap_tree",           # legacy single-method param; still supported

    top_k_local: int = 15,
):
    """
    Universal runner:
      - dynamic model adapter selection (model_type)
      - one or many XAI methods (methods / method)
      - preserves your existing file outputs and meta
    """
    cfg = config or {}
    run_path = pathlib.Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # Resolve requested explainer methods (backward compatible)
    xai_methods = methods if methods is not None else [method]

    # 1) Wrap model via registry
    ModelAdapter = get_model_adapter(model_type)
    m = ModelAdapter(model, class_names=class_names)

    # 2) Save model (optional) + meta
    if save_model:
        joblib.dump(model, run_path / "model.joblib")

    meta = {
        "run_id": str(uuid.uuid4()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "methods": xai_methods,
        "top_k_local": top_k_local,
        "class_names": m.class_names(),
        "feature_count": len(m.feature_names()),
        "xaicompare_version": __version__,  # keep your project name/version field
    }
    (run_path / "meta.json").write_text(json.dumps(make_json_safe(meta), indent=2))

    # 3) Predictions (preserve your probability-topk logic)
    y_pred = m.predict(X_test)
    proba  = m.predict_proba(X_test)

    df_pred = pd.DataFrame({
        "sample_id": np.arange(len(y_pred)),
        "y_pred": y_pred
    })

    if y_test is not None:
        df_pred["y_true"] = y_test

    if proba is not None:
        cls_names = [str(c) for c in m.class_names()] if m.class_names() is not None else [str(i) for i in range(proba.shape[1])]
        topk = 5
        top_idx = np.argsort(-proba, axis=1)[:, :topk]
        top_cols = []
        for i in range(proba.shape[0]):
            row = {cls_names[c]: float(proba[i, c]) for c in top_idx[i]}
            top_cols.append(json.dumps(row))
        df_pred["proba_topk_json"] = top_cols

    # 4) Run each requested XAI method via registry
    store = ArtifactStore(run_path)
    rows_limit_global = int(cfg.get("rows_limit_global", 200))
    rows_limit_local  = int(cfg.get("rows_limit_local", 200))

    for mth in xai_methods:
        Explainer = get_xai_adapter(mth)
        expl = Explainer(m, cfg)

        # Global
        mean_abs, feats = expl.global_importance(X_test, rows_limit=rows_limit_global)
        df_global = (
            pd.DataFrame({"feature": feats, "mean_abs_importance": mean_abs})
            .sort_values("mean_abs_importance", ascending=False)
        )
        store.write_parquet(f"{mth}_global.parquet", df_global)

        # Local (top-k)
        k = min(rows_limit_local, len(df_pred))
        records = []
        for i in range(k):
            vals = expl.local_explanations(X_test[i:i+1])  # signed vector
            idx = np.argsort(np.abs(vals))[-top_k_local:][::-1]
            for j in idx:
                records.append({
                    "sample_id": i,
                    "feature": feats[j],
                    "value": float(vals[j]),
                    "abs_value": float(abs(vals[j]))
                })
        df_local = pd.DataFrame.from_records(records)
        store.write_parquet(f"{mth}_local.parquet", df_local)

    # 5) Text index (optional)
    if raw_text is not None:
        df_text = pd.DataFrame({"sample_id": np.arange(len(raw_text)), "text": raw_text})
    else:
        df_text = pd.DataFrame({"sample_id": df_pred["sample_id"]})

    # 6) Write artifacts common to all runs
    store.write_parquet("predictions.parquet", df_pred)
    if config:
        (run_path / "config_used.yaml").write_text(_to_yaml(config))


def _to_yaml(d):
    import yaml
    return yaml.safe_dump(d, sort_keys=False)