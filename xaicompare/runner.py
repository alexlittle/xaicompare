# xaicompare/runner.py
from typing import Optional, Dict, Any, Sequence, List
import numpy as np
import pandas as pd
import json, pathlib, time, uuid
import joblib

from tqdm.auto import tqdm

from xaicompare.artifacts import ArtifactStore
from xaicompare.helpers import make_json_safe
from xaicompare._version import __version__

from xaicompare.registry.model_registry import get_model_adapter
from xaicompare.registry.xai_registry import get_xai_adapter
from xaicompare.registry.autodiscover import autodiscover_adapters

from xaicompare.consts import META_INFO_FILENAME

def publish_run(
    model,
    X_test,
    y_test: Optional[Sequence] = None,
    raw_text: Optional[Sequence] = None,
    class_names: Optional[Sequence[str]] = None,
    run_dir: str = "runs/_latest",
    config: Optional[Dict[str, Any]] = {},
    save_model: bool = True,

    # registry-driven API
    model_type: str = "sklearn",
    xai_methods: Optional[List[str]] = ['shap_tree'],
    top_k_local: int = 15,
):
    """
    Universal runner:
      - model adapter resolution (model_type)
      - XAI adapter resolution (xai_methods)
      - consistent artifact generation
      - tqdm progress bars for user feedback
    """

    # ------------------------------------------------------------------
    # 0) Ensure registry is populated (recursive autodiscovery)
    # ------------------------------------------------------------------
    autodiscover_adapters()

    run_path = pathlib.Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # Progress configuration (can be overridden via config["progress"])
    pconf = dict(config.get("progress", {}))
    xai_desc    = str(pconf.get("xai_desc", "Running XAI methods"))

    # Helper to optionally wrap iterables with tqdm


    # ------------------------------------------------------------------
    # 1) Wrap model using registry
    # ------------------------------------------------------------------
    model_adapter = get_model_adapter(model_type) #noqa
    m = model_adapter(model, class_names=class_names)

    # ------------------------------------------------------------------
    # 2) Save model + meta
    # ------------------------------------------------------------------
    save_model_file(save_model, model, run_path)
    save_meta_file(model_type, xai_methods, top_k_local, m, run_path)

    # ------------------------------------------------------------------
    # 3) Predictions (probabilities + top-k)
    # ------------------------------------------------------------------
    y_pred = m.predict(X_test)
    proba  = m.predict_proba(X_test)

    df_pred = pd.DataFrame({
        "sample_id": np.arange(len(y_pred)),
        "y_pred": y_pred
    })

    if y_test is not None:
        df_pred["y_true"] = list(y_test)

    if proba is not None:
        cls_names = (
            list(map(str, m.class_names()))
            if m.class_names() is not None
            else list(map(str, range(proba.shape[1])))
        )
        topk = 5
        top_idx = np.argsort(-proba, axis=1)[:, :topk]
        top_json = []
        for i in range(proba.shape[0]):
            row = {cls_names[c]: float(proba[i, c]) for c in top_idx[i]}
            top_json.append(json.dumps(row))
        df_pred["proba_topk_json"] = top_json

    # ------------------------------------------------------------------
    # 4) XAI explanations — with progress bars
    # ------------------------------------------------------------------
    store = ArtifactStore(run_path)
    rows_limit_global = int(config.get("rows_limit_global", 200))
    rows_limit_local  = int(config.get("rows_limit_local", 200))

    # Outer bar over methods
    for mth in pbar(xai_methods, desc=xai_desc):
        Explainer = get_xai_adapter(mth)
        expl = Explainer(m, config)

        # ---- Global importance ----
        # Show a one-step indicator (useful when global explainer is heavy)
        gbar = pbar(total=1, desc=f"[{mth}] Global importance")


        mean_abs, feats = expl.global_importance(X_test, rows_limit=rows_limit_global)

        gbar.update(1)
        gbar.close()

        df_global = (
            pd.DataFrame({"feature": feats, "mean_abs_importance": mean_abs})
            .sort_values("mean_abs_importance", ascending=False)
        )
        store.write_parquet(f"{mth}_global.parquet", df_global)

        # ---- Local explanations (top-k per sample) ----
        k = min(rows_limit_local, len(df_pred))
        records = []

        # Wrap the local loop with a progress bar
        local_iter = pbar(range(k), desc=f"[{mth}] Local explanations")
        for i in local_iter:
            vals = expl.local_explanations(X_test[i:i+1])  # signed vector
            idx = np.argsort(np.abs(vals))[-top_k_local:][::-1]
            for j in idx:
                records.append({
                    "sample_id": i,
                    "feature": feats[j],
                    "value": float(vals[j]),
                    "abs_value": float(abs(vals[j])),
                })

        df_local = pd.DataFrame.from_records(records)
        store.write_parquet(f"{mth}_local.parquet", df_local)

    # ------------------------------------------------------------------
    # 5) Text index (adapter-driven with fallback)
    # ------------------------------------------------------------------
    df_text = m.build_text_index(
        X_test=X_test,
        y_test=y_test,
        raw_text=raw_text,
        class_names=class_names
    )

    store.write_parquet("text_index.parquet", df_text)

    # ------------------------------------------------------------------
    # 6) Common artifacts
    # ------------------------------------------------------------------
    store.write_parquet("predictions.parquet", df_pred)

    if config:
        (run_path / "config_used.yaml").write_text(_to_yaml(config))


def _to_yaml(d):
    import yaml
    return yaml.safe_dump(d, sort_keys=False)

def pbar(iterable=None, total=None, desc=None):
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        mininterval=0.1,
    )

def save_model_file(save_model, model, run_path):
    if save_model:
        joblib.dump(model, run_path / "model.joblib")

def save_meta_file(model_type, xai_methods, top_k_local, m, run_path):
    meta = {
        "run_id": str(uuid.uuid4()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "methods": xai_methods,
        "top_k_local": top_k_local,
        "class_names": m.class_names(),
        "feature_count": len(m.feature_names()),
        "xaicompare_version": __version__,
    }
    (run_path / META_INFO_FILENAME).write_text(json.dumps(make_json_safe(meta), indent=2))