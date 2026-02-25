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


class XAICompareRunner:
    """
    Orchestrates a full 'publish run':
      - Registry autodiscovery
      - Model wrapping via registry
      - Predictions + top-k probabilities
      - XAI global + local explanations for one or more methods
      - Text index
      - Artifact persistence to a run directory

    Usage:
        runner = XAICompareRunner(
            model=my_model,
            x_test=x_test,
            y_test=y_test,
            raw_text=texts,
            class_names=class_names,
            run_dir="runs/_latest",
            config={"rows_limit_global": 200, "rows_limit_local": 200},
            save_model=True,
            model_type="sklearn",
            xai_methods=["shap_tree"],
            top_k_local=15,
        )
        runner.run()
    """

    def __init__(
        self,
        model,
        x_test,
        y_test: Optional[Sequence] = None,
        raw_text: Optional[Sequence] = None,
        class_names: Optional[Sequence[str]] = None,
        run_dir: str = "runs/_latest",
        config: Optional[Dict[str, Any]] = None,
        save_model: bool = True,
        *,
        # registry-driven API
        model_type: str = "sklearn",
        xai_methods: Optional[List[str]] = None,
        top_k_local: int = 15,
    ):
        # -----------------------------
        # Inputs and configuration
        # -----------------------------
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.raw_text = raw_text
        self.user_class_names = class_names
        self.run_path = pathlib.Path(run_dir)
        self.config = dict(config or {})
        self.save_model_flag = save_model

        self.model_type = model_type
        self.xai_methods = xai_methods if xai_methods is not None else ["shap_tree"]
        self.top_k_local = top_k_local

        # Derived / runtime attributes
        self.store: Optional[ArtifactStore] = None
        self.m = None  # wrapped model adapter
        self.df_pred: Optional[pd.DataFrame] = None
        self.df_text: Optional[pd.DataFrame] = None

        # Progress configuration
        pconf = dict(self.config.get("progress", {}))
        self.xai_desc = str(pconf.get("xai_desc", "Running XAI methods"))

        # Rows limits
        self.rows_limit_global = int(self.config.get("rows_limit_global", 200))
        self.rows_limit_local = int(self.config.get("rows_limit_local", 200))

    # -----------------------------
    # Public API
    # -----------------------------
    def run(self):
        '''
        Main entry point
        :return:
        '''
        self._ensure_registry()
        self._prepare_run_dir()
        self._wrap_model()
        self._save_model_if_requested()
        self._save_meta()

        self._compute_predictions()
        self._run_xai()
        self._build_text_index()

        self._write_common_artifacts()
        self._write_config_if_present()

        return {
            "run_dir": str(self.run_path),
            "methods": list(self.xai_methods),
            "n_samples": len(self.df_pred) if self.df_pred is not None else 0,
        }

    # -----------------------------
    # Steps
    # -----------------------------
    def _ensure_registry(self):
        autodiscover_adapters()

    def _prepare_run_dir(self):
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.run_path)

    def _wrap_model(self):
        model_adapter = get_model_adapter(self.model_type)  # noqa
        self.m = model_adapter(self.model, class_names=self.user_class_names)

    def _save_model_if_requested(self):
        if self.save_model_flag:
            joblib.dump(self.model, self.run_path / "model.joblib")

    def _save_meta(self):
        meta = {
            "run_id": str(uuid.uuid4()),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": self.model_type,
            "methods": list(self.xai_methods),
            "top_k_local": self.top_k_local,
            "class_names": self.m.class_names(),
            "feature_count": len(self.m.feature_names()),
            "xaicompare_version": __version__,
        }
        (self.run_path / META_INFO_FILENAME).write_text(
            json.dumps(make_json_safe(meta), indent=2)
        )

    def _compute_predictions(self):
        y_pred = self.m.predict(self.x_test)
        proba = self.m.predict_proba(self.x_test)

        df_pred = pd.DataFrame({
            "sample_id": np.arange(len(y_pred)),
            "y_pred": y_pred,
        })

        if self.y_test is not None:
            df_pred["y_true"] = list(self.y_test)

        if proba is not None:
            cls_names = (
                list(map(str, self.m.class_names()))
                if self.m.class_names() is not None
                else list(map(str, range(proba.shape[1])))
            )
            topk = 5
            top_idx = np.argsort(-proba, axis=1)[:, :topk]
            top_json = []
            for i in range(proba.shape[0]):
                row = {cls_names[c]: float(proba[i, c]) for c in top_idx[i]}
                top_json.append(json.dumps(row))
            df_pred["proba_topk_json"] = top_json

        self.df_pred = df_pred

    def _run_xai(self):
        assert self.store is not None, "ArtifactStore not initialized"
        # Outer loop: methods
        for mth in self._pbar(self.xai_methods, desc=self.xai_desc):
            Explainer = get_xai_adapter(mth)
            expl = Explainer(self.m, self.config)

            # Global importance
            gbar = self._pbar(total=1, desc=f"[{mth}] Global importance")
            mean_abs, feats = expl.global_importance(
                self.x_test, rows_limit=self.rows_limit_global
            )
            gbar.update(1)
            gbar.close()

            df_global = (
                pd.DataFrame({"feature": feats, "mean_abs_importance": mean_abs})
                .sort_values("mean_abs_importance", ascending=False)
            )
            self.store.write_parquet(f"{mth}_global.parquet", df_global)

            # Local explanations
            k = min(self.rows_limit_local, len(self.df_pred))
            records = []

            local_iter = self._pbar(range(k), desc=f"[{mth}] Local explanations")
            for i in local_iter:
                vals = expl.local_explanations(self.x_test[i:i+1])  # signed vector
                idx = np.argsort(np.abs(vals))[-self.top_k_local:][::-1]
                for j in idx:
                    records.append({
                        "sample_id": i,
                        "feature": feats[j],
                        "value": float(vals[j]),
                        "abs_value": float(abs(vals[j])),
                    })

            df_local = pd.DataFrame.from_records(records)
            self.store.write_parquet(f"{mth}_local.parquet", df_local)

    def _build_text_index(
        self,
    ):
        self.df_text = self.m.build_text_index(
            x_test=self.x_test,
            y_test=self.y_test,
            raw_text=self.raw_text,
            class_names=self.user_class_names,
        )
        assert self.store is not None
        self.store.write_parquet("text_index.parquet", self.df_text)

    def _write_common_artifacts(self):
        assert self.store is not None
        if self.df_pred is not None:
            self.store.write_parquet("predictions.parquet", self.df_pred)

    def _write_config_if_present(self):
        if self.config:
            (self.run_path / "config_used.yaml").write_text(self._to_yaml(self.config))

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _to_yaml(d):
        import yaml
        return yaml.safe_dump(d, sort_keys=False)

    @staticmethod
    def _pbar(iterable=None, total=None, desc=None):
        return tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            dynamic_ncols=True,
            mininterval=0.1,
        )
