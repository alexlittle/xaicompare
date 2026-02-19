# xai_kit/adapters/model_sklearn.py
from .model_base import ModelAdapter, ArrayLike
import numpy as np
from expailens.registry.model_registry import register_model


@register_model("sklearn")
class SklearnPipelineAdapter(ModelAdapter):
    def __init__(self, pipeline, class_names=None):
        self.pipeline = pipeline
        self._tfidf = getattr(pipeline, "named_steps", {}).get("tfidf", None)
        self._clf   = getattr(pipeline, "named_steps", {}).get("xgb", None)
        self._class_names = class_names

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        return None

    def feature_names(self):
        if self._tfidf is not None and hasattr(self._tfidf, "get_feature_names_out"):
            return list(self._tfidf.get_feature_names_out())
        # Fallback: model-based indexing
        n = int(self._clf.get_booster().attr("num_feature")) if self._clf else 0
        return [f"f{i}" for i in range(n)]

    def class_names(self):
        names = []
        if self._clf is not None and hasattr(self._clf, "classes_"):
            names = list(self._clf.classes_)
        elif self._class_names is not None:
            names = list(self._class_names)
        # ensure plain str (handles numpy scalars too)
        return [str(x) for x in names]


def is_sparse_input(self) -> bool:
        return True