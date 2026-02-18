# xai_kit/adapters/explainer_shap_tree.py
from .explainer_base import ExplainerAdapter
import numpy as np
import shap
from typing import Dict, Any
from scipy.sparse import spmatrix

class ShapTreeExplainerAdapter(ExplainerAdapter):
    """Sparse-friendly SHAP for tree models."""
    def __init__(self, model_adapter, config: Dict[str, Any]):
        self.m = model_adapter
        self.config = {"batch_size": 2, "approximate": True, "check_additivity": False, **(config or {})}
        # Grab raw model for SHAP
        self._clf = self.m.pipeline.named_steps.get("xgb", None) if hasattr(self.m, "pipeline") else None
        self._explainer = shap.TreeExplainer(self._clf, feature_perturbation="tree_path_dependent")

    def global_importance(self, X, rows_limit=200):
        """Stream mean|SHAP| across a subset to keep memory low."""
        import math
        bs = self.config["batch_size"]
        approx = self.config["approximate"]
        check = self.config["check_additivity"]
        feats = len(self.m.feature_names())
        acc = np.zeros(feats, dtype=np.float64)
        seen = 0

        N = X.shape[0] if hasattr(X, "shape") else len(X)
        limit = min(rows_limit, N)

        for start in range(0, limit, bs):
            end = min(limit, start+bs)
            sv = self._explainer.shap_values(X[start:end], approximate=approx, check_additivity=check)
            abs_2d = self._normalize_to_abs_2d(sv)  # (batch, n_features)
            acc += abs_2d.sum(axis=0)
            seen += abs_2d.shape[0]

        return (acc / max(seen, 1)), self.m.feature_names()

    def local_explanations(self, X_row):
        sv = self._explainer.shap_values(X_row, approximate=self.config["approximate"], check_additivity=self.config["check_additivity"])
        abs_2d = self._normalize_to_abs_2d(sv)  # (1, n_features)
        vals = abs_2d[0]
        raw  = self._to_1d(sv)  # signed values for top-k
        return raw  # length = n_features

    @staticmethod
    def _normalize_to_abs_2d(sv):
        sv = sv if isinstance(sv, list) else np.asarray(sv)
        if isinstance(sv, list):
            parts = [np.abs(s[:, :-1]) for s in sv]  # drop bias
            return np.sum(parts, axis=0)
        if sv.ndim == 2:   # (batch, n_features+1)
            return np.abs(sv[:, :-1])
        if sv.ndim == 3:   # (batch, n_features+1, n_classes)
            return np.abs(sv[:, :-1, :]).sum(axis=2)
        raise ValueError(f"Unexpected shape: {getattr(sv, 'shape', None)}")

    @staticmethod
    def _to_1d(sv):
        # Return signed vector aggregated over classes
        sv = sv if isinstance(sv, list) else np.asarray(sv)
        if isinstance(sv, list):
            parts = [s[0, :-1] for s in sv]
            return np.sum(parts, axis=0)
        if sv.ndim == 2:
            return sv[0, :-1]
        if sv.ndim == 3:
            return sv[0, :-1, :].sum(axis=1)
        raise ValueError(f"Unexpected shape: {getattr(sv, 'shape', None)}")