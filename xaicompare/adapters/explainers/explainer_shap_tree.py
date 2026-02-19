# expailens/adapters/explainers/explainer_shap_tree.py


import numpy as np
import shap
from scipy.sparse import spmatrix
from typing import Any, Dict, List, Tuple

from .explainer_base import ExplainerAdapter
from expailens.registry.xai_registry import register_xai

@register_xai("shap_tree")
class ShapTreeExplainerAdapter(ExplainerAdapter):
    """
    SHAP Tree Explainer with:
      - Safe sparse handling
      - Forced vectorization
      - Guaranteed 2-D input to SHAP
      - Batch processing to avoid memory errors
      - Multi-class normalization
    """

    def __init__(self, model_adapter, config: Dict[str, Any]):
        super().__init__(model_adapter, config)

        self.batch_size = int(self.config.get("batch_size", 2))
        self.approx = bool(self.config.get("approximate", True))
        self.check_add = bool(self.config.get("check_additivity", False))

        # Underlying model (XGBClassifier inside pipeline)
        clf = model_adapter.pipeline.named_steps.get("xgb", None)
        if clf is None:
            raise ValueError("Pipeline must contain an 'xgb' step for ShapTreeExplainerAdapter.")

        # Sparse-friendly SHAP explainer
        self.explainer = shap.TreeExplainer(
            clf,
            feature_perturbation="tree_path_dependent"
        )

        # Cached TF-IDF vectorizer
        self.vectorizer = model_adapter.pipeline.named_steps.get("tfidf", None)
        if self.vectorizer is None:
            raise ValueError("Pipeline must contain a 'tfidf' vectorizer.")

        self.feature_names_list = model_adapter.feature_names()


    # ----------------------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------------------

    def _ensure_vectorized(self, X: Any):
        """
        Guarantee X is a (batch, n_features) 2D sparse matrix.
        Also supports:
          - raw strings
          - lists of strings
          - numpy object arrays
          - pandas series of strings
          - already-sparse matrices
        """
        # 1) Already sparse → ensure 2-D structure
        if isinstance(X, spmatrix):
            return X

        # 2) Convert 1-D raw sample → list → vectorize
        # X might be a string or 1-element array
        if isinstance(X, str):
            return self.vectorizer.transform([X])

        # 3) Convert list-of-strings
        if isinstance(X, list):
            return self.vectorizer.transform(X)

        # 4) Numpy array of dtype object (raw text)
        if isinstance(X, np.ndarray) and X.dtype == object:
            # If it's 1-D, wrap into a 2-D
            if X.ndim == 1:
                X = X.reshape(-1)
                return self.vectorizer.transform(list(X))
            return self.vectorizer.transform(list(X))

        # 5) Pandas series of strings
        if hasattr(X, "dtype") and str(X.dtype) == "object":
            return self.vectorizer.transform(X.astype(str).tolist())

        # 6) Dense numeric array → ensure 2-D
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X.reshape(1, -1)
            return X

        # 7) Anything else: try to force list-based transform
        return self.vectorizer.transform(list(X))


    def _normalize_to_abs_2d(self, sv: Any) -> np.ndarray:
        """
        Convert SHAP output to shape (batch, n_features) as abs values.
        """
        sv = sv if isinstance(sv, list) else np.asarray(sv)

        # Case 1: list of (batch, feats+1)
        if isinstance(sv, list):
            parts = [np.abs(s[:, :-1]) for s in sv]
            return np.sum(parts, axis=0)

        # Case 2: (batch, feats+1)
        if sv.ndim == 2:
            return np.abs(sv[:, :-1])

        # Case 3: (batch, feats+1, classes)
        if sv.ndim == 3:
            return np.abs(sv[:, :-1, :]).sum(axis=2)

        raise ValueError(f"Unexpected SHAP shape: {sv.shape}")


    def _normalize_to_signed_1d(self, sv: Any) -> np.ndarray:
        """
        Normalize a SHAP output for a single sample into a signed (n_features,) vector.
        """
        sv = sv if isinstance(sv, list) else np.asarray(sv)

        # Case 1: list (multi-class)
        if isinstance(sv, list):
            parts = [s[0, :-1] for s in sv]  # remove bias
            return np.sum(parts, axis=0)

        # Case 2: (1, feats+1)
        if sv.ndim == 2:
            return sv[0, :-1]

        # Case 3: (1, feats+1, classes)
        if sv.ndim == 3:
            return sv[0, :-1, :].sum(axis=1)

        raise ValueError(f"Unexpected SHAP shape for local explanation: {sv.shape}")


    # ----------------------------------------------------------------------
    # GLOBAL IMPORTANCE
    # ----------------------------------------------------------------------

    def global_importance(self, X, rows_limit: int = 200) -> Tuple[np.ndarray, List[str]]:
        """
        Compute mean|SHAP| across first rows_limit samples safely.
        """
        # Extract raw values from numpy/pandas objects
        if hasattr(X, "values"):
            X = X.values
        if isinstance(X, np.ndarray) and X.ndim == 0:
            X = X.reshape(1)

        N = len(X)
        limit = min(rows_limit, N)

        acc = None
        seen = 0

        for start in range(0, limit, self.batch_size):
            end = min(limit, start + self.batch_size)

            X_raw = X[start:end]

            # Guarantee vectorized matrix
            X_vec = self._ensure_vectorized(X_raw)

            sv = self.explainer.shap_values(
                X_vec,
                approximate=self.approx,
                check_additivity=self.check_add
            )

            abs_2d = self._normalize_to_abs_2d(sv)

            if acc is None:
                acc = np.zeros(abs_2d.shape[1], dtype=np.float64)

            acc += abs_2d.sum(axis=0)
            seen += abs_2d.shape[0]

        if seen == 0:
            return np.zeros(len(self.feature_names_list)), self.feature_names_list

        mean_abs = acc / seen

        n_vec = len(self.feature_names_list)
        n_shap = mean_abs.shape[0]

        if n_shap != n_vec:
            min_len = min(n_shap, n_vec)
            print(f"[WARN] Feature mismatch: SHAP={n_shap}, TFIDF={n_vec}. "
                  f"Truncating both to {min_len}.")
            mean_abs = mean_abs[:min_len]
            feats = self.feature_names_list[:min_len]
        else:
            feats = self.feature_names_list
        # -----------------------

        return mean_abs, feats


    # ----------------------------------------------------------------------
    # LOCAL EXPLANATIONS
    # ----------------------------------------------------------------------

    def local_explanations(self, X_row) -> np.ndarray:
        """
        Compute SHAP for a single example → return signed vector.
        """
        # Force shape (1, raw_item)
        if isinstance(X_row, str):
            X_row = [X_row]

        if isinstance(X_row, np.ndarray) and X_row.ndim == 0:
            X_row = np.array([X_row])

        # Vectorize
        X_vec = self._ensure_vectorized(X_row)

        sv = self.explainer.shap_values(
            X_vec,
            approximate=self.approx,
            check_additivity=self.check_add
        )

        return self._normalize_to_signed_1d(sv)