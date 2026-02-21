# xaicompare/adapters/explainer_base.py
# SPDX-License-Identifier: MIT
# Minimal abstract interface for explanation backends (SHAP, LIME, etc.)

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union, Optional

import numpy as np

try:
    # Optional, used only for type hints; don't require scipy at import time.
    from scipy.sparse import spmatrix  # type: ignore
    ArrayLike = Union[np.ndarray, "spmatrix", Sequence[Any]]
except Exception:  # pragma: no cover
    ArrayLike = Union[np.ndarray, Sequence[Any]]


class ExplainerAdapter(ABC):
    """
    Base class that standardizes how different explainers are called and what they return.

    Concrete implementations (e.g., SHAP Tree for XGBoost/LightGBM,
    SHAP Kernel for generic models, LIME-Text, etc.) should subclass this.

    Contract:
      - global_importance(X, rows_limit) -> (mean_abs, feature_names)
          * mean_abs: 1D np.ndarray of length n_features (mean absolute importance)
          * feature_names: List[str] of length n_features (ordering aligns with mean_abs)
      - local_explanations(X_row) -> 1D np.ndarray of length n_features (signed values)
          * X_row represents a single sample (shape (1, ...)); caller may pass a slice X[i:i+1]
    """

    def __init__(self, model_adapter: Any, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters
        ----------
        model_adapter : Any
            A model wrapper exposing at least:
              - predict(X), predict_proba(X) (optional)
              - feature_names() -> List[str]
              - class_names()   -> List[str]
              - is_sparse_input() -> bool (optional, defaults False)
        config : dict, optional
            Free-form configuration for the concrete explainer (batch_size, background, etc.).
        """
        self.m = model_adapter
        self.config = config or {}

    # ---------- Required API ----------

    @abstractmethod
    def global_importance(
        self,
        x: ArrayLike,
        rows_limit: int = 200,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute a global importance estimate (typically mean |contribution|) in a
        memory-safe way across up to `rows_limit` rows of X.

        Returns
        -------
        mean_abs : np.ndarray
            1D array of length n_features with mean absolute importance per feature.
        feature_names : List[str]
            The feature names in the same order as 'mean_abs'.
        """
        raise NotImplementedError

    @abstractmethod
    def local_explanations(self, x_row: ArrayLike) -> np.ndarray:
        """
        Explain a single row. Should return a vector of signed contributions whose length
        matches the feature space (n_features).

        Notes
        -----
        - Implementations may aggregate across classes if multi-class (e.g. sum over classes).
        - If there is a bias term, do NOT return it here; only per-feature values.
        """
        raise NotImplementedError

    # ---------- Optional helpers (available to subclasses) ----------

    def name(self) -> str:
        """Human-readable name of the explainer."""
        return self.__class__.__name__

    @staticmethod
    def ensure_2d(x: ArrayLike) -> ArrayLike:
        """
        Ensure that X has a leading sample dimension where possible.
        If X is a single sample 1D array, reshape to (1, n_features).
        No-op for sparse matrices/slices that already have a leading dimension.
        """
        if isinstance(x, np.ndarray) and x.ndim == 1:
            return x.reshape(1, -1)
        return x

    @staticmethod
    def to_top_k(
        values: np.ndarray,
        feature_names: Sequence[str],
        k: int = 15,
        signed: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convert a 1D vector of contributions into a ranked top-k list of dicts.

        Parameters
        ----------
        values : np.ndarray
            1D vector of per-feature contributions (signed or absolute).
        feature_names : Sequence[str]
            Names aligned to 'values'.
        k : int
            Number of top features to return.
        signed : bool
            If True, keep sign in 'value'. If False, return absolute values.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict has keys: {'feature', 'value', 'abs_value'}.
        """
        if values.ndim != 1:
            raise ValueError(f"'values' must be 1D, got shape {values.shape}")
        k = int(max(1, min(k, values.shape[0])))

        abs_vals = np.abs(values)
        top_idx = np.argsort(abs_vals)[-k:][::-1]

        out = []
        for j in top_idx:
            v = float(values[j]) if signed else float(abs_vals[j])
            out.append(
                {
                    "feature": str(feature_names[j]),
                    "value": v,
                    "abs_value": float(abs_vals[j]),
                }
            )
        return out


# -------------------------
# Optional: no-op explainer
# -------------------------
class NoOpExplainerAdapter(ExplainerAdapter):
    """
    A placeholder explainer for smoke tests.
    It returns zeros for global and local importance with correct shapes.
    """

    def global_importance(
        self,
        X: ArrayLike,
        rows_limit: int = 200,
    ) -> Tuple[np.ndarray, List[str]]:
        feats = self.m.feature_names() if hasattr(self.m, "feature_names") else []
        mean_abs = np.zeros(len(feats), dtype=np.float64)
        return mean_abs, list(feats)

    def local_explanations(self, x_row: ArrayLike) -> np.ndarray:
        feats = self.m.feature_names() if hasattr(self.m, "feature_names") else []
        return np.zeros(len(feats), dtype=np.float64)