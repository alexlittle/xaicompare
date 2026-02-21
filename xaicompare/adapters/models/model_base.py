
from typing import List, Optional, Union, Sequence, Any
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

ArrayLike = Union[np.ndarray, spmatrix, list]

class ModelAdapter:

    def __init__(self, model, class_names: Optional[Sequence[str]] = None):
        self.model = model
        self._class_names = (
            list(class_names) if class_names is not None else None
        )

    """Uniform interface over different model types."""
    def predict(self, X: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: ArrayLike) -> Optional[np.ndarray]:
        """Return (n_samples, n_classes) or None if not supported."""
        return None

    def feature_names(self) -> List[str]:
        """Vectorizer / model feature names (tokens, n-grams)."""
        raise NotImplementedError

    def class_names(self) -> List[str]:
        """Human-readable class names."""
        raise NotImplementedError

    def is_sparse_input(self) -> bool:
        return False

    def build_text_index(
            self,
            X_test,
            y_test: Optional[Sequence] = None,
            raw_text: Optional[Sequence] = None,
            class_names: Optional[Sequence[str]] = None,
            **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with at least:
          - id (int)
          - text (str)               # original doc text if available
          - y_true (optional)
          - y_pred (optional)
          - proba_{class} (optional per class)
        Default impl: try best-effort; subclasses can override.
        """
        raise NotImplementedError

