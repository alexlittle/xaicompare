
from typing import List, Optional, Union
import numpy as np
from scipy.sparse import spmatrix

ArrayLike = Union[np.ndarray, spmatrix, list]

class ModelAdapter:
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
