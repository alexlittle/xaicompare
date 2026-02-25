# xaicompare/adapters/viz/viz_base.py
from typing import Union
import numpy as np
from scipy.sparse import spmatrix

ArrayLike = Union[np.ndarray, spmatrix, list]

class VizAdapter:

    def __init__(self, X: ArrayLike) -> np.ndarray:
        self.X = X
