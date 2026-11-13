from __future__ import annotations

import numpy as np

class OpenCVError(Exception):
    pass

def load_or_raise(path: str) -> np.ndarray:
    raise NotImplementedError
