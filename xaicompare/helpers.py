import json
import numpy as np

def make_json_safe(obj):
    """convert numpy/pandas types to built-in Python types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None:
        return None
    if isinstance(obj, (np.ndarray,)):
        # Ensure elements are also converted (important for object dtype arrays)
        return [make_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    # Fall back to string
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)
