import io
import numpy as np

def serialize_tensor(tensor: np.ndarray) -> bytes:
    """
    Converts a NumPy array into a raw byte string.
    """
    buff = io.BytesIO()
    # allow_pickle=False is safer for security
    np.save(buff, tensor, allow_pickle=False)
    return buff.getvalue()

def deserialize_tensor(data: bytes) -> np.ndarray:
    """
    Converts a raw byte string back into a NumPy array.
    """
    buff = io.BytesIO(data)
    return np.load(buff, allow_pickle=False)