import numpy as np
import pickle
from scipy.spatial.distance import cosine

def emb_to_bytes(emb: np.ndarray) -> bytes:
    return pickle.dumps(emb.astype(np.float32))

def emb_from_bytes(b: bytes) -> np.ndarray:
    return pickle.loads(b)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine(a, b))

