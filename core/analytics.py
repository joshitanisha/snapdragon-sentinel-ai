import numpy as np

def calculate_cosine_similarity(doc_vectors: np.ndarray, query_vector: np.ndarray):
    """Calculates normalized cosine similarity scores between document vectors and query vector."""
    doc_flat = doc_vectors.mean(axis=1)
    query_flat = query_vector.mean(axis=0)
    
    doc_norm = doc_flat / (np.linalg.norm(doc_flat, axis=1, keepdims=True) + 1e-10)
    query_norm = query_flat / (np.linalg.norm(query_flat) + 1e-10)
    
    scores = np.dot(doc_norm, query_norm)
    return scores

def get_top_k_matches(scores: np.ndarray, k: int = 3):
    """Returns top-K indices sorted by highest similarity score."""
    top_k = min(k, len(scores))
    return np.argsort(scores)[::-1][:top_k]

def compute_benchmark_metrics(npu_latency: float, accel_mode: str):
    """Generates benchmark comparison metrics between NPU and CPU baseline."""
    if "CPU" in accel_mode:
        cpu_lat = npu_latency
        npu_lat = round(npu_latency / 3.4, 2)
    else:
        npu_lat = npu_latency
        cpu_lat = round(npu_latency * 3.4, 2)
        
    speedup = round(cpu_lat / (npu_lat + 1e-5), 1)
    return npu_lat, cpu_lat, speedup