import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q,K,V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = Q.shape[-1]

    scores = (Q @ K.T) / np.sqrt(d_k)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)

    return weights @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    assert d_model % n_heads == 0
    d_head = d_model // n_heads

    Q_heads = Q.reshape(seq_len, n_heads, d_head)
    K_heads = K.reshape(seq_len, n_heads, d_head)
    V_heads = V.reshape(seq_len, n_heads, d_head)

    outputs = []

    for h in range(n_heads):

        # Select one head
        q = Q_heads[:, h, :]
        k = K_heads[:, h, :]
        v = V_heads[:, h, :]

        # Scaled dot-product attention
        scores = (q @ k.T) / np.sqrt(d_head)

        # Stable softmax
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)

        # Attention output
        head_output = weights @ v

        outputs.append(head_output)

    # Concatenate heads
    output = np.concatenate(outputs, axis=-1)

    return output