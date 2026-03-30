from __future__ import annotations

import mlx.core as mx

def mse(x, y):
    return mx.mean(mx.square(x - y))

def mae(x, y):
    return mx.mean(mx.abs(x - y))

def cosine_similarity(x, y, eps: float = 1e-8):
    x_norm = x / (mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) + eps)
    y_norm = y / (mx.sqrt(mx.sum(y * y, axis=-1, keepdims=True)) + eps)
    return mx.mean(mx.sum(x_norm * y_norm, axis=-1))
