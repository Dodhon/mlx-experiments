import mlx.core as mx


def axpby(x: mx.array, y: mx.array, alpha: float = 1.0, beta: float = 1.0) -> mx.array:
    """Return alpha * x + beta * y for two same-shaped MLX arrays."""
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    return alpha * x + beta * y


def mean_squared_error(predictions: mx.array, targets: mx.array) -> mx.array:
    """Return scalar mean squared error for two same-shaped MLX arrays."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same shape, got {predictions.shape} and {targets.shape}"
        )
    error = predictions - targets
    return mx.mean(mx.square(error))
