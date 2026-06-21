from dataclasses import dataclass

import mlx.core as mx

from .arrays import mean_squared_error


@dataclass(frozen=True)
class LineFit:
    weight: mx.array
    bias: mx.array
    loss: mx.array


def _loss(weight: mx.array, bias: mx.array, x: mx.array, y: mx.array) -> mx.array:
    return mean_squared_error(weight * x + bias, y)


def fit_line(x: mx.array, y: mx.array, *, steps: int = 200, learning_rate: float = 0.05) -> LineFit:
    """Fit y ~= weight * x + bias with gradient descent."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")

    weight = mx.array(0.0)
    bias = mx.array(0.0)
    loss_and_grad = mx.value_and_grad(_loss, argnums=[0, 1])

    loss = mx.array(float("inf"))
    for _ in range(steps):
        loss, (weight_grad, bias_grad) = loss_and_grad(weight, bias, x, y)
        weight = weight - learning_rate * weight_grad
        bias = bias - learning_rate * bias_grad
        mx.eval(weight, bias)

    loss = _loss(weight, bias, x, y)
    mx.eval(weight, bias, loss)
    return LineFit(weight=weight, bias=bias, loss=loss)
