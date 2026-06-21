import pytest
import mlx.core as mx

from mlx_experiments import axpby, fit_line, mean_squared_error


def test_axpby_combines_same_shaped_arrays() -> None:
    result = axpby(mx.array([1.0, 2.0]), mx.array([3.0, 4.0]), alpha=2.0, beta=-1.0)
    mx.eval(result)
    assert result.tolist() == [-1.0, 0.0]


def test_axpby_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        axpby(mx.array([1.0]), mx.array([[1.0]]))


def test_mean_squared_error_returns_scalar_loss() -> None:
    loss = mean_squared_error(mx.array([1.0, 3.0]), mx.array([2.0, 1.0]))
    mx.eval(loss)
    assert float(loss) == pytest.approx(2.5)


def test_fit_line_learns_weight_and_bias() -> None:
    x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = 3.0 * x - 2.0

    fit = fit_line(x, y, steps=300, learning_rate=0.05)
    mx.eval(fit.weight, fit.bias, fit.loss)

    assert float(fit.weight) == pytest.approx(3.0, abs=0.05)
    assert float(fit.bias) == pytest.approx(-2.0, abs=0.05)
    assert float(fit.loss) < 0.01
