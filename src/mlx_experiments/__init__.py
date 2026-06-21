"""Small MLX experiment helpers."""

from .arrays import axpby, mean_squared_error
from .linear_regression import fit_line

__all__ = ["axpby", "fit_line", "mean_squared_error"]
