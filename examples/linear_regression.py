import mlx.core as mx

from mlx_experiments import fit_line


def main() -> None:
    x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = 3.0 * x - 2.0
    fit = fit_line(x, y, steps=300, learning_rate=0.05)
    mx.eval(fit.weight, fit.bias, fit.loss)
    print(f"weight={float(fit.weight):.3f} bias={float(fit.bias):.3f} loss={float(fit.loss):.6f}")


if __name__ == "__main__":
    main()
