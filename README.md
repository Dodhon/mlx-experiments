# MLX Experiments

Small Apple MLX experiments for learning array operations, autodiff, and tiny training loops on Apple silicon.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

MLX requires Apple silicon, macOS 14+, and a native arm64 Python.

## Smoke test

```bash
pytest
python examples/linear_regression.py
```
