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

## Tiny LLM lab

Train a byte-level decoder-only Transformer on tiny text, save a checkpoint, and sample from it:

```bash
python tiny_llm_lab/train.py \
  --steps 200 \
  --context-size 64 \
  --layers 2 \
  --d-model 128 \
  --heads 4 \
  --checkpoint tiny_llm_lab/checkpoints/latest

python tiny_llm_lab/sample.py \
  --checkpoint tiny_llm_lab/checkpoints/latest \
  --prompt "Tiny models" \
  --max-new-tokens 80
```

Generated datasets, checkpoints, and samples under `tiny_llm_lab/` are ignored by Git.
