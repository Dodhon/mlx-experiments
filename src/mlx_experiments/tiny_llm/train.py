from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .data import Batch, load_text, make_batch, prepare_tokens
from .model import TinyGPT, TinyGPTConfig, language_model_loss
from .sample import generate


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    batch_size: int = 16
    learning_rate: float = 3e-4
    seed: int = 42
    log_every: int = 25

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")


@dataclass(frozen=True)
class TrainSummary:
    losses: list[float]
    elapsed_seconds: float
    tokens_per_second: float
    sample: str


def _train_step_fn(model: TinyGPT):
    return nn.value_and_grad(model, language_model_loss)


def train_overfit_batch(
    batch: Batch,
    *,
    model_config: TinyGPTConfig,
    train_config: TrainConfig,
) -> tuple[TinyGPT, TrainSummary]:
    mx.random.seed(train_config.seed)
    model = TinyGPT(model_config)
    optimizer = optim.AdamW(learning_rate=train_config.learning_rate)
    loss_and_grad = _train_step_fn(model)
    losses: list[float] = []
    started = time.perf_counter()

    for step in range(train_config.steps):
        loss, grads = loss_and_grad(model, batch.inputs, batch.targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 0 or (step + 1) % train_config.log_every == 0 or step + 1 == train_config.steps:
            losses.append(float(loss))

    elapsed = time.perf_counter() - started
    token_count = train_config.steps * batch.inputs.size
    summary = TrainSummary(
        losses=losses,
        elapsed_seconds=elapsed,
        tokens_per_second=token_count / elapsed if elapsed else 0.0,
        sample=generate(model, "Tiny", max_new_tokens=40),
    )
    return model, summary


def train_on_text(
    text: str,
    *,
    model_config: TinyGPTConfig,
    train_config: TrainConfig,
) -> tuple[TinyGPT, TrainSummary]:
    token_ids = prepare_tokens(text, min_length=model_config.context_size + 1)
    rng = random.Random(train_config.seed)
    mx.random.seed(train_config.seed)
    model = TinyGPT(model_config)
    optimizer = optim.AdamW(learning_rate=train_config.learning_rate)
    loss_and_grad = _train_step_fn(model)
    losses: list[float] = []
    started = time.perf_counter()

    for step in range(train_config.steps):
        batch = make_batch(
            token_ids,
            batch_size=train_config.batch_size,
            context_size=model_config.context_size,
            rng=rng,
        )
        loss, grads = loss_and_grad(model, batch.inputs, batch.targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 0 or (step + 1) % train_config.log_every == 0 or step + 1 == train_config.steps:
            losses.append(float(loss))

    elapsed = time.perf_counter() - started
    token_count = train_config.steps * train_config.batch_size * model_config.context_size
    summary = TrainSummary(
        losses=losses,
        elapsed_seconds=elapsed,
        tokens_per_second=token_count / elapsed if elapsed else 0.0,
        sample=generate(model, "Tiny", max_new_tokens=40),
    )
    return model, summary


def _print_summary(summary: TrainSummary, model_config: TinyGPTConfig, train_config: TrainConfig) -> None:
    payload = {
        "model": model_config.to_json_dict(),
        "training": asdict(train_config),
        "summary": asdict(summary),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny byte-level decoder-only LLM in MLX.")
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("tiny_llm_lab/checkpoints/latest"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_config = TinyGPTConfig(
        context_size=args.context_size,
        n_layers=args.layers,
        d_model=args.d_model,
        n_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    )
    train_config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_every=args.log_every,
    )
    text = load_text(args.text_file, args.text)
    model, summary = train_on_text(text, model_config=model_config, train_config=train_config)
    model.save_checkpoint(args.checkpoint)
    _print_summary(summary, model_config, train_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
