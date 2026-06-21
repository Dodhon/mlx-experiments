from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx

from .byte_tokenizer import ByteTokenizer


@dataclass(frozen=True)
class Batch:
    inputs: mx.array
    targets: mx.array


DEFAULT_CORPUS = """
Tiny language models learn by predicting the next token.
A small model can still teach attention, loss curves, sampling, and optimization.
This lab starts by overfitting tiny text before scaling to larger corpora.
""".strip()


def load_text(path: Path | None, inline_text: str | None = None) -> str:
    if path is not None and inline_text is not None:
        raise ValueError("provide either path or inline_text, not both")
    if path is not None:
        return path.read_text(encoding="utf-8")
    if inline_text is not None:
        return inline_text
    return DEFAULT_CORPUS


def prepare_tokens(text: str, *, min_length: int) -> list[int]:
    if min_length < 2:
        raise ValueError("min_length must be at least 2")
    token_ids = ByteTokenizer().encode(text)
    if not token_ids:
        raise ValueError("training text must not be empty")
    while len(token_ids) < min_length:
        token_ids.extend(token_ids)
    return token_ids


def make_batch(
    token_ids: Sequence[int],
    *,
    batch_size: int,
    context_size: int,
    rng: random.Random,
) -> Batch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if len(token_ids) <= context_size:
        raise ValueError("token_ids must contain at least context_size + 1 tokens")

    max_start = len(token_ids) - context_size - 1
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for _ in range(batch_size):
        start = rng.randint(0, max_start)
        end = start + context_size
        inputs.append(list(token_ids[start:end]))
        targets.append(list(token_ids[start + 1 : end + 1]))

    return Batch(inputs=mx.array(inputs, dtype=mx.int32), targets=mx.array(targets, dtype=mx.int32))
