from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import mlx.core as mx

from .byte_tokenizer import ByteTokenizer
from .model import TinyGPT


def generate_tokens(
    model: TinyGPT,
    prompt: str,
    *,
    max_new_tokens: int = 80,
) -> list[int]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    tokenizer = ByteTokenizer()
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [10]

    for _ in range(max_new_tokens):
        context = token_ids[-model.config.context_size :]
        logits = model(mx.array([context], dtype=mx.int32))[0, -1]
        next_token = int(mx.argmax(logits))
        token_ids.append(next_token)

    return token_ids


def generate(
    model: TinyGPT,
    prompt: str,
    *,
    max_new_tokens: int = 80,
) -> str:
    return ByteTokenizer().decode(
        generate_tokens(model, prompt, max_new_tokens=max_new_tokens)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample from a trained tiny byte-level MLX GPT.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Tiny models")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = TinyGPT.load_checkpoint(args.checkpoint)
    model.eval()
    print(generate(model, args.prompt, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
