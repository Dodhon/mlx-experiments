from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class TinyGPTConfig:
    vocab_size: int = 256
    context_size: int = 128
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_json_dict(self) -> dict[str, int]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        queries = queries.reshape(batch, tokens, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch, tokens, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch, tokens, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.head_dim**-0.5,
            mask="causal",
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, channels)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        hidden_dim = config.mlp_ratio * config.d_model
        self.up = nn.Linear(config.d_model, hidden_dim)
        self.down = nn.Linear(hidden_dim, config.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.gelu(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class TinyGPT(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have shape [batch, tokens], got {token_ids.shape}")
        _, tokens = token_ids.shape
        if tokens > self.config.context_size:
            raise ValueError(
                f"input length {tokens} exceeds context_size {self.config.context_size}"
            )

        positions = mx.arange(tokens, dtype=mx.int32)[None, :]
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def save_checkpoint(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "config.json"
        weights_path = directory / "weights.safetensors"
        config_path.write_text(_json_dumps(self.config.to_json_dict()), encoding="utf-8")
        self.save_weights(str(weights_path))

    @classmethod
    def load_checkpoint(cls, directory: Path) -> "TinyGPT":
        import json

        config = TinyGPTConfig(**json.loads((directory / "config.json").read_text(encoding="utf-8")))
        model = cls(config)
        model.load_weights(str(directory / "weights.safetensors"))
        mx.eval(model.parameters())
        return model


def _json_dumps(payload: dict[str, int]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def language_model_loss(model: TinyGPT, inputs: mx.array, targets: mx.array) -> mx.array:
    logits = model(inputs)
    flat_logits = logits.reshape(-1, model.config.vocab_size)
    flat_targets = targets.reshape(-1)
    return mx.mean(nn.losses.cross_entropy(flat_logits, flat_targets))
