import random

import mlx.core as mx
import pytest

from mlx_experiments.tiny_llm import (
    Batch,
    ByteTokenizer,
    TinyGPT,
    TinyGPTConfig,
    TrainConfig,
    generate_tokens,
    language_model_loss,
    make_batch,
    prepare_tokens,
    train_overfit_batch,
    train_on_text,
)


def tiny_config() -> TinyGPTConfig:
    return TinyGPTConfig(context_size=8, n_layers=1, d_model=32, n_heads=4, mlp_ratio=2)


def test_byte_tokenizer_round_trips_utf8_text() -> None:
    tokenizer = ByteTokenizer()
    text = "hello π"

    token_ids = tokenizer.encode(text)

    assert all(0 <= token_id < 256 for token_id in token_ids)
    assert tokenizer.decode(token_ids) == text


def test_make_batch_returns_next_token_targets() -> None:
    token_ids = list(range(12))

    batch = make_batch(token_ids, batch_size=2, context_size=4, rng=random.Random(0))
    mx.eval(batch.inputs, batch.targets)

    assert batch.inputs.shape == (2, 4)
    assert batch.targets.shape == (2, 4)
    for inputs, targets in zip(batch.inputs.tolist(), batch.targets.tolist(), strict=True):
        assert targets == [token + 1 for token in inputs]


def test_prepare_tokens_repeats_tiny_text_to_minimum_length() -> None:
    token_ids = prepare_tokens("ab", min_length=9)

    assert len(token_ids) >= 9
    assert token_ids[:4] == [97, 98, 97, 98]


def test_tiny_gpt_returns_logits_for_each_position() -> None:
    model = TinyGPT(tiny_config())
    inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    logits = model(inputs)
    mx.eval(logits)

    assert logits.shape == (1, 4, 256)


def test_causal_attention_does_not_read_future_tokens() -> None:
    model = TinyGPT(tiny_config())
    prefix = [1, 2, 3]
    first = mx.array([prefix + [4]], dtype=mx.int32)
    second = mx.array([prefix + [99]], dtype=mx.int32)

    first_logits = model(first)[:, : len(prefix)]
    second_logits = model(second)[:, : len(prefix)]
    mx.eval(first_logits, second_logits)

    assert bool(mx.allclose(first_logits, second_logits, atol=1e-5))


def test_language_model_loss_returns_scalar() -> None:
    model = TinyGPT(tiny_config())
    inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    targets = mx.array([[2, 3, 4, 5]], dtype=mx.int32)

    loss = language_model_loss(model, inputs, targets)
    mx.eval(loss)

    assert loss.shape == ()
    assert float(loss) > 0


def test_overfit_batch_reduces_loss() -> None:
    config = tiny_config()
    inputs = mx.array([[65, 66, 65, 66, 65, 66, 65, 66]], dtype=mx.int32)
    targets = mx.array([[66, 65, 66, 65, 66, 65, 66, 65]], dtype=mx.int32)
    batch = Batch(inputs=inputs, targets=targets)

    _, summary = train_overfit_batch(
        batch,
        model_config=config,
        train_config=TrainConfig(steps=12, batch_size=1, learning_rate=1e-2, seed=7, log_every=1),
    )

    assert summary.losses[-1] < summary.losses[0]
    assert summary.tokens_per_second > 0



def test_train_on_text_runs_corpus_batches() -> None:
    _, summary = train_on_text(
        "the tiny model sees more than one possible window",
        model_config=tiny_config(),
        train_config=TrainConfig(steps=3, batch_size=2, learning_rate=1e-3, seed=3, log_every=1),
    )

    assert len(summary.losses) == 3
    assert summary.tokens_per_second > 0

def test_generate_extends_prompt() -> None:
    config = tiny_config()
    model = TinyGPT(config)

    token_ids = generate_tokens(model, "A", max_new_tokens=3)

    assert len(token_ids) == 4
