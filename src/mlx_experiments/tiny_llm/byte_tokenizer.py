from __future__ import annotations


class ByteTokenizer:
    """UTF-8 byte tokenizer with a fixed 256-token vocabulary."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token id must be in [0, 255], got {token_id}")
        return bytes(token_ids).decode("utf-8", errors="replace")
