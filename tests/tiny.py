"""Tiny randomly initialised Qwen2 models for testing the full loop on a CPU in seconds."""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

# With the default initializer_range (0.02) the models are near-uniform, so p is almost q,
# nearly everything is accepted, and the rejection path never runs. A larger scale makes
# the two models genuinely disagree.
INIT_RANGE = 0.5


def tiny_model(seed: int, vocab_size: int, init_range: float = INIT_RANGE) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        initializer_range=init_range,
    )
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model
