"""Configuration utilities for the ISAC 3B language model."""

from __future__ import annotations

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig


#
# We mirror the tokenizer used by the upstream SKIS-ICT-I/ISAC-V0-3B release so that
# checkpoints initialised from this repository can immediately interoperate with the
# published weights.  The tokenizer is intentionally loaded at import time because it
# is lightweight and frequently required by downstream tooling (training scripts,
# evaluation harnesses, etc.).
#
tokenizer = AutoTokenizer.from_pretrained("SKIS-ICT-I/ISAC-V0-3B", use_fast=True)


def _load_reference_config() -> AutoConfig:
    """Load the reference configuration we benchmark against (Qwen3 family).

    The reference is used solely to recover the canonical vocabulary size so that
    checkpoints remain weight-compatible with the tokenizer.  We avoid relying on
    values baked into the tokenizer itself to ensure consistency with the modelling
    stack (particularly for architectures that inherit the Qwen3 tokeniser).
    """

    return AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)


vocab = _load_reference_config().vocab_size


class ModelConfig(PretrainedConfig):
    """Configuration class for the custom 3B decoder-only architecture.

    The defaults roughly match a 3B parameter budget inspired by the
    Qwen3-Next-80B-A3B architecture but scaled down to a student-friendly size.  The
    design keeps RMSNorm, RoPE and SwiGLU as required by the user specification and
    optionally enables experimental blocks such as gated attention and a
    Gated-DeltaNet style feed-forward gating mechanism.
    """

    model_type = "custom_transformer"

    def __init__(
        self,
        vocab_size: int = vocab,
        hidden_size: int = 2304,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 36,
        num_key_value_heads: int = 6,
        max_position_embeddings: int = 128_000,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        sliding_window: int | None = 32_000,
        tie_word_embeddings: bool = True,
        use_gated_attention: bool = False,
        use_gated_delta_net: bool = False,
        gated_delta_net_bias: bool = True,
        attention_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        if rope_scaling is None:
            rope_scaling = {
                "type": "ntk",
                "factor": 31.25,
                "original_max_position_embeddings": 4096,
            }

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.tie_word_embeddings = tie_word_embeddings
        self.use_gated_attention = use_gated_attention
        self.use_gated_delta_net = use_gated_delta_net
        self.gated_delta_net_bias = gated_delta_net_bias
        self.attention_dropout = attention_dropout

        super().__init__(**kwargs)
