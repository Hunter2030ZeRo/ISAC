"""Public package surface for the ISAC 3B model."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForCausalLM

from .ISACConfig import ModelConfig, tokenizer, vocab
from .Modeling_ISAC import CustomTransformerForCausalLM

__all__ = [
    "CustomTransformerForCausalLM",
    "ModelConfig",
    "tokenizer",
    "vocab",
]


# Ensure that Hugging Face can locate the custom architecture through the standard
# registration hooks.  This mirrors the behaviour of upstream transformer packages and
# keeps the implementation compatible with AutoModel / AutoConfig factories.
AutoConfig.register(ModelConfig.model_type, ModelConfig)
AutoModelForCausalLM.register(ModelConfig, CustomTransformerForCausalLM)
