from transformers import PretrainedConfig, AutoConfig, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SKIS-ICT-I/ISAC-V0-3B", use_fast=True)

def get_ref_m():
    reference_config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)
    return reference_config

vocab = get_ref_m().vocab_size

class ModelConfig(PretrainedConfig):
    model_type = "custom_transformer"

    def __init__(
        self,
        vocab_size=vocab,
        hidden_size=2304,
        intermediate_size=6144,
        num_hidden_layers=40,
        num_attention_heads=36,
        num_key_value_heads=6,  # GQA: fewer KV heads than Q heads
        max_position_embeddings=128_000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling={"type": "ntk", "factor": 31.25, "original_max_position_embeddings": 4096},
        sliding_window=32_000,
        tie_word_embeddings=True,
        **kwargs
    ):
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
        super().__init__(**kwargs)