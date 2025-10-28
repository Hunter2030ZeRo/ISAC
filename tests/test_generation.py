import pytest

torch = pytest.importorskip("torch")

from isac import ModelConfig, CustomTransformerForCausalLM


def test_model_generate_runs_with_cache():
    config = ModelConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=144,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        sliding_window=None,
    )
    model = CustomTransformerForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=4, use_cache=True)

    assert generated.shape[1] == input_ids.shape[1] + 4
