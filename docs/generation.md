# Text generation with the ISAC decoder

The custom decoder is registered with Hugging Face so it can be instantiated via
`AutoModelForCausalLM` or directly through `isac.CustomTransformerForCausalLM`. The
example below shows how to create a small instance for smoke testing and sample text
with caching enabled:

```python
from isac import ModelConfig, CustomTransformerForCausalLM
import torch

config = ModelConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=144,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=256,
)
model = CustomTransformerForCausalLM(config)
model.eval()

prompt = torch.tensor([[1, 5, 9, 2]])
outputs = model.generate(prompt, max_new_tokens=16, use_cache=True)
print(outputs)
```

Any tokenizer compatible with the configured vocabulary can be used. For checkpoints
trained on SKIS-ICT-I/ISAC-V0-3B, load the published tokenizer through
`isac.tokenizer` and feed its encoded prompts into `generate`.
