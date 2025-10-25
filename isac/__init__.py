from Modeling_ISAC import CustomTransformerForCausalLM
from ISACConfig import ModelConfig, tokenizer, vocab
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoConfig, AutoModelForCausalLM

with ThreadPoolExecutor(max_workers=os.cpu_count()-2) as executor:
    if len(tokenizer) > vocab:
        tokenizer.add_tokens([f"<extra_{i}>" for i in range(vocab - len(tokenizer))])

AutoConfig.register("custom_transformer", ModelConfig)
AutoModelForCausalLM.register(ModelConfig, CustomTransformerForCausalLM)