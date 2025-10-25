from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin, CausalLMOutputWithPast
from ISACConfig import vocab, ModelConfig, tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from concurrent.futures import ThreadPoolExecutor


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=128_000, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.sliding_window = None

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        base = config.rope_theta
        if getattr(config, "rope_scaling", None):
          rs = config.rope_scaling or {}
          if rs.get("type") == "ntk":
              base = base * float(rs.get("factor", 1.0))

        self.rotary_emb = RotaryEmbedding(
          self.head_dim,
          max_position_embeddings=config.max_position_embeddings,
          base=base,
      )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.size(-2)
        if past_key_value is not None and past_key_value[0] is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos_k, sin_k = self.rotary_emb(value_states, seq_len=kv_seq_len)  # for keys
        cos_q, sin_q = cos_k[-q_len:], sin_k[-q_len:]                     # last q_len slice for queries
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_q, sin_q)

        if past_key_value is not None and past_key_value[0] is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states)

        # Repeat KV heads for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Prepare attention mask for SDPA (causal + sliding window)
        base_mask = None  # sliding-window/causal 추가 마스크 (bool, [1,1,q,kv])

# (1) sliding window mask (optional)
        if self.sliding_window is not None:
            base = torch.ones(q_len, kv_seq_len, dtype=torch.bool, device=query_states.device)
            for i in range(q_len):
                past = kv_seq_len - q_len
                left = max(0, past + i + 1 - self.sliding_window)
                right = past + i + 1
                base[i, left:right] = False
            base_mask = base[None, None, :, :]  # [1,1,q,kv]

# (2) padding mask from additive attention mask [B,1,1,S]
        pad_mask = None
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
            pad = attention_mask.squeeze(1).squeeze(1) < 0  # [B,S]
            pad_len = pad.shape[-1]
            if pad_len > kv_seq_len:
                pad = pad[..., pad_len - kv_seq_len :]
            elif pad_len < kv_seq_len:
                pad = F.pad(pad, (kv_seq_len - pad_len, 0), value=True)
            pad_mask = pad[:, None, None, :]  # [B,1,1,kv]

# (3) 최종 attn_mask (bool) 결합
#        if base_mask is not None and pad_mask is not None:
#            attn_mask = base_mask | pad_mask               # [B,1,q,kv] (broadcast)
#        elif base_mask is not None:
#            attn_mask = base_mask                          # [1,1,q,kv] → broadcast to [B,1,q,kv]
#        elif pad_mask is not None:
#            attn_mask = pad_mask.expand(-1, 1, q_len, -1)  # [B,1,1,kv] → [B,1,q,kv]
#        else:
#            attn_mask = None

# --- SDPA ---

        attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=pad_mask,       # bool mask OK
        dropout_p=0.0,
        is_causal=True,            # causal은 항상 켜두고, 추가 제약은 attn_mask로
    )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class CustomTransformerModel(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
    ):
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)

        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = None

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]            # [B,1,1,S]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -1e4 

        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            hidden_states, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )

            if use_cache:
                next_decoder_cache += (present_key_value,)

        hidden_states = self.norm(hidden_states)

        return hidden_states, next_decoder_cache


class CustomTransformerForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomTransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    supports_gradient_checkpointing = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing:
            use_cache = False
        elif use_cache is None:
            use_cache = getattr(self.config, "use_cache", True)

        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + (past_key_values,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Required for beam search generation"""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        use_cache = kwargs.get("use_cache", getattr(self.config, "use_cache", True))
        if self.gradient_checkpointing:
            use_cache = False
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }

config = ModelConfig()
student_model = CustomTransformerForCausalLM(config)

with ThreadPoolExecutor(max_workers=os.cpu_count()-2) as executor:
    student_model.resize_token_embeddings(vocab)

