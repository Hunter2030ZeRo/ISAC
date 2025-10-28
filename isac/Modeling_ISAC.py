"""PyTorch implementation of the custom ISAC 3B decoder-only model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CausalLMOutputWithPast, GenerationMixin, PreTrainedModel

from .ISACConfig import ModelConfig


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

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids is not None:
            t = position_ids.to(self.inv_freq.device).float()
        else:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb.to(dtype=x.dtype)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GatedDeltaNet(nn.Module):
    """Implements a lightweight Gated-DeltaNet style residual update."""

    def __init__(self, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, delta: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(delta))
        return residual + gate * delta


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.sliding_window = config.sliding_window
        self.dropout = config.attention_dropout

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

        past_key = past_value = None
        past_len = 0
        if past_key_value is not None and past_key_value[0] is not None:
            past_key, past_value = past_key_value
            past_len = past_key.shape[2]

        total_len = past_len + key_states.size(2)
        cos, sin = self.rotary_emb(key_states, seq_len=total_len)
        cos_new = cos[:, :, past_len:, :]
        sin_new = sin[:, :, past_len:, :]
        key_states = apply_rotary_pos_emb(key_states, cos_new, sin_new)
        query_cos = cos[:, :, total_len - q_len :, :]
        query_sin = sin[:, :, total_len - q_len :, :]
        query_states = apply_rotary_pos_emb(query_states, query_cos, query_sin)

        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states)
        kv_seq_len = key_states.size(2)

        attn_key = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_value = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Prepare attention mask for SDPA (causal + sliding window)
        base_mask = None
        if self.sliding_window is not None:
            k_positions = torch.arange(kv_seq_len, device=query_states.device)
            q_positions = k_positions[-q_len:]
            if kv_seq_len > q_len:
                q_positions = torch.arange(kv_seq_len - q_len, kv_seq_len, device=query_states.device)
            distance = q_positions[:, None] - k_positions[None, :]
            base = distance > self.sliding_window
            base = base.to(query_states.device)
            base_mask = base[None, None, :, :]

        pad_mask = None
        if isinstance(attention_mask, torch.Tensor):
            mask = (attention_mask == 0).to(device=query_states.device)
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            mask = mask[..., -kv_seq_len:]
            pad_mask = mask

        combined_mask = None
        if base_mask is not None:
            combined_mask = base_mask
        if pad_mask is not None:
            combined_mask = pad_mask if combined_mask is None else combined_mask | pad_mask

        attn_mask = None
        if combined_mask is not None:
            combined_mask = combined_mask.expand(bsz, 1, q_len, kv_seq_len)
            attn_mask = combined_mask.expand(-1, self.num_heads, -1, -1)

        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            attn_key,
            attn_value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_gated_attention = config.use_gated_attention
        self.use_gated_delta_net = config.use_gated_delta_net
        if self.use_gated_attention:
            self.attn_gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        if self.use_gated_delta_net:
            self.delta_net = GatedDeltaNet(config.hidden_size, bias=config.gated_delta_net_bias)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        if self.use_gated_attention:
            gate = torch.sigmoid(self.attn_gate_proj(residual))
            hidden_states = hidden_states * gate
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.use_gated_delta_net:
            hidden_states = self.delta_net(hidden_states, residual)
        else:
            hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class CustomTransformerModel(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

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
            past_length = 0
            if past_key_values is not None and len(past_key_values) > 0:
                past_length = past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                past_length, past_length + seq_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_states.device)

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

    def __init__(self, config: ModelConfig):
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
        position_ids = kwargs.get("position_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            elif attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids[:, -1:]
            else:
                position_ids = None
        use_cache = kwargs.get("use_cache", getattr(self.config, "use_cache", True))
        if self.gradient_checkpointing:
            use_cache = False
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, CustomTransformerModel):
            module.gradient_checkpointing = value

