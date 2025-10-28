"""PyTorch implementation of the custom ISAC 3B decoder-only model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from ISACConfig import ModelConfig


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
        self.register_buffer("_cos_cached", None, persistent=False)
        self.register_buffer("_sin_cached", None, persistent=False)
        self._seq_len_cached: int = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self._seq_len_cached:
            if self._cos_cached is not None and self._sin_cached is not None:
                if self._cos_cached.device == device and self._cos_cached.dtype == dtype:
                    return

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self.register_buffer("_cos_cached", cos, persistent=False)
        self.register_buffer("_sin_cached", sin, persistent=False)
        self._seq_len_cached = seq_len

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        required_len = seq_len
        if position_ids is not None:
            required_len = max(required_len, int(position_ids.max().item()) + 1)
        required_len = min(required_len, self.max_position_embeddings)
        self._build_cache(required_len, device=device, dtype=dtype)

        if position_ids is not None:
            cos = self._cos_cached[..., position_ids, :]
            sin = self._sin_cached[..., position_ids, :]
        else:
            cos = self._cos_cached[..., :seq_len, :]
            sin = self._sin_cached[..., :seq_len, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


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

    def forward(self, x, token_mask: Optional[torch.Tensor] = None):
        if token_mask is None or bool(token_mask.all()):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_output = output.view(-1, output.size(-1))
        flat_mask = token_mask.view(-1)
        active_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            return output

        active_hidden = flat_x.index_select(0, active_indices)
        activated = F.silu(self.gate_proj(active_hidden)) * self.up_proj(active_hidden)
        projected = self.down_proj(activated)
        flat_output.index_copy_(0, active_indices, projected)
        return output


class TokenRouter(nn.Module):
    """Predicts per-token gates following the Qwen3-Next routing strategy."""

    def __init__(
        self,
        hidden_size: int,
        reduction: int = 16,
        bias: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        reduction = max(1, reduction)
        reduced_dim = max(1, hidden_size // reduction)
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.down_proj = nn.Linear(hidden_size, reduced_dim, bias=bias)
        self.act = nn.SiLU()
        self.up_proj = nn.Linear(reduced_dim, 1, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routed = self.norm(hidden_states)
        routed = self.act(self.down_proj(routed))
        return torch.sigmoid(self.up_proj(routed))


class ResidualGate(nn.Module):
    """Shared squeeze-and-gate block used by attention and feed-forward deltas."""

    def __init__(self, hidden_size: int, reduction: int = 16, bias: bool = True) -> None:
        super().__init__()
        reduction = max(1, reduction)
        reduced_dim = max(1, hidden_size // reduction)
        self.down_proj = nn.Linear(hidden_size, reduced_dim, bias=bias)
        self.act = nn.SiLU()
        self.up_proj = nn.Linear(reduced_dim, hidden_size, bias=bias)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        gate = self.up_proj(self.act(self.down_proj(activations)))
        return torch.sigmoid(gate)


class GatedDeltaNet(nn.Module):
    """Implements an efficient Gated-DeltaNet style residual update."""

    def __init__(
        self,
        hidden_size: int,
        reduction: int = 16,
        bias: bool = True,
        use_residual_gate: bool = True,
    ) -> None:
        super().__init__()
        self.gate = (
            ResidualGate(hidden_size, reduction=reduction, bias=bias) if use_residual_gate else None
        )

    def forward(
        self,
        delta: torch.Tensor,
        residual: torch.Tensor,
        gate_values: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate = gate_values
        if self.gate is not None:
            local_gate = self.gate(delta)
            gate = local_gate if gate is None else gate * local_gate

        if gate is None:
            gate = 1.0

        if token_mask is not None:
            mask = token_mask.unsqueeze(-1).to(delta.dtype)
            delta = delta * mask
            if isinstance(gate, torch.Tensor):
                gate = gate * mask
            else:
                gate = mask

        if isinstance(gate, torch.Tensor):
            return residual + gate * delta
        return residual + delta


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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        token_mask: Optional[torch.Tensor] = None,
        router_scores: Optional[torch.Tensor] = None,
    ):
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
        if token_mask is not None:
            token_mask = token_mask.to(device=query_states.device)
            if token_mask.dtype != torch.bool:
                token_mask = token_mask > 0

        if token_mask is not None and not bool(token_mask.all()):
            attn_output = torch.zeros(bsz, q_len, self.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
            for batch_idx in range(bsz):
                active = token_mask[batch_idx]
                if active.all():
                    q = query_states[batch_idx : batch_idx + 1]
                    mask_b = attn_mask[batch_idx : batch_idx + 1] if attn_mask is not None else None
                    attn_slice = F.scaled_dot_product_attention(
                        q,
                        attn_key[batch_idx : batch_idx + 1],
                        attn_value[batch_idx : batch_idx + 1],
                        attn_mask=mask_b,
                        dropout_p=dropout_p,
                        is_causal=True,
                    )
                    attn_slice = attn_slice.transpose(1, 2).contiguous().reshape(1, q_len, self.hidden_size)
                    attn_output[batch_idx] = attn_slice[0]
                    continue

                if not active.any():
                    continue

                active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
                q = query_states[
                    batch_idx : batch_idx + 1,
                    :,
                    active_idx,
                    :,
                ]
                mask_b = None
                if attn_mask is not None:
                    mask_b = attn_mask[
                        batch_idx : batch_idx + 1,
                        :,
                        active_idx,
                        :,
                    ]
                attn_slice = F.scaled_dot_product_attention(
                    q,
                    attn_key[batch_idx : batch_idx + 1],
                    attn_value[batch_idx : batch_idx + 1],
                    attn_mask=mask_b,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
                attn_slice = attn_slice.transpose(1, 2).contiguous().reshape(len(active_idx), self.hidden_size)
                attn_output[batch_idx, active_idx] = attn_slice
        else:
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

        if router_scores is not None:
            attn_output = attn_output * router_scores

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
            self.attn_router = TokenRouter(
                config.hidden_size,
                reduction=getattr(config, "gated_attention_reduction", 16),
                bias=False,
                eps=config.rms_norm_eps,
            )
            self.attn_threshold = getattr(config, "gated_attention_threshold", 0.0)
            self.attn_min_tokens = getattr(config, "gated_attention_min_tokens", 0.0)
        if self.use_gated_delta_net:
            self.delta_router = TokenRouter(
                config.hidden_size,
                reduction=getattr(config, "gated_delta_net_reduction", 16),
                bias=config.gated_delta_net_bias,
                eps=config.rms_norm_eps,
            )
            self.delta_net = GatedDeltaNet(
                config.hidden_size,
                reduction=getattr(config, "gated_delta_net_reduction", 16),
                bias=config.gated_delta_net_bias,
                use_residual_gate=False,
            )
            self.delta_threshold = getattr(config, "gated_delta_net_threshold", 0.0)
            self.delta_min_tokens = getattr(config, "gated_delta_net_min_tokens", 0.0)

    @staticmethod
    def _compute_gate_mask(
        gate_scores: torch.Tensor,
        threshold: float,
        min_tokens: float,
    ) -> Optional[torch.Tensor]:
        if gate_scores is None:
            return None
        values = gate_scores.squeeze(-1)
        if threshold <= 0 and min_tokens <= 0:
            return None

        mask = values >= threshold

        seq_len = values.size(-1)
        if min_tokens > 0:
            if min_tokens < 1:
                keep = max(1, int(seq_len * min_tokens))
            else:
                keep = min(seq_len, int(min_tokens))
            topk_values, _ = torch.topk(values, keep, dim=-1)
            dynamic_threshold = topk_values[..., -1:]
            mask = mask | (values >= dynamic_threshold)

        return mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        normed_hidden = self.input_layernorm(hidden_states)

        attn_router_scores = None
        attn_token_mask = None
        if self.use_gated_attention:
            attn_router_scores = self.attn_router(normed_hidden)
            attn_token_mask = self._compute_gate_mask(
                attn_router_scores,
                self.attn_threshold,
                self.attn_min_tokens,
            )

        attn_output, present_key_value = self.self_attn(
            hidden_states=normed_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            token_mask=attn_token_mask,
            router_scores=attn_router_scores,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        normed_hidden = self.post_attention_layernorm(hidden_states)

        if self.use_gated_delta_net:
            delta_router_scores = self.delta_router(normed_hidden)
            delta_token_mask = self._compute_gate_mask(
                delta_router_scores,
                self.delta_threshold,
                self.delta_min_tokens,
            )
            mlp_output = self.mlp(normed_hidden, token_mask=delta_token_mask)
            hidden_states = self.delta_net(
                mlp_output,
                residual,
                gate_values=delta_router_scores,
                token_mask=delta_token_mask,
            )
        else:
            mlp_output = self.mlp(normed_hidden)
            hidden_states = residual + mlp_output

        outputs: Tuple[torch.Tensor, ...] = (hidden_states, present_key_value)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        output_attentions = bool(output_attentions) if output_attentions is not None else False
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if position_ids is None:
            past_length = 0
            if past_key_values is not None and len(past_key_values) > 0:
                past_length = past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                past_length, past_length + seq_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_states.device)

        next_decoder_cache = () if use_cache else None
        all_hidden_states: Tuple[torch.Tensor, ...] = ()
        all_self_attns: Tuple[torch.Tensor, ...] = ()

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            present_key_value = layer_outputs[1]
            attn_weights = layer_outputs[2] if output_attentions and len(layer_outputs) > 2 else None

            if use_cache:
                next_decoder_cache += (present_key_value,)

            if output_attentions:
                all_self_attns += (attn_weights,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            outputs: Tuple = (hidden_states, next_decoder_cache)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attns,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_self_attns if output_attentions else None,
        )


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
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = bool(output_attentions) if output_attentions is not None else False
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False

        if self.gradient_checkpointing:
            use_cache = False
        elif use_cache is None:
            use_cache = getattr(self.config, "use_cache", True)

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            hidden_states = model_outputs.last_hidden_state
            past_key_values = model_outputs.past_key_values
            all_hidden_states = model_outputs.hidden_states
            all_attentions = model_outputs.attentions
        else:
            hidden_states = model_outputs[0]
            past_key_values = model_outputs[1]
            all_hidden_states = model_outputs[2] if output_hidden_states else None
            attention_offset = 2 + (1 if output_hidden_states else 0)
            all_attentions = model_outputs[attention_offset] if output_attentions else None

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output: Tuple = (logits, past_key_values)
            if output_hidden_states:
                output += (all_hidden_states,)
            if output_attentions:
                output += (all_attentions,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
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

