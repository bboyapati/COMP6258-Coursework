from typing import Dict, Tuple

import torch
import torch as th
import torch.nn as nn
import math

class QuantGemma3Attention(nn.Module):
    def __init__(self, orig_attn: nn.Module, weight_quant_params: dict, act_quant_params: dict, layer_idx: int, config=None):
        super().__init__()
        self.orig_attn = orig_attn
        self.layer_idx = layer_idx
        
        # Gemma 3 is a Vision-Language Model. 
        # Its global config is a composite, and the text transformer parameters 
        # are nested specifically inside 'text_config'.
        text_config = getattr(config, "text_config", config) if config is not None else None
        
        if text_config is not None:
            # Prioritize the text_config, but fallback to orig_attn if needed
            self.num_heads = getattr(text_config, "num_attention_heads", getattr(orig_attn, "num_heads", None))
            self.num_kv_heads = getattr(text_config, "num_key_value_heads", getattr(orig_attn, "num_key_value_heads", self.num_heads))
            
            hidden_size = getattr(text_config, "hidden_size", 2304)
            fallback_dim = hidden_size // self.num_heads if self.num_heads else 256
            self.head_dim = getattr(text_config, "head_dim", getattr(orig_attn, "head_dim", fallback_dim))
            
            self.sliding_window = getattr(text_config, "sliding_window", getattr(orig_attn, "sliding_window", None))
        else:
            # Absolute fallback if neither config nor text_config is found
            self.num_heads = getattr(orig_attn, "num_heads", 8)
            self.num_kv_heads = getattr(orig_attn, "num_key_value_heads", 4)
            self.head_dim = getattr(orig_attn, "head_dim", 256)
            self.sliding_window = getattr(orig_attn, "sliding_window", None)
        
        # Determine if this is a local (sliding window) or global layer
        self.is_local_layer = (self.sliding_window is not None) and (self.sliding_window > 0)

        # parent decoder layer strictly looks for during the forward pass.
        self.is_sliding = getattr(orig_attn, "is_sliding", self.is_local_layer)
        
        # Quantization state
        self.act_quant = False
        
        # Extract attention quantization params from DGQ config
        self.attn_bits = act_quant_params.get('attn_bits', 8)
        self.log_base = act_quant_params.get('log_base', 2.0)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.act_quant = act_quant

    def _quantize_attention_probs(self, attn_probs: torch.Tensor, vision_token_indices: list):
        """
        Applies DGQ's Logarithmic Quantization to the Softmax probabilities.
        Protects the BOS token and Vision tokens from quantization to preserve attention sinks.
        """
        if not self.act_quant:
            return attn_probs

        # 1. Store full precision values for attention sinks
        # BOS is typically at index 0. vision_token_indices contains the dense SigLIP sequence range.
        protected_indices = [0] + vision_token_indices
        fp_sinks = attn_probs[..., :, protected_indices].clone()

        # 2. Logarithmic Quantization of the probability distribution
        # attn_probs are between 0 and 1. We map to log space, quantize, and map back.
        eps = 1e-8
        log_probs = torch.log(attn_probs + eps) / math.log(self.log_base)
        
        # Scale and round to target bits
        q_max = (1 << self.attn_bits) - 1
        scale = q_max / torch.abs(log_probs).max().clamp(min=1e-5)
        
        quantized_log_probs = torch.round(log_probs * scale) / scale
        
        # Dequantize back to linear probability space
        quantized_probs = torch.pow(self.log_base, quantized_log_probs)

        # 3. Restore the full precision attention sinks
        quantized_probs[..., :, protected_indices] = fp_sinks

        # 4. Re-normalize to ensure probabilities still sum to 1
        quantized_probs = quantized_probs / quantized_probs.sum(dim=-1, keepdim=True)
        
        return quantized_probs
    
    def forward(self, *args, **kwargs):
        """
        Delegates the complex Multimodal RoPE and Sliding Window math 
        back to the native Gemma 3 Attention module. 
        
        Because we already wrapped the q, k, v, and o projections with QuantLayers 
        during initialization, orig_attn will automatically quantize the matrix 
        multiplications for us
        """
        return self.orig_attn(*args, **kwargs)

    # def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
    #     bsz, q_len, _ = hidden_states.size()

    #     # 1. Compute Q, K, V using the QuantLayer projections
    #     query_states = self.orig_attn.q_proj(hidden_states)
    #     key_states = self.orig_attn.k_proj(hidden_states)
    #     value_states = self.orig_attn.v_proj(hidden_states)

    #     query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    #     key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    #     value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    #     # Apply Gemma's RoPE (Rotary Position Embeddings)
    #     # Note: orig_attn.rotary_emb handles the 1M base freq for global layers automatically
    #     cos, sin = self.orig_attn.rotary_emb(value_states, position_ids)
    #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    #     # 2. Handle Grouped-Query Attention (GQA)
    #     # Broadcast K and V to match the number of Q heads
    #     key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_heads // self.num_kv_heads)
    #     value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_heads // self.num_kv_heads)

    #     # 3. Compute Raw Attention Scores
    #     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    #     if attention_mask is not None:
    #         # If this is a local layer, the attention_mask provided by the Gemma 3 processor 
    #         # will already contain the sparse sliding window masking.
    #         attn_weights = attn_weights + attention_mask

    #     # 4. Softmax
    #     attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    #     # 5. DGQ: Modality-Aware Logarithmic Quantization
    #     # Detect where the vision tokens are in the sequence. 
    #     # (Assuming kwargs['vision_token_mask'] is passed from our calibration dataloader)
    #     vision_indices = []
    #     if 'vision_token_mask' in kwargs and kwargs['vision_token_mask'] is not None:
    #         vision_indices = kwargs['vision_token_mask'][0].nonzero(as_tuple=True)[0].tolist()

    #     attn_probs = self._quantize_attention_probs(attn_probs, vision_token_indices=vision_indices)

    #     # 6. Final output projection
    #     attn_output = torch.matmul(attn_probs, value_states)
    #     attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        
    #     attn_output = self.orig_attn.o_proj(attn_output)

    #     return attn_output, None, past_key_value
