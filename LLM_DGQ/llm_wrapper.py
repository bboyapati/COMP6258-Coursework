import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, CacheLayerMixin

def quantise_tensor(tensor, bits=8):
    q_max = (1 << (bits - 1)) - 1
    q_min = -(1 << (bits - 1))
    
    # Scale across the last dimension
    scale = tensor.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / q_max
    q_tensor = torch.round(tensor / scale).clamp(q_min, q_max)
    
    # Dequantise back to float space for the matrix multiplication math
    dq_tensor = q_tensor * scale
    return dq_tensor.contiguous()


class DGQDynamicLayer(DynamicLayer):
    """
    A single cache layer that quantises KV states on the fly, preserving
    attention sink tokens (e.g. BOS) in full precision.
    """
    def __init__(self, kv_bits=4, num_sink_tokens=1):
        super().__init__()
        self.kv_bits = kv_bits
        self.num_sink_tokens = num_sink_tokens
        self._prefill_done = False

    def update(self, key_states, value_states, *args, **kwargs):
        """
        Quantise KV entries before caching. Sink tokens stay full-precision
        during the initial prefill; all subsequent tokens are quantised.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if not self._prefill_done:
            # Prefill phase
            self._prefill_done = True
            seq_len = key_states.shape[-2]

            if seq_len > self.num_sink_tokens:
                sink_k = key_states[..., :self.num_sink_tokens, :]
                rest_k = quantise_tensor(key_states[..., self.num_sink_tokens:, :], bits=self.kv_bits)
                key_states = torch.cat([sink_k, rest_k], dim=-2)

                sink_v = value_states[..., :self.num_sink_tokens, :]
                rest_v = quantise_tensor(value_states[..., self.num_sink_tokens:, :], bits=self.kv_bits)
                value_states = torch.cat([sink_v, rest_v], dim=-2)
        else:
            # Decode phase: quantise every new token
            key_states = quantise_tensor(key_states, bits=self.kv_bits)
            value_states = quantise_tensor(value_states, bits=self.kv_bits)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values


class DGQKVCache(Cache):
    """
    A custom KV-Cache that quantises cached key/value states to reduce memory
    during autoregressive generation, while preserving attention sink tokens
    (e.g. BOS) in full precision.
    
    Compatible with the modern transformers Cache API (layer-based architecture).
    Adapted from the DGQ paper's attention-aware quantisation principle.
    """
    def __init__(self, kv_bits=4, num_sink_tokens=1):
        super().__init__(layers=[])
        self.kv_bits = kv_bits
        self.num_sink_tokens = num_sink_tokens

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        """
        Override to lazily create DGQDynamicLayer instances (with quantisation)
        for each new layer, then delegate to the standard Cache.update() flow.
        """
        while len(self.layers) <= layer_idx:
            self.layers.append(DGQDynamicLayer(
                kv_bits=self.kv_bits,
                num_sink_tokens=self.num_sink_tokens,
            ))
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

class DGQLinear(nn.Module):
    """
    Wraps the linear layer with logic for DGQ
    """
    def __init__(self, in_features, out_features, outlier_indices, weight_bits=8, act_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        
        if not isinstance(outlier_indices, torch.Tensor):
            outlier_indices = torch.tensor(outlier_indices, dtype=torch.long)
            
        self.register_buffer("outlier_indices", outlier_indices)
        
        # Create a mask to easily separate normal channels from outlier channels
        normal_mask = torch.ones(in_features, dtype=torch.bool)
        if len(outlier_indices) > 0:
            normal_mask[outlier_indices] = False
        self.register_buffer("normal_mask", normal_mask)

        self.register_buffer("w_combined", None)
        self.bias = None

    def prepare_weights(self, weight_data):
        """
        Pre-computes and caches the combined weight matrix. Outlier columns
        keep full-precision weights; normal columns are quantised. This is
        done once so that forward() only needs a single matmul.
        """
        if len(self.outlier_indices) == 0:
            # No outliers: quantise the entire weight matrix
            self.w_combined = quantise_tensor(weight_data, bits=self.weight_bits).contiguous()
        else:
            # Build the combined weight matrix once
            w = weight_data.clone()
            # Quantise only the normal (non-outlier) columns
            w[:, self.normal_mask] = quantise_tensor(
                weight_data[:, self.normal_mask], bits=self.weight_bits
            )
            # Outlier columns remain as-is (full precision)
            self.w_combined = w.contiguous()

    def forward(self, x):
        # Quantise only the normal activation channels; outlier channels
        # stay in full precision (the DGQ principle).
        if len(self.outlier_indices) == 0:
            x_q = quantise_tensor(x, bits=self.act_bits)
        else:
            x_normal = torch.where(self.normal_mask, x, 0.0)
            
            q_max = (1 << (self.act_bits - 1)) - 1
            q_min = -(1 << (self.act_bits - 1))
            scale = x_normal.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / q_max
            
            x_q_all = torch.round(x / scale).clamp(q_min, q_max) * scale
            x_q = torch.where(self.normal_mask, x_q_all, x)
            
        return F.linear(x_q.contiguous(), self.w_combined.contiguous(), self.bias.contiguous() if self.bias is not None else None)

def replace_linear_with_dgq(module, outlier_map, prefix="", weight_bits=4, act_bits=8):
    """
    Recursively replaces all nn.Linear layers with our DGQLinear wrapper.
    """
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # We don't want to quantise the language model head usually
        if isinstance(child, nn.Linear) and "lm_head" not in full_name:
            # If none are provided, it returns an empty list
            outliers = outlier_map.get(full_name, [])
            
            # Create our custom layer
            dgq_layer = DGQLinear(
                child.in_features, 
                child.out_features, 
                outliers, 
                weight_bits=weight_bits, 
                act_bits=act_bits
            )
            
            # Copy the pre-trained weights over and pre-quantise them
            dgq_layer.prepare_weights(child.weight.data)
            if child.bias is not None:
                dgq_layer.bias = nn.Parameter(child.bias.data.clone())
            else:
                dgq_layer.bias = None
                
            # Swap the layer
            print(f"Replacing {full_name} with DGQLinear (weight_bits={weight_bits}, act_bits={act_bits}, outliers={len(outliers)})")
            setattr(module, name, dgq_layer)
        else:
            replace_linear_with_dgq(child, outlier_map, full_name, weight_bits, act_bits)


class DGQModelWrapper(nn.Module):
    """
    A top-level wrapper for the entire LLM.
    """
    def __init__(self, model, outlier_map=None, weight_bits=4, act_bits=8,
                 quantise_kv_cache=False, kv_bits=4, num_sink_tokens=1,
                 compile_model=False):
        super().__init__()
        self.model = model
        self.quantise_kv_cache = quantise_kv_cache
        self.kv_bits = kv_bits
        self.num_sink_tokens = num_sink_tokens
        
        if outlier_map is None:
            outlier_map = {}
            
        # Swap the layers upon initialization
        print("Wrapping model with DGQ Linear layers...")
        replace_linear_with_dgq(
            self.model, 
            outlier_map, 
            weight_bits=weight_bits, 
            act_bits=act_bits
        )
        print("Wrapping complete.")
        
        if self.quantise_kv_cache:
            print(f"KV-cache quantisation enabled: {kv_bits}-bit with {num_sink_tokens} sink token(s) in full precision.")

        if compile_model:
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
            print("Compilation complete.")

    def forward(self, *args, **kwargs):
        # Delegate the forward pass to the wrapped model
        return self.model(*args, **kwargs)
        
    def generate(self, *args, **kwargs):
        if self.quantise_kv_cache and 'past_key_values' not in kwargs:
            kwargs['past_key_values'] = DGQKVCache(
                kv_bits=self.kv_bits,
                num_sink_tokens=self.num_sink_tokens
            )
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        """
        Delegates the Hugging Face save_pretrained method to the underlying wrapped model.
        This saves the quantized state_dict.
        """
        self.model.save_pretrained(save_directory, **kwargs)