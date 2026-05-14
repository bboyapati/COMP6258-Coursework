import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

def quantise_tensor(tensor, bits=8):
    """
    Standard Min-Max Round-To-Nearest (RTN) Quantization.
    """
    q_max = (1 << (bits - 1)) - 1
    q_min = -(1 << (bits - 1))
    
    # Scale across the last dimension
    scale = tensor.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / q_max
    q_tensor = torch.round(tensor / scale).clamp(q_min, q_max)
    
    # Dequantise back to float space for the matrix multiplication math
    dq_tensor = q_tensor * scale
    return dq_tensor


class DGQKVCache(DynamicCache):
    """
    A custom KV-Cache that quantises cached key/value states to reduce memory
    during autoregressive generation, while preserving attention sink tokens
    (e.g. BOS) in full precision.
    
    Adapted from the DGQ paper's attention-aware quantisation principle.
    """
    def __init__(self, kv_bits=4, num_sink_tokens=1):
        super().__init__()
        self.kv_bits = kv_bits
        self.num_sink_tokens = num_sink_tokens
        self.key_cache = []
        self.value_cache = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        Override the default cache update to quantise new KV entries.
        
        During the prefill phase (first call per layer), the sink tokens
        are kept in full precision and the rest are quantised.
        During the decode phase (subsequent calls), every new token's
        KV entry is quantised before being appended.
        """
        if len(self.key_cache) <= layer_idx:
            # Prefill phase: first update for this layer
            seq_len = key_states.shape[-2]
            
            if seq_len > self.num_sink_tokens:
                # Keep sink tokens (e.g. BOS) in full precision
                sink_k = key_states[..., :self.num_sink_tokens, :]
                rest_k = quantise_tensor(key_states[..., self.num_sink_tokens:, :], bits=self.kv_bits)
                key_states = torch.cat([sink_k, rest_k], dim=-2)
                
                sink_v = value_states[..., :self.num_sink_tokens, :]
                rest_v = quantise_tensor(value_states[..., self.num_sink_tokens:, :], bits=self.kv_bits)
                value_states = torch.cat([sink_v, rest_v], dim=-2)
            
            # Store (sink tokens are already full precision, rest quantised)
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Decode phase: quantise the new token's KV and append
            key_q = quantise_tensor(key_states, bits=self.kv_bits)
            value_q = quantise_tensor(value_states, bits=self.kv_bits)
            
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_q], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_q], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

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
        
        # Ensure outlier_indices is a tensor of long integers
        if not isinstance(outlier_indices, torch.Tensor):
            outlier_indices = torch.tensor(outlier_indices, dtype=torch.long)
            
        self.register_buffer("outlier_indices", outlier_indices)
        
        # Create a mask to easily separate normal channels from outlier channels
        normal_mask = torch.ones(in_features, dtype=torch.bool)
        if len(outlier_indices) > 0:
            normal_mask[outlier_indices] = False
        self.register_buffer("normal_mask", normal_mask)

        # We will populate these buffers during conversion
        self.register_buffer("w_normal_q", None)
        self.register_buffer("w_outliers", None)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def prepare_weights(self, weight_data):
        """
        Pre-computes and caches the quantised weights to avoid doing it
        dynamically during every forward pass.
        """
        if len(self.outlier_indices) == 0:
            self.w_normal_q = quantise_tensor(weight_data, bits=self.weight_bits)
        else:
            w_outliers = weight_data[:, self.outlier_indices]
            w_normal = weight_data[:, self.normal_mask]
            
            self.w_outliers = w_outliers.clone()
            
            self.w_normal_q = quantise_tensor(w_normal, bits=self.weight_bits)

    def forward(self, x):
        # If there are no outliers specified, just quantise the activations
        if len(self.outlier_indices) == 0:
            x_q = quantise_tensor(x, bits=self.act_bits)
            return F.linear(x_q, self.w_normal_q, self.bias)

        # 1. Split Inputs (isolate outlier channels)
        x_outliers = x[..., self.outlier_indices]
        x_normal = x[..., self.normal_mask]
        
        # 2. DGQ Quantization
        # Outliers stay full precision, normal activations get quantised dynamically.
        x_normal_q = quantise_tensor(x_normal, bits=self.act_bits)
        
        # 3. Compute Linear projections using the pre-quantised weights
        outlier_out = F.linear(x_outliers, self.w_outliers)
        normal_out = F.linear(x_normal_q, self.w_normal_q)
        
        out = outlier_out + normal_out
        if self.bias is not None:
            out += self.bias
            
        return out

def replace_linear_with_dgq(module, outlier_map, prefix="", weight_bits=4, act_bits=8):
    """
    Recursively replaces all nn.Linear layers with our DGQLinear wrapper.
    """
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # We don't want to quantise the language model head usually
        if isinstance(child, nn.Linear) and "lm_head" not in full_name:
            # Fetch the outlier indices you calibrated for this specific layer
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
            
            # Copy the pre-trained weights over and pre-quantise them!
            dgq_layer.prepare_weights(child.weight.data)
            if child.bias is not None:
                dgq_layer.bias = nn.Parameter(child.bias.data.clone())
            else:
                dgq_layer.bias = None
                
            # Swap the layer
            setattr(module, name, dgq_layer)
        else:
            replace_linear_with_dgq(child, outlier_map, full_name, weight_bits, act_bits)


class DGQModelWrapper(nn.Module):
    """
    A top-level wrapper for the entire LLM.
    """
    def __init__(self, model, outlier_map=None, weight_bits=4, act_bits=8,
                 quantise_kv_cache=False, kv_bits=4, num_sink_tokens=1):
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

    def forward(self, *args, **kwargs):
        # Delegate the forward pass to the wrapped model
        return self.model(*args, **kwargs)
        
    def generate(self, *args, **kwargs):
        # If KV-cache quantisation is enabled, inject our custom cache
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