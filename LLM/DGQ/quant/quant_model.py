import torch
import torch.nn as nn
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

# Corrected: Importing QuantLayer instead of QuantModule
from .quant_layer import QuantLayer, QMODE 
# This will be our custom attention wrapper to handle Gemma 3's Local/Global layers
from .quant_block import QuantGemma3Attention 

class QuantGemma3Model(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Wraps a Hugging Face Gemma 3 model for Distribution-Aware Group Quantization.
        """
        super().__init__()
        self.model = model 
        self.quant_module_list = []
        
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        
        target_layers = self._get_decoder_layers()
        self._wrap_gemma_layers(target_layers)

    def _get_decoder_layers(self):
        """
        Safely extracts the sequential transformer layers regardless of how the VLM is wrapped.
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            return self.model.model.language_model.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            return self.model.language_model.layers
        else:
            print(self.model)
            raise AttributeError("Could not dynamically locate the text decoder layers.")

    def _wrap_gemma_layers(self, layers: nn.ModuleList):
        """Iterate through the sequential decoder layers."""
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, Gemma3DecoderLayer):
                self._wrap_single_decoder_layer(layer, layer_idx)

    def _wrap_single_decoder_layer(self, layer, layer_idx: int):
        """Replace standard linear projections with DGQ QuantLayers."""
        
        # Sanitize the parameters so the original DGQ QuantLayer doesn't crash
        # from our custom VLM variables
        custom_keys = ['attn_bits', 'log_base', 'num_groups']
        ql_weight_params = {k: v for k, v in self.weight_quant_params.items() if k not in custom_keys}
        ql_act_params = {k: v for k, v in self.act_quant_params.items() if k not in custom_keys}

        # 1. Wrap the MLP block (gate, up, down projections) using the SANITIZED dicts
        mlp = layer.mlp
        mlp.gate_proj = QuantLayer(mlp.gate_proj, ql_weight_params, ql_act_params)
        mlp.up_proj = QuantLayer(mlp.up_proj, ql_weight_params, ql_act_params)
        mlp.down_proj = QuantLayer(mlp.down_proj, ql_weight_params, ql_act_params)

        # 2. Wrap the Attention block projections (Q, K, V, O)
        attn = layer.self_attn
        attn.q_proj = QuantLayer(attn.q_proj, ql_weight_params, ql_act_params)
        attn.k_proj = QuantLayer(attn.k_proj, ql_weight_params, ql_act_params)
        attn.v_proj = QuantLayer(attn.v_proj, ql_weight_params, ql_act_params)
        attn.o_proj = QuantLayer(attn.o_proj, ql_weight_params, ql_act_params)

        # 3. Replace the attention computation itself
        # Notice we pass the FULL un-sanitized dicts here because our custom 
        # QuantGemma3Attention class actually uses log_base and attn_bits.
        quant_attn = QuantGemma3Attention(
            attn, 
            self.weight_quant_params, 
            self.act_quant_params, 
            layer_idx=layer_idx,
            config=self.model.config
        )
        layer.self_attn = quant_attn

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        """Pass the forward call directly to the wrapped HF model."""
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, # For the SigLIP vision tokens
            **kwargs
        )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """Toggle quantization on and off during the block-wise reconstruction phase."""
        for m in self.model.modules():
            if isinstance(m, (QuantLayer, QuantGemma3Attention)):
                m.set_quant_state(weight_quant, act_quant)