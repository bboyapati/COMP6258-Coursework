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

    def _wrap_single_decoder_layer(self, layer: Gemma3DecoderLayer, layer_idx: int):
        """Replace standard linear projections with DGQ QuantLayers."""
        
        # 1. Wrap the MLP block (gate, up, down projections)
        mlp = layer.mlp
        mlp.gate_proj = QuantLayer(mlp.gate_proj, self.weight_quant_params, self.act_quant_params)
        mlp.up_proj = QuantLayer(mlp.up_proj, self.weight_quant_params, self.act_quant_params)
        mlp.down_proj = QuantLayer(mlp.down_proj, self.weight_quant_params, self.act_quant_params)
        
        self.quant_module_list.extend([mlp.gate_proj, mlp.up_proj, mlp.down_proj])

        # 2. Wrap the Attention block projections (Q, K, V, O)
        attn = layer.self_attn
        attn.q_proj = QuantLayer(attn.q_proj, self.weight_quant_params, self.act_quant_params)
        attn.k_proj = QuantLayer(attn.k_proj, self.weight_quant_params, self.act_quant_params)
        attn.v_proj = QuantLayer(attn.v_proj, self.weight_quant_params, self.act_quant_params)
        attn.o_proj = QuantLayer(attn.o_proj, self.weight_quant_params, self.act_quant_params)
        
        self.quant_module_list.extend([attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj])

        # 3. Replace the attention computation itself to support Logarithmic Attention Quantization
        # We pass layer_idx so the attention module knows if it is a local (sliding-window) or global layer
        quant_attn = QuantGemma3Attention(
            attn, 
            self.weight_quant_params, 
            self.act_quant_params, 
            layer_idx=layer_idx,
            config=self.model.config # <--- Add this line
        )
        layer.self_attn = quant_attn
        
        # NOTE: We do NOT quantize layer.input_layernorm or layer.post_attention_layernorm 
        # to prevent zero-point drift in the hidden dimension outliers.

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