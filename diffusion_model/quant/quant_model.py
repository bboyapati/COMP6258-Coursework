from typing import List
import torch.nn as nn
import torch

from collections import namedtuple
from quant.quant_block import QuantBasicTransformerBlock, QuantResnetBlock2D, QuantTemporalInformationBlock, b2qb, BaseQuantBlock, QuantFlux2Attention, QuantFlux2TransformerBlock, QuantFlux2SingleTransformerBlock
from quant.quant_layer import QMODE, QuantLayer, StraightThrough

from quant.adaptive_rounding import AdaRoundQuantizer
from quant.quant_layer import UniformAffineQuantizer
from quant.quant_block import T2ILogQuantizer
class CFG:
    in_channels = 0
    sample_size = 0
    time_cond_proj_dim = 0
    addition_time_embed_dim = 0

class QuantModel(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 wq_params: dict = {},
                 aq_params: dict = {},
                 softmax_aq_params: dict = {},
                 cali: bool = True,
                 tib_recon: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.model = model

        # for diffusers pipeline
        self.config = CFG()
        
        self.config.in_channels = model.config.in_channels
        if hasattr(model.config, 'sample_size'):
            self.config.sample_size = model.config.sample_size
        if hasattr(model.config, 'time_cond_proj_dim'):
            self.config.time_cond_proj_dim = model.config.time_cond_proj_dim
        if hasattr(model.config, 'addition_time_embed_dim'):
            self.config.addition_time_embed_dim = model.config.addition_time_embed_dim
        self.tib_recon = tib_recon
        

        self.B = b2qb()
        self.quant_module(self.model, wq_params, aq_params, 
                          aq_mode=kwargs.get("aq_mode", [QMODE.NORMAL.value]), 
                          prev_name=None)
        self.quant_block(self.model, wq_params, aq_params,
                         softmax_aq_params)
        if cali and tib_recon:
            self.get_tib(self.model, wq_params, aq_params)
        self.fix_flux2_attention_refs()
            


    def get_tib(self,
                    module: nn.Module,
                    wq_params: dict = {},
                    aq_params: dict = {},
                    ) -> QuantTemporalInformationBlock:
        for name, child in module.named_children():
            if name == 'time_embedding':
                self.tib = QuantTemporalInformationBlock(child, aq_params)
            elif isinstance(child, QuantResnetBlock2D):
                self.tib.add_emb_layer(child.time_emb_proj)
            else:
                self.get_tib(child, wq_params, aq_params)


    def quant_module(self,
                     module: nn.Module,
                     wq_params: dict = {},
                     aq_params: dict = {},
                     aq_mode: List[int] = [QMODE.NORMAL.value],
                     prev_name: str = None,
                     ) -> None:
        for name, child in module.named_children():
            if "attention" in child.__class__.__name__.lower():
                continue
            elif isinstance(child, tuple(QuantLayer.QMAP.keys())):
                if name in ['time_embedding', 'time_proj', 'time_emb_proj'] and self.tib_recon: 
                    # for keep time embedding while block reconstruction
                    # refer to TFMQ-DM
                    setattr(module, name, QuantLayer(child, wq_params, aq_params, aq_mode=aq_mode, quant_emb=True))
                else:
                    setattr(module, name, QuantLayer(child, wq_params, aq_params, aq_mode=aq_mode))
                    
            elif isinstance(child, StraightThrough):
                continue
            else:
                self.quant_module(child, wq_params, aq_params, aq_mode=aq_mode, prev_name=name)

    def quant_block(self, module, wq_params={}, aq_params={}, softmax_aq_params={}):

        for name, child in module.named_children():

            cls_name = child.__class__.__name__

            if cls_name in self.B:

                wrapped = self.B[cls_name](child, aq_params, softmax_aq_params)

                setattr(module, name, wrapped)

                # 🔥 CRITICAL: force attribute re-resolution
                if hasattr(module, "__dict__"):
                    module.__dict__[name] = wrapped

                # 🔥 also rebind common Flux2 cached references
                if hasattr(module, "_modules"):
                    module._modules[name] = wrapped

            else:
                self.quant_block(child, wq_params, aq_params, softmax_aq_params)

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.model.modules():
            if isinstance(m, (BaseQuantBlock, QuantLayer)):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # def forward(
    #     self, sample, timesteps, encoder_hidden_states, *args, **kwargs
    #     ):
    #     return self.model(sample, timesteps, encoder_hidden_states, *args, **kwargs)

    # def disable_out_quantization(self) -> None:
    #     # conv_in, conv_out are too much sensitive to quantization
    #     self.model.conv_in.use_wq = False
    #     self.model.conv_in.disable_aq = True

    #     self.model.conv_out.use_wq = False
    #     self.model.conv_out.disable_aq = True

    def disable_out_quantization(self) -> None:
        """
        Disable sensitive quantization layers for any supported architecture
        (UNet-style or Transformer-style like Flux2).
        """

        # Traverse all submodules instead of assuming architecture
        for m in self.model.modules():
            # Only operate on quantized layers
            if hasattr(m, "use_wq"):
                m.use_wq = False

            if hasattr(m, "disable_aq"):
                m.disable_aq = True

    def synchorize_activation_statistics(self):
        import linklink.dist_helper as dist
        for module in self.modules():
            if isinstance(module, QuantLayer):
                if module.aqtizer.delta is not None:
                    dist.allaverage(module.aqtizer.delta)


    def set_group_num(self,
                         group_num: int = 1
                         ) -> None:
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.set_group_num(group_num)
            if isinstance(m, QuantBasicTransformerBlock):
                m.attn1.aqtizer_q.group_num = group_num
                m.attn1.aqtizer_k.group_num = group_num
                m.attn1.aqtizer_v.group_num = group_num
                
                m.attn2.aqtizer_q.group_num = group_num
                m.attn2.aqtizer_k.group_num = group_num
                m.attn2.aqtizer_v.group_num = group_num
                
                
    def done_group_num(self,
                       group_num,
                       mode
                       ) -> None:
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.done_group_num(group_num, mode=mode)
            if isinstance(m, QuantBasicTransformerBlock):
                m.attn1.aqtizer_q.done_group_num(group_num, mode=mode)
                m.attn1.aqtizer_k.done_group_num(group_num, mode=mode)
                m.attn1.aqtizer_v.done_group_num(group_num, mode=mode)

                m.attn2.aqtizer_q.done_group_num(group_num, mode=mode)
                m.attn2.aqtizer_k.done_group_num(group_num, mode=mode)
                m.attn2.aqtizer_v.done_group_num(group_num, mode=mode)
                
    def set_running_stat(self,
                         running_stat: bool = False
                         ) -> None:
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                m.attn1.aqtizer_q.running_stat = running_stat
                m.attn1.aqtizer_k.running_stat = running_stat
                m.attn1.aqtizer_v.running_stat = running_stat
                m.attn1.aqtizer_w.running_stat = running_stat
                m.attn2.aqtizer_q.running_stat = running_stat
                m.attn2.aqtizer_k.running_stat = running_stat
                m.attn2.aqtizer_v.running_stat = running_stat
                m.attn2.aqtizer_w.running_stat = running_stat
            elif isinstance(m, QuantLayer):
                m.set_running_stat(running_stat)

    def half(self):
        print("QuantModel half")
        super().half()
        for m in self.model.modules():
            if isinstance(m, (AdaRoundQuantizer, UniformAffineQuantizer)):
                m.half()
            if isinstance(m, QuantLayer):
                m.half()
        return self
    
    def float(self):
        print("QuantModel float")
        super().float()
        for m in self.model.modules():
            if isinstance(m, (AdaRoundQuantizer, UniformAffineQuantizer)):
                m.float()
            if isinstance(m, QuantLayer):
                m.float()
        return self
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def cache_context(self, *args, **kwargs):
        return self.model.cache_context(*args, **kwargs)
    
    def fix_flux2_attention_refs(self):
        for m in self.modules():
            if hasattr(m, "attn") and hasattr(m.attn, "forward"):
                # force Python attribute resolution refresh
                m.attn = m.attn