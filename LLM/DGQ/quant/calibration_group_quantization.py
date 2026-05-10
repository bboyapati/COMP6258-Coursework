from typing import Any, Dict, Tuple, Union
import torch.nn as nn
import torch
import numpy as np
from quant.quant_block import BaseQuantBlock
from quant.quant_model import QuantModel
from quant.quant_layer import QuantLayer
from quant.adaptive_rounding import AdaRoundQuantizer, RMODE
from quant.quant_layer import UniformAffineQuantizer
from tqdm import trange

from quant.reconstruction import block_reconstruction, layer_reconstruction, tib_reconstruction
import linklink as dist
import logging
import os
import yaml
logger = logging.getLogger(__name__)

def uaq2adar(model: nn.Module):
    for _, child in model.named_children():
        if isinstance(child, QuantLayer):
            if not child.ignore_recon:
                child.wqtizer = AdaRoundQuantizer(child.wqtizer,
                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                w = child.original_w.data)
        elif isinstance(child, BaseQuantBlock):
            if not child.ignore_recon:
                for _, sub_child in child.named_modules():
                    if isinstance(sub_child, QuantLayer):
                        if not hasattr(sub_child, 'wqtizer1'):
                            sub_child.wqtizer = AdaRoundQuantizer(sub_child.wqtizer,
                                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                                w = sub_child.original_w.data)
                        else:
                            sub_child.wqtizer = AdaRoundQuantizer(sub_child.wqtizer,
                                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                                w = sub_child.original_w.data[:, :sub_child.split, ...])
                            sub_child.wqtizer1 = AdaRoundQuantizer(sub_child.wqtizer1,
                                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                                w = sub_child.original_w.data[:, sub_child.split:, ...])
        else:
            uaq2adar(child)

@torch.no_grad()
def cali_model_aq(model_type, qnn: QuantModel, a_cali_data, model_dict, group_num, interval, group_mode):
    qnn.cuda()
    qnn.eval()
    cali_data = a_cali_data
    for time in range(cali_data[0].shape[0] // interval):
        t_cali_data = tuple([x[time * interval: (time + 1) * interval] for x in cali_data])
        
        qnn.set_quant_state(use_wq = True, use_aq = True)
        qnn.disable_out_quantization()

        for name, module in qnn.model.named_modules():
            if 'aqtizer' in name:
                if isinstance(module, UniformAffineQuantizer):
                    del module.delta
                    del module.zero_point
                    module.delta = None
                    module.zero_point = None
                    module.init = False
                else:
                    del module.delta
                    module.delta = None
                    module.init = False

        # --------- activation quantization -------- #
        # batch_size
        if model_type == 'sd':
            batch_size = min(8, t_cali_data[0].shape[0]) 
        elif model_type == 'sdxl':
            batch_size = min(4, t_cali_data[0].shape[0])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # calibrate activation quantization
        with torch.no_grad():
            inds = np.random.choice(t_cali_data[0].shape[0], batch_size, replace=False)
            inputs = (x[inds].cuda() for x in t_cali_data)
            _ = qnn(*inputs)
            logger.info(f'group_num: {group_num} running stat for activation calibration...')
            inds = np.arange(t_cali_data[0].shape[0])
            np.random.shuffle(inds)
            qnn.set_group_num(group_num)
            for i in trange(0, t_cali_data[0].shape[0], batch_size):
                inputs = (x[inds[i: i + batch_size]].cuda() for x in t_cali_data)
                _ = qnn(*inputs)
            qnn.done_group_num(group_num, mode=group_mode)
            logger.info(f'group_num: {group_num} running stat for activation calibration done.')
            torch.cuda.empty_cache()

        # save the quantization parameters
        for name, module in qnn.model.named_modules():
            if 'aqtizer' in name:
                if isinstance(module, UniformAffineQuantizer) and module.delta is not None:
                    if not torch.is_tensor(module.zero_point):
                        module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                    else:
                        module.zero_point = nn.Parameter(module.zero_point)
    
        temp = {}
        for name, module in qnn.model.named_modules():
            if 'aqtizer' in name and len(list(module.cpu().state_dict().keys())) == 2:
                temp['model.' + name + '.delta'] = module.cpu().state_dict()['delta']
                temp['model.' + name + '.zero_point'] = module.cpu().state_dict()['zero_point']
        model_dict['act_{}'.format(time)] = temp
    
    return model_dict

class Gemma3GroupQuantizer:
    def __init__(self, act_quant_params: dict):
        self.num_hidden_groups = act_quant_params.get('num_groups', 4) # Number of G groups for hidden dims
        self.act_bits = act_quant_params.get('act_bits', 8)
        
    def _sort_and_group_hidden_dims(self, calibration_activations: torch.Tensor):
        """
        LLMs exhibit massive, systematic magnitude spikes in specific hidden dimensions.
        We sort dimensions by their maximum absolute activation and chunk them into G groups.
        """
        # calibration_activations shape: [batch, seq_len, hidden_size]
        # Get max absolute value for each hidden dimension across the calibration batch/sequence
        max_vals = calibration_activations.abs().amax(dim=(0, 1))
        
        # Sort hidden dimensions by magnitude
        sorted_indices = torch.argsort(max_vals)
        
        # Split the sorted indices into G groups
        # Group 0 will have the smallest activations, Group G-1 will contain the massive outliers
        hidden_groups = torch.tensor_split(sorted_indices, self.num_hidden_groups)
        
        return hidden_groups

    def _group_tokens_by_modality(self, seq_len: int, vision_token_mask: torch.Tensor = None):
        """
        Gemma 3 mixes dense SigLIP vision tokens with sparse text tokens.
        We must group them separately because their activation distributions are wildly different.
        """
        all_indices = torch.arange(seq_len)
        
        # 1. Isolate Attention Sinks
        bos_token_idx = torch.tensor([0]) # Always index 0
        
        # 2. Isolate Vision Tokens
        if vision_token_mask is not None:
            # Assuming mask is 1 for vision tokens, 0 for text
            vision_indices = vision_token_mask[0].nonzero(as_tuple=True)[0]
        else:
            vision_indices = torch.tensor([], dtype=torch.long)
            
        # 3. Isolate Standard Text Tokens
        protected_set = set(bos_token_idx.tolist() + vision_indices.tolist())
        text_indices = torch.tensor([idx for idx in all_indices.tolist() if idx not in protected_set])
        
        return bos_token_idx, vision_indices, text_indices

    def get_quantization_scales(self, activations: torch.Tensor, vision_token_mask: torch.Tensor = None):
        """
        Calculates specific quantization scales (step size, zero-point) for each group.
        """
        bsz, seq_len, hidden_size = activations.shape
        
        hidden_groups = self._sort_and_group_hidden_dims(activations)
        bos_idx, vision_indices, text_indices = self._group_tokens_by_modality(seq_len, vision_token_mask)
        
        scales = {}
        zero_points = {}
        
        # We calculate scales separately for Text Tokens and Vision Tokens
        # across each of the G hidden dimension groups.
        for g_idx, dim_indices in enumerate(hidden_groups):
            
            # --- Text Token Quantization ---
            if len(text_indices) > 0:
                text_acts = activations[:, text_indices, :][:, :, dim_indices]
                t_scale, t_zp = self._calc_minmax_params(text_acts)
                scales[f'group_{g_idx}_text'] = t_scale
                zero_points[f'group_{g_idx}_text'] = t_zp
                
            # --- Vision Token Quantization ---
            if len(vision_indices) > 0:
                vision_acts = activations[:, vision_indices, :][:, :, dim_indices]
                v_scale, v_zp = self._calc_minmax_params(vision_acts)
                scales[f'group_{g_idx}_vision'] = v_scale
                zero_points[f'group_{g_idx}_vision'] = v_zp
                
            # Note: We do NOT calculate scales for bos_idx because it remains in Full Precision
            
        return hidden_groups, bos_idx, vision_indices, text_indices, scales, zero_points

    def _calc_minmax_params(self, tensor: torch.Tensor):
        """Standard Min-Max asymmetric quantization parameter calculation."""
        q_min = 0
        q_max = (1 << self.act_bits) - 1
        
        min_val = tensor.min()
        max_val = tensor.max()
        
        scale = (max_val - min_val).clamp(min=1e-5) / (q_max - q_min)
        zero_point = q_min - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, q_min, q_max)
        
        return scale, zero_point

def act_group_quant(model_type,
                    qnn: QuantModel,
                      a_cali_data: Tuple[torch.Tensor],
                      path: str = None,
                      group_num: int = 1,
                      interval: int = 128,
                      group_mode = 'minmax',
                      **kwargs
                      ) -> None:
    logger.info("Calibrating...")

    # --------- activation quantization -------- #
    model_dict = {}
    model_dict = cali_model_aq(model_type, qnn, a_cali_data, model_dict, group_num, interval, group_mode=group_mode)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model_dict, path)
    logger.info("calibrated model saved to {}".format(path))
    logger.info("Calibration done.")
    