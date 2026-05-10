from quant.quant_block import BaseQuantBlock, QuantBasicTransformerBlock, QuantTemporalInformationBlock
from quant.quant_layer import QMODE, QuantLayer, StraightThrough
from quant.quant_model import QuantModel
from quant.adaptive_rounding import AdaRoundQuantizer, RMODE
from quant.reconstruction_util import RLOSS, LossFuncTimeEmbedding
from quant.reconstruction_util import LossFunc
from typing import Tuple
from quant.data_utill import save_inout, save_grad
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import linklink as link
import gc

def update_cached_states_with_quantized_outputs(q_layer, cached_inputs, device):
    """
    Passes the existing CPU-cached inputs through the newly tuned quantized layer,
    and returns the outputs back to the CPU cache to be used by the next layer.
    """
    new_cached_inputs = []
    
    # We do NOT need gradients for this forward pass; we just want the final activations
    q_layer.eval() 
    
    with torch.no_grad():
        for batch_in in cached_inputs:
            # Move inputs to GPU
            hidden_states = batch_in['hidden_states'].to(device)
            attention_mask = batch_in['attention_mask'].to(device)
            position_ids = batch_in['position_ids'].to(device)
            vision_token_mask = batch_in['vision_token_mask'].to(device) if batch_in['vision_token_mask'] is not None else None
            
            # Forward pass through the tuned quantized block
            q_out = q_layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                vision_token_mask=vision_token_mask
            )[0]
            
            # Save the new hidden states back to CPU RAM
            new_cached_inputs.append({
                "hidden_states": q_out.cpu(), 
                "attention_mask": batch_in['attention_mask'], # Masks don't change
                "position_ids": batch_in['position_ids'],     # Position IDs don't change
                "vision_token_mask": batch_in['vision_token_mask']
            })
            
            # Delete GPU tensors
            del hidden_states, attention_mask, position_ids, vision_token_mask, q_out
            
    return new_cached_inputs

def layer_wise_reconstruction_single_gpu(
    quant_model: nn.Module, 
    fp_model: nn.Module, 
    calibration_dataloader, 
    reconstruction_epochs: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda"
):
    """
    VRAM-optimized sequential tuning loop for a single local GPU.
    """
    # 1. Cache the initial embeddings to CPU RAM
    # This ensures we don't need the embedding layers on the GPU anymore
    cached_inputs = cache_initial_hidden_states(fp_model, calibration_dataloader, device)
    
    # 2. Aggressive VRAM Clearing: Offload the entire FP model to CPU
    # We will only pull the specific FP layer we need onto the GPU, one at a time.
    fp_model.cpu()
    quant_model.cpu()
    torch.cuda.empty_cache()
    
    layers = quant_model.model.model.layers
    fp_layers = fp_model.model.model.layers
    
    # 3. Sequential Iteration
    for layer_idx in range(len(layers)):
        print(f"\n--- Reconstructing Layer {layer_idx} / {len(layers)} ---")
        
        q_layer = layers[layer_idx]
        fp_layer = fp_layers[layer_idx]
        
        # Pull ONLY the current Target FP Layer and Quant Layer onto the GPU
        q_layer.to(device)
        fp_layer.to(device)
        fp_layer.eval() # FP layer must be in eval mode
        
        # Turn on gradients for the DGQ step sizes
        trainable_params = enable_quant_param_gradients(q_layer)
        optimizer = optim.Adam(trainable_params, lr=lr)
        
        # Configuration for 16GB VRAM
        gradient_accumulation_steps = 4 
        
        # 4. Block-wise Optimization Loop
        pbar = tqdm(range(reconstruction_epochs), desc=f"Tuning L{layer_idx}")
        for epoch in pbar:
            total_loss = 0
            optimizer.zero_grad() # Zero gradients at the start of the epoch
            
            for step, batch_in in enumerate(cached_inputs):
                hidden_states = batch_in['hidden_states'].to(device).to(torch.bfloat16)
                attention_mask = batch_in['attention_mask'].to(device)
                position_ids = batch_in['position_ids'].to(device)
                vision_token_mask = batch_in['vision_token_mask'].to(device) if batch_in['vision_token_mask'] is not None else None

                # A. Full Precision Forward Pass
                with torch.no_grad():
                    fp_out = fp_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]

                # B. Quantized Forward Pass
                q_out = q_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, vision_token_mask=vision_token_mask)[0]
                
                # C. Compute scaled MSE Loss
                loss = F.mse_loss(q_out, fp_out) / gradient_accumulation_steps
                
                # D. Backpropagate (accumulating gradients)
                loss.backward()
                
                # Step the optimizer only after 'gradient_accumulation_steps' batches
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(cached_inputs):
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += (loss.item() * gradient_accumulation_steps)
                
                # Aggressively delete variables
                del hidden_states, attention_mask, position_ids, vision_token_mask, fp_out, q_out, loss
                
        # 5. Update Cached Inputs for the Next Layer
        print(f"Forwarding quantized states to cache for Layer {layer_idx + 1}...")
        cached_inputs = update_cached_states_with_quantized_outputs(
            q_layer, cached_inputs, device
        )
        
        # 6. Cleanup: Freeze params and kick layers back to CPU
        disable_quant_param_gradients(q_layer)
        q_layer.cpu()
        fp_layer.cpu()
        
        # Force garbage collection to prevent VRAM fragmentation
        gc.collect()
        torch.cuda.empty_cache()

def layer_reconstruction(model: QuantModel,
                         layer: QuantLayer,
                         cali_data: Tuple[torch.Tensor],
                         batch_size: int = 128,
                         iters: int = 20000,
                         w: float = 0.001,
                         opt_mode: RLOSS = RLOSS.MSE,
                         asym: bool = False,
                         include_act_func: bool = True,
                         b_range: tuple = (20, 2),
                         warmup: float = 0.0,
                         use_aq: bool = False,
                         lr: float = 4e-5,
                         p: float = 2.0,
                         multi_gpu: bool = False,
                         keep_gpu=True,
                         **kwargs
                         ) -> None:
    model.set_quant_state(use_wq=False, use_aq=False)
    layer.set_quant_state(use_wq=True, use_aq=use_aq)
    if not include_act_func:
        org_act_func = layer.act_func
        layer.act_func = StraightThrough()

    if not use_aq:
        layer.wqtizer = AdaRoundQuantizer(uaqtizer=layer.wqtizer,
                                            rmode=RMODE.LEARNED_HARD_SIGMOID,
                                            w=layer.original_w.data)
        layer.wqtizer.soft_tgt = True
        opt_params = [layer.wqtizer.alpha]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        opt_params = [layer.aqtizer.delta]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)
    loss_func = LossFunc(o=layer,
                         round_loss=RLOSS.NONE if use_aq else RLOSS.RELAXATION,
                         w=w,
                         max_count=iters,
                         rec_loss=opt_mode,
                         b_range=b_range,
                         decay_start=0.0,
                         warmup=warmup,
                         p=p)
    cached_inputs, cached_outputs = save_inout(model, layer, cali_data, asym, use_aq, batch_size, keep_gpu)
    if opt_mode != RLOSS.MSE:
        cached_grads = save_grad(model, layer, cali_data, asym, use_aq, batch_size, keep_gpu)
    else:
        cached_grads = None
    device = next(layer.parameters()).device
    for i in range(iters):
        idx = torch.randperm(cached_inputs[0].size(0))[: batch_size]
        cur_inputs = (x[idx].to(device=device) for x in cached_inputs) # ^x
        cur_outputs = cached_outputs[idx].to(device=device) # z
        cur_grads = cached_grads[idx].to(device=device) if opt_mode != RLOSS.MSE else None
        optimizer.zero_grad()
        out_quant = layer(*cur_inputs) # ^z
        err = loss_func(out_quant, cur_outputs, cur_grads)
        err.backward(retain_graph=True)
        if multi_gpu:
            for param in opt_params: # output layer does not use quantizer
                if param.grad is not None:
                    link.allreduce(param.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()
    torch.cuda.empty_cache()
    layer.wqtizer.soft_tgt = False
    if not include_act_func:
        layer.act_func = org_act_func



def block_reconstruction(model: QuantModel,
                         block: BaseQuantBlock,
                         cali_data: torch.Tensor,
                         batch_size: int = 32,
                         iters: int = 20000,
                         w: float = 0.01,
                         opt_mode: RLOSS = RLOSS.MSE,
                         asym: bool = False,
                         include_act_func: bool = True,
                         b_range: tuple = (20, 2),
                         warmup: float = 0.0,
                         use_aq: bool = False,
                         lr: float = 4e-5,
                         p: float = 2.0,
                         multi_gpu: bool = True,
                         keep_gpu=True,
                         **kwargs
                         ) -> None:
    model.set_quant_state(use_wq=False, use_aq=False)
    block.set_quant_state(use_wq=True, use_aq=use_aq)

    if not include_act_func:
        org_act_func = block.act_func
        block.act_func = StraightThrough()

    if not use_aq:
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer) and module.quant_emb is False:
                # for shortcut in ResBlock or ResnetBlock, no single layer has shortcut
                if module.split != 0 and QMODE.QDIFF.value in module.aq_mode:
                    module.wqtizer = AdaRoundQuantizer(uaqtizer=module.wqtizer,
                                                        rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                        w=module.original_w.data[:, :module.split, ...])
                    module.wqtizer.soft_tgt = True
                    module.wqtizer1 = AdaRoundQuantizer(uaqtizer=module.wqtizer1,
                                                        rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                        w=module.original_w.data[:, module.split:, ...])
                    module.wqtizer1.soft_tgt = True
                    opt_params += [module.wqtizer.alpha, module.wqtizer1.alpha]
                else:
                    module.wqtizer = AdaRoundQuantizer(uaqtizer=module.wqtizer,
                                                        rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                        w=module.original_w.data)
                    module.wqtizer.soft_tgt = True
                    opt_params.append(module.wqtizer.alpha)
        if len(opt_params) == 0: # for QuantSMVMatMul and QuantQKMatMul
            return
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer) and module.quant_emb is False:
                if module.aqtizer.delta:
                    opt_params.append(module.aqtizer.delta)
                # for shortcut in ResBlock or ResnetBlock, no single layer has shortcut
                if hasattr(module, 'aqtizer1') and module.aqtizer1.delta is not None:
                    opt_params.append(module.aqtizer1.delta)
        A = []
        if isinstance(block, QuantBasicTransformerBlock):
            A = [block.attn1.aqtizer_q, block.attn1.aqtizer_k, block.attn1.aqtizer_v,\
                block.attn2.aqtizer_q, block.attn2.aqtizer_k, block.attn2.aqtizer_v]
            if block.attn1.aqtizer_w.level != (2 ** 16):
                A.append(block.attn1.aqtizer_w)
            if block.attn2.aqtizer_w.level != (2 ** 16):
                A.append(block.attn2.aqtizer_w)
        
        for aqtizer in A:
            if aqtizer.delta:
                opt_params.append(aqtizer.delta)
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)
    loss_func = LossFunc(o=block,
                         round_loss=RLOSS.NONE if use_aq else RLOSS.RELAXATION,
                         w=w,
                         max_count=iters,
                         rec_loss=opt_mode,
                         b_range=b_range,
                         decay_start=0.0,
                         warmup=warmup,
                         p=p)
    cached_inputs, cached_outputs = save_inout(model, block, cali_data, asym, use_aq, batch_size, keep_gpu)
    if opt_mode != RLOSS.MSE:
        cached_grads = save_grad(model, block, cali_data, asym, use_aq, batch_size, keep_gpu)
    else:
        cached_grads = None
    device = next(block.parameters()).device
    for i in range(iters):
        idx = torch.randperm(cached_inputs[0].size(0))[: batch_size]
        cur_inputs = (x[idx].to(device=device) for x in cached_inputs)
        cur_outputs = cached_outputs[idx].to(device=device)
        cur_grads = cached_grads[idx].to(device=device) if opt_mode != RLOSS.MSE else None
        optimizer.zero_grad()

        # ResBlock's split or ResnetBlock's split has been set in save_inout or even before, and cur_inputs does not contain split
        out_quant = block(*cur_inputs)
        err = loss_func(out_quant, cur_outputs, cur_grads)
        err.backward(retain_graph=True)
        if multi_gpu:
            for param in opt_params:
                link.allreduce(param.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()
    torch.cuda.empty_cache()
    for _, module in block.named_modules():
        if isinstance(module, QuantLayer) and module.quant_emb is False:
            if module.split != 0 and QMODE.QDIFF.value in module.aq_mode:
                module.wqtizer.soft_tgt = False
                module.wqtizer1.soft_tgt = False
            else:
                module.wqtizer.soft_tgt = False

    if not include_act_func:
        block.act_func = org_act_func


def tib_reconstruction(block: BaseQuantBlock,
                                  cali_data: torch.Tensor,
                                  batch_size: int = 32,
                                  iters: int = 20000,
                                  w: float = 0.01,
                                  opt_mode: RLOSS = RLOSS.MSE,
                                  asym: bool = False,
                                  include_act_func: bool = True,
                                  b_range: tuple = (20, 2),
                                  warmup: float = 0.0,
                                  use_aq: bool = False,
                                  lr: float = 4e-5,
                                  p: float = 2.0,
                                  multi_gpu: bool = True,
                                  keep_gpu=True) -> None:
    block.set_quant_state(use_wq=True, use_aq=use_aq)

    if not include_act_func:
        org_act_func = block.act_func
        block.act_func = StraightThrough()

    if not use_aq:
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer):
                module.wqtizer = AdaRoundQuantizer(uaqtizer=module.wqtizer,
                                                    rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                    w=module.original_w.data)
                module.wqtizer.soft_tgt = True
                opt_params.append(module.wqtizer.alpha)
        if isinstance(block, QuantTemporalInformationBlock):
            for emb_layers in block.emb_layers:
                for _, module in emb_layers.named_modules():
                    if isinstance(module, QuantLayer):
                        module.wqtizer = AdaRoundQuantizer(uaqtizer=module.wqtizer,
                                                            rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                            w=module.original_w.data)
                        module.wqtizer.soft_tgt = True
                        opt_params.append(module.wqtizer.alpha)
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer):
                if module.aqtizer.delta:
                    opt_params.append(module.aqtizer.delta)
        if isinstance(block, QuantTemporalInformationBlock):
            for emb_layers in block.emb_layers:
                for _, module in emb_layers.named_modules():
                    if isinstance(module, QuantLayer):
                        opt_params.append(module.aqtizer.delta)
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)
    loss_func = LossFuncTimeEmbedding(o=block,
                         round_loss=RLOSS.NONE if use_aq else RLOSS.RELAXATION,
                         w=w,
                         max_count=iters,
                         rec_loss=opt_mode,
                         b_range=b_range,
                         decay_start=0.0,
                         warmup=warmup,
                         p=p)
    cached_inputs, cached_outputs = save_inout(block, block, cali_data, asym, use_aq, batch_size, keep_gpu)
    assert opt_mode == RLOSS.MSE
    device = next(block.parameters()).device
    for i in range(iters):
        idx = torch.randperm(cached_inputs[0].size(0))[: batch_size]
        cur_inputs = (x[idx].to(device=device) for x in cached_inputs)
        cur_outputs = (x[idx].to(device=device) for x in cached_outputs)
        optimizer.zero_grad()
        out_quant = block(*cur_inputs)
        err = loss_func(out_quant, cur_outputs)
        err.backward(retain_graph=True)
        if multi_gpu:
            for param in opt_params:
                if param.grad == None:
                    param.grad = torch.zeros_like(param)
                link.allreduce(param.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()
    torch.cuda.empty_cache()
    for _, module in block.named_modules():
        if isinstance(module, QuantLayer):
            module.wqtizer.soft_tgt = False
    if isinstance(block, QuantTemporalInformationBlock):
        for emb_layers in block.emb_layers:
            for _, module in emb_layers.named_modules():
                if isinstance(module, QuantLayer):
                    module.wqtizer.soft_tgt = False
    else:
        for temb_proj in block.temb_projs:
            assert isinstance(temb_proj, QuantLayer)
            temb_proj.wqtizer.soft_tgt = False
    if not include_act_func:
        block.act_func = org_act_func

def layer_wise_reconstruction(
    quant_model: nn.Module, 
    fp_model: nn.Module, 
    calibration_dataloader, 
    reconstruction_epochs: int = 1000,
    lr: float = 1e-3
):
    """
    Sequentially tunes the quantization scales for each Gemma 3 layer.
    """
    # 1. Cache inputs for the first layer to avoid full model forward passes during tuning
    # (In practice, you run the calibration batches through the embeddings and save the hidden states)
    cached_inputs = cache_initial_hidden_states(fp_model, calibration_dataloader)
    
    layers = quant_model.model.model.layers
    fp_layers = fp_model.model.model.layers
    
    # 2. Iterate sequentially through every transformer layer
    for layer_idx in range(len(layers)):
        print(f"Reconstructing Layer {layer_idx} / {len(layers)}")
        
        q_layer = layers[layer_idx]
        fp_layer = fp_layers[layer_idx]
        
        # Turn on gradients ONLY for the quantization parameters (scales) in this specific layer
        # We do not update the actual model weights.
        trainable_params = enable_quant_param_gradients(q_layer)
        optimizer = optim.Adam(trainable_params, lr=lr)
        
        # 3. Block-wise Optimization Loop
        for epoch in range(reconstruction_epochs):
            total_loss = 0
            
            for batch_in in cached_inputs:
                hidden_states = batch_in['hidden_states']
                attention_mask = batch_in['attention_mask']
                position_ids = batch_in['position_ids']
                vision_token_mask = batch_in['vision_token_mask'] # Specific to our Gemma 3 setup

                # A. Get the Target: Full Precision output of this specific block
                with torch.no_grad():
                    fp_out = fp_layer(
                        hidden_states, 
                        attention_mask=attention_mask, 
                        position_ids=position_ids
                    )[0]

                # B. Get the Quantized output of this block
                q_out = q_layer(
                    hidden_states, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    vision_token_mask=vision_token_mask # Passed down to QuantGemma3Attention
                )[0]
                
                # C. Compute the Reconstruction Loss (MSE)
                # We want the quantized block's output to mimic the full precision block
                loss = F.mse_loss(q_out, fp_out)
                
                # D. Backpropagate and update the quantization scales (step sizes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
        # 4. Crucial Step: Update the cached inputs for the NEXT layer
        # The next layer must be calibrated using the outputs of the NOW QUANTIZED current layer.
        # This prevents error compounding.
        cached_inputs = update_cached_states_with_quantized_outputs(
            q_layer, cached_inputs
        )
        
        # Freeze this layer's quantization parameters before moving to the next
        disable_quant_param_gradients(q_layer)

def enable_quant_param_gradients(layer: nn.Module):
    """Finds all DGQ scale parameters in the block and sets requires_grad=True."""
    params = []
    for name, module in layer.named_modules():
        # Target the scale parameters inside our QuantLayer and QuantGemma3Attention modules
        if hasattr(module, 'act_scale') and module.act_scale is not None:
            module.act_scale.requires_grad = True
            params.append(module.act_scale)
        if hasattr(module, 'weight_scale') and module.weight_scale is not None:
            module.weight_scale.requires_grad = True
            params.append(module.weight_scale)
    return params