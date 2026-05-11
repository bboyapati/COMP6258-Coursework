import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import linklink as link
import gc

from quant.data_utill import cache_initial_hidden_states

def update_cached_states_with_quantized_outputs(quant_layer, cached_inputs, device):
    """
    Passes the cached hidden states through the currently quantized layer 
    to generate the input states for the next layer.
    """
    print("Forwarding quantized states to cache for the next layer...")
    quant_layer.eval()
    new_cached_inputs = []
    
    # We need our recursive tuple pusher here too!
    def push_to_device(item):
        if isinstance(item, torch.Tensor):
            return item.to(device)
        if isinstance(item, tuple):
            return tuple(push_to_device(t) for t in item)
        return item

    with torch.no_grad():
        for batch_in in cached_inputs:
            hidden_states = batch_in['hidden_states'].to(device).to(torch.bfloat16)
            
            # 1. Dynamically extract all kwargs (RoPE tuples, attention masks, etc.)
            kwargs = {k: push_to_device(v) for k, v in batch_in.items() if k != 'hidden_states'}
            
            # 2. Aggressively scrub the KV cache state so it doesn't duplicate dimensions
            kwargs['use_cache'] = False
            kwargs.pop('past_key_values', None)
            kwargs.pop('cache_position', None)
            
            # 3. Push through the quantized layer
            out = quant_layer(hidden_states, **kwargs)[0]
            
            # 4. Copy the old kwargs, but update the hidden_states for the next layer
            new_batch = batch_in.copy()
            new_batch['hidden_states'] = out.detach().cpu()
            new_cached_inputs.append(new_batch)
            
            # 5. Aggressive VRAM cleanup
            del hidden_states, kwargs, out
            torch.cuda.empty_cache()
            
    return new_cached_inputs

def layer_wise_reconstruction_single_gpu(
    quant_model: nn.Module, 
    fp_model: nn.Module, 
    calibration_dataloader, 
    reconstruction_epochs: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda",
    init_hidden_states_cache_path: str = "initial_hidden_states.pt"
):
    """
    VRAM-optimized sequential tuning loop for a single local GPU.
    """
    # 1. Cache the initial embeddings to CPU RAM
    # This ensures we don't need the embedding layers on the GPU anymore
    cached_inputs = cache_initial_hidden_states(fp_model, calibration_dataloader, device, init_hidden_states_cache_path)
    
    # 2. Aggressive VRAM Clearing: Offload the entire FP model to CPU
    # We will only pull the specific FP layer we need onto the GPU, one at a time.
    fp_model.cpu()
    quant_model.cpu()
    torch.cuda.empty_cache()
    
    # Grab the wrapped quantized layers using the bulletproof dynamic helper we built earlier
    layers = quant_model._get_decoder_layers()
    
    # Grab the raw full-precision layers using the exact Gemma 3 VLM path
    fp_layers = fp_model.model.language_model.layers
    
    # 3. Sequential Iteration
    for layer_idx in range(len(layers)):
        print(f"\n--- Reconstructing Layer {layer_idx} / {len(layers)} ---")
        
        q_layer = layers[layer_idx]
        fp_layer = fp_layers[layer_idx]
        
        # Pull ONLY the current Target FP Layer and Quant Layer onto the GPU
        q_layer.to(device)
        fp_layer.to(device)
        fp_layer.eval() # FP layer must be in eval mode
        
        # Force the quantizers to dynamically create their learnable parameters
        # by feeding them the first cached batch before we create the optimizer.
        print(f"Initializing quantizer parameters for Layer {layer_idx}...")
        with torch.no_grad():
            dummy_inps = cached_inputs[0]
            d_hidden = dummy_inps['hidden_states'].to(device).to(torch.bfloat16)
            
            # Recursive helper to push tensors AND RoPE tuples to the GPU
            def push_to_device(item):
                if isinstance(item, torch.Tensor):
                    return item.to(device)
                if isinstance(item, tuple):
                    return tuple(push_to_device(t) for t in item)
                return item
            
            # Dynamically push all intercepted kwargs to the device
            kwargs = {k: push_to_device(v) for k, v in dummy_inps.items() if k != 'hidden_states'}

            # Fire the forward pass!
            _ = q_layer(d_hidden, **kwargs)
            
            # Clean up VRAM instantly
            del d_hidden
            torch.cuda.empty_cache()

        # Turn on gradients for the DGQ step sizes
        trainable_params = []
        
        # 1. Freeze all standard weights so we don't accidentally fine-tune the LLM!
        for p in q_layer.parameters():
            p.requires_grad = False
            
        # 2. Aggressively hunt down the quantizer step sizes
        for name, module in q_layer.named_modules():
            # Look for step sizes ('delta' or 'scale') regardless of the class name
            for param_name in ['delta', 'scale']:
                if hasattr(module, param_name) and getattr(module, param_name) is not None:
                    step_size = getattr(module, param_name)
                    
                    # Ensure PyTorch recognizes it as a learnable parameter
                    if not isinstance(step_size, torch.nn.Parameter):
                        step_size = torch.nn.Parameter(step_size)
                        setattr(module, param_name, step_size)
                        
                    step_size.requires_grad = True
                    trainable_params.append(step_size)

        if len(trainable_params) == 0:
            raise RuntimeError("Critical Failure: No quantizer step sizes were found in the module tree!")
        
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
                kwargs = {k: push_to_device(v) for k, v in batch_in.items() if k != 'hidden_states'}
        
                # scrub all variations of the KV Cache
                kwargs.pop('past_key_values', None) # Remove the new plural key
                kwargs.pop('cache_position', None)  # Remove the cache tracker just in case

                # Block-wise tuning processes the full sequence in parallel like standard training.
                # We must disable the cache so the layers don't endlessly append to the same object in-place
                kwargs['use_cache'] = False

                # A. Full Precision Forward Pass
                with torch.no_grad():
                    fp_out = fp_layer(hidden_states, **kwargs)[0]

                # B. Quantized Forward Pass
                q_out = q_layer(hidden_states, **kwargs)[0]
                
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
                del hidden_states, kwargs, fp_out, q_out, loss
                
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

def disable_quant_param_gradients(layer):
    """
    Freezes all parameters in the layer after tuning is complete.
    This clears the autograd graph and saves massive amounts of VRAM.
    """
    for param in layer.parameters():
        param.requires_grad = False