import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from mmlu_pro_dataloader import get_mmlu_pro_dataloader
from llm_wrapper import DGQModelWrapper

def get_outlier_channels(model, dataloader, outlier_fraction=0.01, device="cuda"):
    """
    Runs a calibration dataset through the model and identifies the input channels with the highest activation magnitudes for each linear layer.
    
    Args:
        model: The PyTorch model.
        dataloader: A PyTorch DataLoader yielding dictionaries with 'input_ids', etc.
        outlier_fraction: The fraction of input channels to keep in full precision (e.g., 0.01 = 1%).
        device: The device to run calibration on.
        
    Returns:
        A dictionary mapping layer names (e.g., "model.layers.0.mlp.gate_proj") to a list of outlier channel indices.
    """
    model.eval()
    model.to(device)
    
    # This dictionary will store the max absolute activation for each input channel for each layer
    # Format: { "layer_name": tensor of shape [in_features] }
    channel_max_mags = {}
    
    # Store hooks so we can remove them later
    hooks = []
    
    # Define the forward hook function
    def get_activation_hook(name):
        def hook(module, input, output):
            # input[0] shape: [batch_size, seq_len, in_features]
            x = input[0].detach()
            
            # Find the max absolute value across batch and sequence dimensions
            # Resulting shape: [in_features]
            current_max = x.abs().max(dim=0)[0].max(dim=0)[0]
            
            if name not in channel_max_mags:
                channel_max_mags[name] = current_max
            else:
                # Update the running maximum
                channel_max_mags[name] = torch.maximum(channel_max_mags[name], current_max)
        return hook

    # Register hooks on all linear layers except the lm_head
    print("Registering hooks for calibration...")
    for name, module in tqdm(model.named_modules()):
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            hooks.append(module.register_forward_hook(get_activation_hook(name)))
            
    # Run the calibration data through the model
    print(f"Running {len(dataloader)} batches for calibration...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calibrating"):
            # Extract only the inputs the model expects and move to device
            model_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            
            # Forward pass
            model(**model_inputs)
            
    # Clean up hooks so the model returns to normal
    for hook in hooks:
        hook.remove()
        
    # Process the recorded maximums to find the top K outlier channels
    print("Processing activation magnitudes to find outliers...")
    outlier_map = {}
    
    for name, max_mags in tqdm(channel_max_mags.items()):
        in_features = max_mags.shape[0]
        
        # Calculate how many channels represent the desired fraction
        num_outliers = max(1, int(in_features * outlier_fraction)) 
        
        # Get the indices of the channels with the highest magnitudes
        _, topk_indices = torch.topk(max_mags, k=num_outliers)
        
        # Convert to a standard Python list
        outlier_map[name] = topk_indices.cpu().tolist()
        
    print(f"Calibration complete! Found outliers for {len(outlier_map)} layers.")
    return outlier_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--outlier_fraction", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--act_bits", type=int, default=8)
    args = parser.parse_args()
    print("Quantising model: ", args.model_name, " with outlier fraction: ", args.outlier_fraction, " using batch size: ", args.batch_size, " on device: ", args.device)
    
    print("Loading tokeniser and model...")
    tokeniser = AutoTokeniser.from_pretrained(args.model_name)
    # Ensure a pad token exists for batching
    if tokeniser.pad_token_id is None:
        tokeniser.pad_token = tokeniser.eos_token
        
    print("Loading calibration dataset (MMLU-Pro)...")
    dataloader = get_mmlu_pro_dataloader(batch_size=args.batch_size, tokeniser=tokeniser)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16)
    
    print("Finding outliers...")
    outlier_map = get_outlier_channels(model, dataloader, args.outlier_fraction, args.device)

    print("Quantising model...")
    quantised_model = DGQModelWrapper(
        model, 
        outlier_map, 
        weight_bits=args.weight_bits, 
        act_bits=args.act_bits
    )

    print("Saving quantised model...")
    quantised_model.save_pretrained(f"{args.model_name}_quantised_w{args.weight_bits}_a{args.act_bits}")
    tokeniser = AutoTokenizer.from_pretrained(args.model_name)
    tokeniser.save_pretrained(f"{args.model_name}_quantised_w{args.weight_bits}_a{args.act_bits}")

if __name__ == "__main__":
    main()