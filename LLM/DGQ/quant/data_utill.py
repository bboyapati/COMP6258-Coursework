import torch
import os
from torch.utils.data import DataLoader, Subset
from transformers import Gemma3Processor
from datasets import load_dataset
from typing import Dict, Union
import torch.nn.functional as F
import torch.nn as nn
from quant.quant_layer import QuantLayer
from typing import Union, Tuple
import logging
logger = logging.getLogger(__name__)


def build_gemma3_calibration_loader(
    model_id: str, 
    dataset_name: str = "HuggingFaceM4/the_cauldron", # Standard interleaved image-text dataset
    batch_size: int = 1,
    num_calibration_samples: int = 64,
    max_seq_length: int = 1024
):
    """
    Builds a PyTorch DataLoader that feeds interleaved image/text pairs to the DGQ reconstruction loop.
    """
    # 1. Load the Gemma 3 Processor (handles both Tokenization and SigLIP Image processing)
    processor = Gemma3Processor.from_pretrained(model_id, use_fast=True)
    
    # 2. Load and subset the calibration dataset
    # You only need a small representative subset to calibrate the scales
    dataset = load_dataset(dataset_name, "vqav2", split="train")
    dataset = Subset(dataset, range(num_calibration_samples))

    # 3. The DGQ-Specific Collate Function
    def collate_fn(batch):
        texts = []
        images = []
        for item in batch:
            # Extract the Cauldron text data
            conversation = item["texts"][0] 
            user_text = conversation["user"]
            assistant_text = conversation["assistant"]
            
            # 1. Use the native Hugging Face message dictionary format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"}, # Let the template handle the exact image token insertion
                        {"type": "text", "text": user_text}
                    ]
                },
                {
                    "role": "assistant", # Hugging Face maps 'assistant' to Gemma's internal 'model' token
                    "content": [
                        {"type": "text", "text": assistant_text}
                    ]
                }
            ]
            
            # 2. Convert the dictionary into Gemma 3's strict control-token string
            # tokenize=False returns the raw string with the exact <image> token layout the processor expects
            formatted_text = processor.apply_chat_template(messages, tokenize=False)
            texts.append(formatted_text)
            
            # 3. Append the PIL image
            images.append([item["images"][0]])
            
        # 4. Pass the perfectly formatted strings and images to the processor
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length
        )
        return inputs

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def cache_initial_hidden_states(fp_model, calibration_dataloader, device, cache_path: str = "gemma3_initial_states.pt"):
    """
    Intercepts the multimodal embeddings right before they enter the first transformer layer.
    """
    if os.path.exists(cache_path):
        print(f"Loading pre-computed hidden states from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    print("Intercepting initial hidden states via forward hook...")
    fp_model.eval()
    cached_inputs = []

    # Safely locate the first layer using the path we mapped out earlier
    first_layer = fp_model.model.language_model.layers[0]

    # Create a custom exception to cleanly abort the forward pass
    class StopForwardException(Exception):
        pass

    for batch in calibration_dataloader:
        # Move inputs to GPU for the embedding phase
        model_device = next(fp_model.parameters()).device
        batch = {k: v.to(model_device) for k, v in batch.items()}
        inps = {}

        # This function executes right before the first layer calculates anything
        def cache_hook(module, args, kwargs):
            inps['hidden_states'] = args[0].detach().cpu()
            
            # Recursive helper to move tensors AND tuples of tensors to CPU
            def detach_item(item):
                if isinstance(item, torch.Tensor):
                    return item.detach().cpu()
                if isinstance(item, tuple):
                    return tuple(detach_item(t) for t in item)
                return item

            # 1. Catch everything passed as a keyword argument
            for k, v in kwargs.items():
                inps[k] = detach_item(v)
                
            # 2. Catch anything passed positionally (just in case)
            arg_names = ['hidden_states', 'attention_mask', 'position_ids', 'past_key_value', 'output_attentions', 'use_cache', 'cache_position', 'position_embeddings_global', 'position_embeddings_local']
            for i in range(1, len(args)):
                if i < len(arg_names):
                    inps[arg_names[i]] = detach_item(args[i])

            # Abort the forward pass
            raise StopForwardException

        # Attach the wiretap
        handle = first_layer.register_forward_pre_hook(cache_hook, with_kwargs=True)

        try:
            with torch.no_grad():
                # Trigger a standard forward pass. 
                # This automatically handles the Vision Tower and Multimodal Projector flawlessly.
                fp_model(**batch)
        except StopForwardException:
            pass # We successfully caught the inputs
        
        # Remove the wiretap so it doesn't interfere later
        handle.remove()
        
        cached_inputs.append(inps)
        
        # Aggressive VRAM cleanup
        del batch
        torch.cuda.empty_cache()

    print(f"Saving extracted hidden states to {cache_path}...")
    torch.save(cached_inputs, cache_path)

    return cached_inputs

class StopForwardException(Exception):
    pass

class GradSaverHook:
    def __init__(self, 
                 store_grad: bool = True
                 ) -> None:
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, 
                 module: nn.Module, 
                 grad_input: torch.Tensor, 
                 grad_output: torch.Tensor
                 ) -> None:
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException