import torch
import gc
import os
from transformers import Gemma3ForConditionalGeneration
from safetensors.torch import load_file, save_file
from quant.quant_model import QuantGemma3Model
from quant.data_utill import build_gemma3_calibration_loader
from quant.reconstruction import layer_wise_reconstruction_single_gpu

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    model_id = "google/gemma-3-4b-it"
    weight_dir = "./gemma-3-w4a16-weights" # Output from script 1
    final_output_dir = "./gemma-3-dgq-w4a8"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the FP target model
    fp_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True 
    )
    
    # 2. Load the base architecture for our quant model
    quant_base_model = Gemma3ForConditionalGeneration.from_pretrained(
        weight_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    # 3. Apply DGQ Wrappers with FULL parameters
    q_model = QuantGemma3Model(
        quant_base_model, 
        weight_quant_params = {'bits': 4}, 
        act_quant_params = {
            'bits': 8, 
            'attn_bits': 8, 
            'log_base': 2.0, 
            'num_groups': 4 
        }
    )
    
    # Load the 4-bit weight scales we calculated in Script 1
    q_model.load_state_dict(load_file(os.path.join(weight_dir, "model.safetensors")), strict=False)
    
    # Enable BOTH weight and activation quantization for tuning
    q_model.set_quant_state(weight_quant=True, act_quant=True)

    # 4. Calibration Data (Restricted for 16GB VRAM)
    dataloader = build_gemma3_calibration_loader(
        model_id=model_id, batch_size=64, num_calibration_samples=128, max_seq_length=1024
    )

    gc.collect()
    torch.cuda.empty_cache()

    # 5. Tune the Activation Scales
    print("Initiating Block-Wise Activation Tuning...")
    layer_wise_reconstruction_single_gpu(
        quant_model=q_model,
        fp_model=fp_model,
        calibration_dataloader=dataloader,
        reconstruction_epochs=500, 
        lr=1e-3,
        device=device,
        init_hidden_states_cache_path="gemma3_initial_states_64_128.pt"
    )

    # 6. Save final W4A8 model
    # Free up RAM
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    os.makedirs(final_output_dir, exist_ok=True)
    
    print("Scrubbing state_dict for safetensors compatibility...")
    clean_state_dict = {}
    seen_pointers = set() # NEW: Track memory addresses to catch tied weights
    
    for k, v in q_model.state_dict().items():
        if isinstance(v, torch.Tensor):
            # Check if we have already saved this exact block of memory
            ptr = v.data_ptr()
            if ptr in seen_pointers:
                print(f"Skipping tied weight to prevent safetensors crash: {k}")
                continue
            
            # Register the pointer and clean the tensor
            seen_pointers.add(ptr)
            clean_state_dict[k] = v.detach().cpu().contiguous()
            
    # Save the sanitized and deduplicated weights
    save_file(clean_state_dict, os.path.join(final_output_dir, "model.safetensors"))
    
    # Save the Hugging Face config so the model can be loaded later
    q_model.model.config.save_pretrained(final_output_dir)

    print("Activation Tuning Complete. W4A8 Model Saved.")

if __name__ == "__main__":
    main()