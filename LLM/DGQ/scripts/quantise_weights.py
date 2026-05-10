import torch
import os
from tqdm import tqdm
from transformers import Gemma3ForConditionalGeneration
from safetensors.torch import save_file
from quant.quant_model import QuantGemma3Model

def main():
    model_id = "google/gemma-3-4b-it"
    output_dir = "./gemma-3-w4a16-weights"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {model_id} for Weight Quantization...")
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    # Wrap the model. Notice we only pass weight params, no activation params yet.
    q_model = QuantGemma3Model(
        base_model, 
        weight_quant_params={'bits': 4}, 
        act_quant_params={} # Disabled for this phase
    )
    
    # Enable ONLY weight quantization
    q_model.set_quant_state(weight_quant=True, act_quant=False)

    print("Calculating Min-Max scales for 4-bit weights...")
    with torch.no_grad():
        target_modules = [m for m in q_model.modules() if hasattr(m, '_quantize_weight')]

        for m in tqdm(target_modules, desc="Quantizing Projections"):
            m._quantize_weight(m.orig_linear.weight)

    print(f"Saving W4A16 intermediate state to {output_dir}...")
    state_dict = {k: v.cpu().clone() for k, v in q_model.state_dict().items()}
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    q_model.model.config.save_pretrained(output_dir)
    
    print("Weight Quantization Complete.")

if __name__ == "__main__":
    main()