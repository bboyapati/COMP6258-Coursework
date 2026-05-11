import torch
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
from safetensors.torch import load_file
from quant.quant_model import QuantGemma3Model
from PIL import Image
import requests

def main():
    final_model_dir = "./gemma-3-dgq-w4a8" # Output from script 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Processor...")
    processor = Gemma3Processor.from_pretrained(model_id, use_fast=True)

    print("Loading Base Architecture...")
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        final_model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    print("Applying DGQ Quantization Grids...")
    # Wrap it with the same W4A8 parameters so the forward pass knows to apply the STE rounding
    q_model = QuantGemma3Model(
        base_model, 
        weight_quant_params={'bits': 4}, 
        act_quant_params={'act_bits': 8, 'attn_bits': 8, 'log_base': 2.0, 'num_groups': 4}
    )
    
    # Load the perfectly tuned scales
    q_model.load_state_dict(load_file(f"{final_model_dir}/model.safetensors"), strict=False)
    q_model.set_quant_state(weight_quant=True, act_quant=True)
    q_model.to(device)
    q_model.eval()

    # --- Run a Multimodal Test ---
    print("\n--- Running Multimodal Inference Test ---")
    
    # Load a test image (using a standard URL for testing, replace with local if needed)
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    prompt = "Describe the object in this image and explain how its engine works."
    
    # Gemma 3 requires alternating image and text format
    inputs = processor(
        text=[prompt], 
        images=[image], 
        return_tensors="pt"
    ).to(device)

    # Generate
    with torch.no_grad():
        output_ids = q_model.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )

    # Decode and print
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"\nUser: {prompt}")
    print(f"\nGemma 3 (W4A8): {generated_text}")

if __name__ == "__main__":
    main()