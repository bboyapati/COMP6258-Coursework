import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_wrapper import DGQModelWrapper


def load_base_model(model_name: str, device: str):
    """
    Loads the base model without any quantisation (baseline).
    """
    print(f"Loading tokeniser from {model_name}...")
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    if tokeniser.pad_token_id is None:
        tokeniser.pad_token = tokeniser.eos_token

    print(f"Loading base model from {model_name} (no quantisation)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    return model, tokeniser


def load_quantised_model(model_name: str, outlier_map_path: str,
                         weight_bits: int, act_bits: int, device: str,
                         quantise_kv_cache: bool = False, kv_bits: int = 4,
                         num_sink_tokens: int = 1):
    """
    Loads the base model and re-applies the DGQ wrapper using a previously
    saved outlier_map JSON. This perfectly recreates the quantised forward
    pass without needing a custom model class registered with HuggingFace.
    """
    print(f"Loading tokeniser from {model_name}...")
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    if tokeniser.pad_token_id is None:
        tokeniser.pad_token = tokeniser.eos_token

    print(f"Loading base model from {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )

    print(f"Loading outlier map from {outlier_map_path}...")
    with open(outlier_map_path, "r") as f:
        outlier_map = json.load(f)

    print("Re-applying DGQ quantisation...")
    model = DGQModelWrapper(
        base_model,
        outlier_map=outlier_map,
        weight_bits=weight_bits,
        act_bits=act_bits,
        quantise_kv_cache=quantise_kv_cache,
        kv_bits=kv_bits,
        num_sink_tokens=num_sink_tokens,
    )
    model.to(device)
    model.eval()
    return model, tokeniser


@torch.no_grad()
def run_inference(model, tokeniser, prompt: str, device: str,
                  max_new_tokens: int = 256, temperature: float = 1.0,
                  do_sample: bool = False):
    """
    Tokenises the prompt and runs greedy or sampled generation.
    Returns a tuple of (decoded_text, vram_stats_dict).
    """

    inputs = tokeniser(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    # Measure VRAM before generation (model weights only)
    if device == "cuda":
        torch.cuda.synchronize()
        vram_before = torch.cuda.memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        pad_token_id=tokeniser.pad_token_id,
    )

    # Measure VRAM after generation (includes KV-cache + activations peak)
    vram_stats = {}
    num_generated = output_ids.shape[-1] - prompt_len
    if device == "cuda":
        torch.cuda.synchronize()
        vram_after = torch.cuda.memory_allocated() / 1e9
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        vram_stats = {
            "model_vram_gb": round(vram_before, 2),
            "peak_vram_gb": round(vram_peak, 2),
            "generation_overhead_gb": round(vram_peak - vram_before, 2),
            "prompt_tokens": prompt_len,
            "generated_tokens": num_generated,
        }
    else:
        vram_stats = {
            "prompt_tokens": prompt_len,
            "generated_tokens": num_generated,
        }

    decoded = tokeniser.decode(output_ids[0], skip_special_tokens=True)
    return decoded, vram_stats


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a DGQ-quantised LLM."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Base HuggingFace model name (e.g. google/gemma-3-4b-it)."
    )
    parser.add_argument(
        "--outlier_map", type=str, default=None,
        help="Path to outlier_map.json. If omitted, runs the base model without quantisation."
    )
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--act_bits", type=int, default=8)
    parser.add_argument("--quantise_kv_cache", action="store_true",
                        help="Enable KV-cache quantisation (attention sink preservation).")
    parser.add_argument("--kv_bits", type=int, default=4,
                        help="Bit-width for quantised KV-cache entries.")
    parser.add_argument("--num_sink_tokens", type=int, default=1,
                        help="Number of initial tokens to keep in full precision in the KV-cache.")
    parser.add_argument(
        "--prompt", type=str,
        default="What is the difference between a CPU and a GPU?",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.outlier_map:
        model, tokeniser = load_quantised_model(
            args.model_name, args.outlier_map,
            args.weight_bits, args.act_bits, args.device,
            quantise_kv_cache=args.quantise_kv_cache,
            kv_bits=args.kv_bits,
            num_sink_tokens=args.num_sink_tokens,
        )
    else:
        print("No outlier map provided — running base model without quantisation.")
        model, tokeniser = load_base_model(args.model_name, args.device)

    print("\n" + "=" * 60)
    print(f"PROMPT:\n{args.prompt}")
    print("=" * 60)

    response, vram_stats = run_inference(
        model, tokeniser, args.prompt, args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )

    print(f"RESPONSE:\n{response}")
    print("=" * 60)

    if vram_stats:
        print(f"\n--- Generation Stats ---")
        print(f"Prompt tokens:           {vram_stats['prompt_tokens']}")
        print(f"Generated tokens:        {vram_stats['generated_tokens']}")
        if 'model_vram_gb' in vram_stats:
            print(f"Model loaded:            {vram_stats['model_vram_gb']:.2f} GB")
            print(f"Peak during generation:  {vram_stats['peak_vram_gb']:.2f} GB")
            print(f"Generation overhead:     {vram_stats['generation_overhead_gb']:.2f} GB")


if __name__ == "__main__":
    main()
