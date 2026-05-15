import torch
from collections import defaultdict
from pathlib import Path
from utils import prepare_pipe

# ============================================================
# DGQ / W4A6 AUDIT SCRIPT
# ============================================================
#
# Usage:
#
#   python audit_quant.py /path/to/checkpoint.pt
#
# Optional:
#   - integrate your model loader in `load_model()`
#   - run a forward pass to verify activation quantisation
#
# ============================================================


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

CHECK_FORWARD = True   # set True if you can run inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------

def load_checkpoint(path):
    print("\n================ LOADING CHECKPOINT ================\n")

    ckpt = torch.load(path, map_location="cpu")

    print("Checkpoint type:", type(ckpt))

    if isinstance(ckpt, dict):
        print("Top-level keys:")
        for k in ckpt.keys():
            print("  ", k)

    return ckpt


# ------------------------------------------------------------
# EXTRACT STATE_DICT
# ------------------------------------------------------------

def extract_state_dict(ckpt):

    if isinstance(ckpt, dict):

        possible_keys = [
            "state_dict",
            "model",
            "module",
            "model_state_dict",
            "weight",
        ]

        for k in possible_keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                print(f"\nUsing state_dict from key: {k}")
                return ckpt[k]

        tensor_count = sum(torch.is_tensor(v) for v in ckpt.values())

        if tensor_count > 0:
            print("\nUsing checkpoint directly as state_dict")
            return ckpt

    raise RuntimeError("Could not locate state_dict")


# ------------------------------------------------------------
# CHECK STORAGE TYPES
# ------------------------------------------------------------

def inspect_tensor_dtypes(state_dict):

    print("\n================ TENSOR DTYPE AUDIT ================\n")

    dtype_stats = defaultdict(int)

    int_like = []
    float_like = []

    total_params = 0
    total_bytes = 0

    for k, v in state_dict.items():

        if not torch.is_tensor(v):
            continue

        dtype_stats[str(v.dtype)] += 1

        numel = v.numel()
        total_params += numel
        total_bytes += numel * v.element_size()

        if v.dtype in [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
        ]:
            int_like.append(k)

        if v.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ]:
            float_like.append(k)

    print("Dtype counts:")
    for k, v in dtype_stats.items():
        print(f"{k:20s}: {v}")

    print("\nApprox checkpoint tensor memory:")
    print(f"{total_bytes / 1024**3:.3f} GB")

    print("\nInteger-like tensors:")
    for k in int_like[:50]:
        print(" ", k)

    if len(int_like) > 50:
        print(f" ... ({len(int_like)-50} more)")

    print("\nFloat-like tensors:")
    for k in float_like[:50]:
        print(" ", k)

    if len(float_like) > 50:
        print(f" ... ({len(float_like)-50} more)")

    return {
        "dtype_stats": dtype_stats,
        "total_params": total_params,
        "total_bytes": total_bytes,
    }


# ------------------------------------------------------------
# DETECT QUANTISATION METADATA
# ------------------------------------------------------------

def detect_quant_metadata(state_dict):

    print("\n================ QUANT METADATA AUDIT ================\n")

    quant_keywords = [
        "scale",
        "zero",
        "zp",
        "qweight",
        "weight_q",
        "packed",
        "quant",
        "amax",
        "act_scale",
    ]

    hits = []

    for k in state_dict.keys():

        lower = k.lower()

        if any(q in lower for q in quant_keywords):
            hits.append(k)

    if len(hits) == 0:
        print("No obvious quant metadata found.")
    else:
        print(f"Found {len(hits)} quant-related tensors:\n")

        for k in hits[:200]:
            print(" ", k)

        if len(hits) > 200:
            print(f"\n... ({len(hits)-200} more)")

    return hits


# ------------------------------------------------------------
# ESTIMATE EFFECTIVE WEIGHT BITWIDTH
# ------------------------------------------------------------

def estimate_effective_bits(state_dict):

    print("\n================ EFFECTIVE BITWIDTH ESTIMATE ================\n")

    total_bits = 0
    total_elems = 0

    for k, v in state_dict.items():

        if not torch.is_tensor(v):
            continue

        bits = v.element_size() * 8

        total_bits += bits * v.numel()
        total_elems += v.numel()

    avg_bits = total_bits / total_elems

    print(f"Average stored bits per tensor element: {avg_bits:.2f}")

    if avg_bits <= 6:
        print("Likely REAL low-bit storage.")
    elif avg_bits <= 10:
        print("Possibly packed quant storage + metadata.")
    else:
        print("Likely fp16/bf16 storage with fake quantisation.")

    return avg_bits


# ------------------------------------------------------------
# OPTIONAL MODEL LOADER
# ------------------------------------------------------------

def load_model(checkpoint_path):

    # --------------------------------------------------------
    # IMPORTS
    # --------------------------------------------------------

    import os

    from utils import prepare_pipe
    from quant.quant_layer import Scaler
    from quant.load_qmodel_util import get_qmodel

    MODEL_TYPE = os.environ.get("DIFFUSERS_REWRITE", "flux2")

    # --------------------------------------------------------
    # LOAD PIPE
    # --------------------------------------------------------

    pipe = prepare_pipe(MODEL_TYPE)

    # --------------------------------------------------------
    # QUANT CONFIG
    # --------------------------------------------------------

    wq_params = {
        "bits": 4,
        "channel_wise": True,
        "scaler": Scaler.MINMAX,
    }

    aq_params = {
        "bits": 6,
        "channel_wise": False,
        "scaler": Scaler.MINMAX,
        "leaf_param": True,
    }

    softmax_aq_params = {
        "softmax_a_bit": 6,
        "t2i_log_quant": False,
        "t2i_real_time": False,
        "t2i_start_peak": False,
        "log_max_1": False,
    }

    # --------------------------------------------------------
    # BUILD QUANTISED MODEL
    # --------------------------------------------------------

    qnn = get_qmodel(
        MODEL_TYPE,
        pipe,
        checkpoint_path,
        wq_params,
        True,                       # use_aq
        aq_params,
        softmax_aq_params,
        False,                      # use_group
        num_inference_steps=4,
        time_aware_aqtizer=False,
    )

    # --------------------------------------------------------
    # ATTACH TO PIPE
    # --------------------------------------------------------

    if hasattr(pipe, "unet"):
        pipe.unet = qnn
        model = pipe.unet
    else:
        pipe.transformer = qnn
        model = pipe.transformer

    model.eval()

    return model

# ------------------------------------------------------------
# MODULE AUDIT
# ------------------------------------------------------------

def audit_modules(model):

    print("\n================ MODULE AUDIT ================\n")

    quant_modules = []
    act_quantizers = []
    weight_quantizers = []

    for name, module in model.named_modules():

        cls = type(module).__name__

        if "Quant" in cls or "quant" in cls:
            quant_modules.append((name, cls))

        if hasattr(module, "weight_quantizer"):
            weight_quantizers.append((name, module.weight_quantizer))

        if hasattr(module, "act_quantizer"):
            act_quantizers.append((name, module.act_quantizer))

    print(f"Quant-like modules found: {len(quant_modules)}\n")

    for name, cls in quant_modules[:200]:
        print(f"{name:60s} {cls}")

    print("\nWeight quantizers:\n")

    for name, q in weight_quantizers[:100]:

        bits = getattr(q, "n_bits", "UNKNOWN")

        print(f"{name:60s} bits={bits}")

    print("\nActivation quantizers:\n")

    for name, q in act_quantizers[:100]:

        bits = getattr(q, "n_bits", "UNKNOWN")

        print(f"{name:60s} bits={bits}")


# ------------------------------------------------------------
# FORWARD HOOK VERIFICATION
# ------------------------------------------------------------

def register_hooks(model):

    print("\n================ REGISTERING FORWARD HOOKS ================\n")

    hooks = []

    def hook_fn(name):

        def fn(module, inp, out):

            print(f"\nHOOK: {name}")

            if hasattr(module, "weight_quantizer"):
                q = module.weight_quantizer
                print(
                    "  Weight bits:",
                    getattr(q, "n_bits", "UNKNOWN")
                )

            if hasattr(module, "act_quantizer"):
                q = module.act_quantizer
                print(
                    "  Act bits:",
                    getattr(q, "n_bits", "UNKNOWN")
                )

            if len(inp) > 0 and torch.is_tensor(inp[0]):
                print("  Input dtype:", inp[0].dtype)

            if torch.is_tensor(out):
                print("  Output dtype:", out.dtype)

        return fn

    for name, module in model.named_modules():

        if (
            hasattr(module, "weight_quantizer")
            or hasattr(module, "act_quantizer")
        ):

            h = module.register_forward_hook(hook_fn(name))
            hooks.append(h)

    print(f"Registered {len(hooks)} hooks")

    return hooks


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python audit_quant.py checkpoint.pt")
        return

    ckpt_path = sys.argv[1]

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(ckpt_path)

    # --------------------------------------------------------
    # CHECKPOINT AUDIT
    # --------------------------------------------------------

    ckpt = load_checkpoint(ckpt_path)

    state_dict = extract_state_dict(ckpt)

    inspect_tensor_dtypes(state_dict)

    detect_quant_metadata(state_dict)

    estimate_effective_bits(state_dict)

    # --------------------------------------------------------
    # OPTIONAL FULL MODEL AUDIT
    # --------------------------------------------------------

    try:

        model = load_model(ckpt_path)

        model.to(DEVICE)
        model.eval()

        audit_modules(model)

        if CHECK_FORWARD:

            hooks = register_hooks(model)

            print("\nRUNNING TEST FORWARD...\n")

            dummy_input = torch.randn(
                1,
                4,
                64,
                64,
                device=DEVICE
            )
                
            print("\nRUNNING PIPELINE FORWARD TEST...\n")

            from pytorch_lightning import seed_everything

            seed_everything(42)

            prompt = "a photo of a cat"

            if hasattr(model, "disable_out_quantization"):
                model.disable_out_quantization()

            with torch.no_grad():
                pipe = prepare_pipe("flux2")
                pipe.transformer = model

                pipe(
                    prompt=[prompt],
                    num_inference_steps=1,
                    guidance_scale=1.0,
                )

    except NotImplementedError:
        print(
            "\nSkipping model-level audit "
            "(load_model not implemented)."
        )


if __name__ == "__main__":
    main()