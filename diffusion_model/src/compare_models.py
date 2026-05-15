from utils import prepare_pipe
from quant.load_qmodel_util import get_qmodel
from quant.quant_layer import Scaler

import torch
import matplotlib.pyplot as plt
from torch import nn


# ---------------------------------------------------
# Extract FP weights
# ---------------------------------------------------

def extract_fp_weights(model):

    weights = []
    for name, module in model.named_modules():
        print("FP NAME ORIGINAL ", name)
        if not hasattr(module, "weight") or module.weight == None:
            continue

        weights.append({
            "name": name,
            "weight": module.weight.detach().float().cpu()
        })

        print("[KEEP]", name)

    return weights


# ---------------------------------------------------
# Extract Quant weights
# ---------------------------------------------------


def extract_quant_weights(model):

    weights = []

    for name, module in model.named_modules():

        # only QuantLayer-style modules
        if not hasattr(module, "wqtizer"):
            continue

        # QuantLayer stores real weight here
        if not hasattr(module, "w"):
            print(f"[SKIP] {name} -> no .w")
            continue

        w_fp = module.w.detach()

        # apply fake quant manually
        w_q = module.wqtizer(module.w).detach()

        # normalise name
        name = normalise_quant_name(name)
        
        weights.append({
            "name": name,
            "fp_weight": w_fp.float().cpu(),
            "quant_weight": w_q.float().cpu(),
            "use_wq": getattr(module, "use_wq", None),
            "bits": int(torch.log2(torch.tensor(module.wqtizer.level)).item())
        })

        print(
            f"[OK] {name:<60} "
            f"shape={tuple(w_fp.shape)} "
            f"bits={int(torch.log2(torch.tensor(module.wqtizer.level)).item())} "
            f"use_wq={module.use_wq}"
        )

    return weights


# ---------------------------------------------------
# Compare
# ---------------------------------------------------

def compare_weight_lists(fp_weights, q_weights):

    print(f"FP layers : {len(fp_weights)}")
    print(f"Q layers  : {len(q_weights)}")

    fp_dict = {w["name"]: w for w in fp_weights}
    q_dict  = {w["name"]: w for w in q_weights}

    fp_keys = set(fp_dict.keys())
    q_keys  = set(q_dict.keys())
    
    all_keys = sorted(fp_keys & q_keys)
    print("FP KEYS ", fp_keys)
    print("Q KEYS ", q_keys)

    print(f"Matched layers: {len(all_keys)}")
    print(f"FP-only layers: {len(fp_keys - q_keys)}")
    print(f"Q-only layers : {len(q_keys - fp_keys)}")

    for name in all_keys:

        w_fp = fp_dict[name]["weight"].flatten()
        w_q = q_dict[name]["quant_weight"].flatten()

        # unique values
        uniq_fp = len(torch.unique(w_fp))
        uniq_q  = len(torch.unique(w_q))

        mse = torch.mean((w_fp - w_q) ** 2).item()

        print("=" * 80)
        print(f"FP: {name}")
        print(f"Q : {name}")

        print(f"FP unique : {uniq_fp}")
        print(f"Q unique  : {uniq_q}")
        print(f"MSE       : {mse:.8f}")

        # histogram
        plt.figure(figsize=(8,4))

        plt.hist(
            w_fp.numpy(),
            bins=200,
            density=True,
            alpha=0.5,
            label="original"
        )

        plt.hist(
            w_q.numpy(),
            bins=200,
            density=True,
            alpha=0.5,
            label="quantised"
        )

        plt.legend()
        plt.title(name)
        plt.savefig(f"figures/comparison_{name}.png")
def normalise_quant_name(name):

    # remove QuantModel wrapper path differences
    name = name.replace("model.", "")
    name = name.replace(".block", "")

    return name

# def filter_fp_weights(fp_weights, q_weights):

#     q_names = set(
#         normalise_quant_name(q["name"])
#         for q in q_weights
#     )
#     print("Q NAMES ", q_names)

#     filtered = []

#     for fp in fp_weights:

#         fp_name = fp["name"]
#         print("FP NAME ", fp_name)
#         if fp_name in q_names:

#             filtered.append(fp)

#             print(f"[MATCH] {fp_name}")

#     return filtered

def main():
    pipe = prepare_pipe("flux2")
    # quantize model
    wq_params = {"bits": 4,
                 "channel_wise": True,
                 "scaler": Scaler.MINMAX
                 }
    
    aq_params = {"bits": 6,
                 "channel_wise": False,
                 "scaler": Scaler.MINMAX,
                 "leaf_param": True}
    
    softmax_aq_params = {"softmax_a_bit": 6,
                 "t2i_log_quant": False,
                 "t2i_real_time": False,
                 "t2i_start_peak": False,
                 "log_max_1": False}
    
    cali_ckpt = "results/merged/w4a6g16/cali_ckpt_activation_w4a6g16.pth_merged"
    fp_weights_all = extract_fp_weights(pipe.transformer)
    
    qmodel = get_qmodel("flux2", pipe, cali_ckpt, wq_params, True, aq_params, softmax_aq_params, 
                     False, num_inference_steps=4, time_aware_aqtizer=False)
    
    q_weights  = extract_quant_weights(qmodel)


    compare_weight_lists(fp_weights_all, q_weights)
    
if __name__ == "__main__":
    main()