from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline
from diffusers_rewrite.sd import UNet2DConditionModel

import torch
import os, sys
import numpy as np
import argparse

from typing import List, Tuple, Dict

from quant.calibration import cali_model, cali_model_multi
from quant.quant_layer import QMODE, Scaler
from quant.quant_model import QuantModel
from quant.reconstruction_util import RLOSS

from pytorch_lightning import seed_everything
import logging
import datetime
import pandas as pd

from dataset_generation import calibration_data_generation
from utils import str2bool, nested_device, nested_float, prepare_pipe

import torch.multiprocessing as mp
from collections import defaultdict

MODEL_TYPE = os.environ.get("DIFFUSERS_REWRITE", "sd")



def parse_args():
    parser = argparse.ArgumentParser(description='Activation Quantization for Diffusion Models')
    
    # common setting
    parser.add_argument('--outdir', type=str, default='results', help='Output directory')

    # quantization setting
    parser.add_argument('--wq', type=int, default=4, help='Weight quantization bits')
    parser.add_argument('--aq', type=int, default=8, help='Activation quantization bits')
    parser.add_argument('--softmax_a_bit', type=int, default=8, help='Softmax activation quantization bits')
    parser.add_argument('--use_aq', action='store_true', help='Use activation quantization')
    parser.add_argument('--resume_w', type=str, default=None, help='Resume weight quantization')

    # calibration setting
    parser.add_argument('--cali', action='store_true', help='Use calibration')
    parser.add_argument('--cali_prompt_data_n', default=64, type=int, help='Number of calibration prompt data')
    parser.add_argument('--cali_data_path', type=str, default='./data/cali_data', help='Calibration data path')
    parser.add_argument('--cali_data_size', type=int, default=-1, help='Calibration data size (latents per each timestep) (only for debug!)')
    parser.add_argument('--step_size', type=int, default=50)
    
    # wqtizer setting
    parser.add_argument('--tib_recon', type=str2bool, default=False, help='qdiff (no tib)')
    parser.add_argument('--no_recon', type=str2bool, default=False, help='no reconstruction on weight quantization')

    # aqtizer setting 
    # # doesn't affect anything to the weight quantization
    parser.add_argument('--time_aware_aqtizer', type=str2bool, help='time aware quantizer')
    parser.add_argument('--t2i_log_quant', type=str2bool, help='use log quantizer for transformer block')
    parser.add_argument('--t2i_real_time', type=str2bool, help='use real time quantizer for transformer block')
    parser.add_argument('--t2i_start_peak', type=str2bool, help='start peak for transformer block')

    # default settings for quantization : don't change
    parser.add_argument('--rloss', type=str, default='mse', help='Reconstruction loss')
    parser.add_argument('--iters', default=20000, type=int, help='Number of iterations')
    
    # for debugging
    parser.add_argument('--fast', type=str2bool, default=False, help='Fast mode')
    parser.add_argument('--debug', action='store_true', help='Debug mode, same with --fast --cali_data_size 4 --iters 10')
    
    # for reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--coco_path', type=str, default='/scratch/ac5n22/datasets/annotations/captions_train2017.json', help='COCO dataset path')
    
    
    # multi-gpu configs
    parser.add_argument('--multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3367', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')

    opt = parser.parse_args()
    opt.running_stat = True
    opt.asym = True

    if opt.debug:
        print("Debug mode, it's same with --fast --cali_data_size 4 --iters 10")
        opt.fast = True
        opt.cali_data_size = 4
        opt.iters = 10

    assert not(opt.no_recon and opt.multi_gpu), 'no_recon and multi_gpu cannot be co-setted'
    assert not(opt.resume_w and opt.multi_gpu), 'resume_w and multi_gpu cannot be co-setted'

    if opt.resume_w:
        opt.fast = True

    return opt

def setup(seed, world_size, outdir):
    print("Setting up - seed, logging, multi-gpu, outpath")

    # setting up the environment
    seed_everything(seed)

    # multi-gpu setting
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node * world_size

    # outpath setting
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)

    # logging setting
    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    return ngpus_per_node, world_size, outpath, logger

def setup_pipe_to_calibrate(model_type, pipe):
    pipe.unet.float()
    if model_type == "sdxl":
        def forward(
            self, sample, timesteps, encoder_hidden_states, text_embeds, time_ids, **kwargs
        ):
            self.original_forward(sample, timesteps, encoder_hidden_states, 
                                {"text_embeds": text_embeds, "time_ids": time_ids}, **kwargs)
    
        setattr(pipe.unet, "original_forward", pipe.unet.forward)
        setattr(pipe.unet, "forward", forward.__get__(pipe.unet))
    else:
        pass

def main():
    opt = parse_args()

    # load model
    pipe = prepare_pipe(MODEL_TYPE)
    
    # setup
    ngpus_per_node, world_size, outpath, logger = setup(opt.seed, opt.world_size, opt.outdir)
    opt.world_size = world_size
    
    # log setting
    logger.info(f"sys.argv: {sys.argv}")

    # calibration data generation
    cali_data_args = dict(pipe=pipe, 
                        cali_data_path = opt.cali_data_path,
                        coco_path = opt.coco_path,
                        cali_prompt_data_n = opt.cali_prompt_data_n,
                        step_size = opt.step_size,
                        time_aware_aqtizer = opt.time_aware_aqtizer,
                        cali_data_size = opt.cali_data_size)
    w_cali_data, a_cali_data, interval = calibration_data_generation(MODEL_TYPE, **cali_data_args)

    # quantization setting
    wq_params = {"bits": opt.wq,
                 "channel_wise": True,
                 "scaler": Scaler.MINMAX if opt.fast else Scaler.MSE,
                 "leaf_param": opt.no_recon}
    
    aq_params = {"bits": opt.aq,
                 "channel_wise": False,
                 "scaler": Scaler.MSE if opt.cali else Scaler.MINMAX,
                 "leaf_param": opt.use_aq}
    
    softmax_aq_params = {"softmax_a_bit": opt.softmax_a_bit,
                 "t2i_log_quant": opt.t2i_log_quant,
                 "t2i_real_time": opt.t2i_real_time,
                 "t2i_start_peak": opt.t2i_start_peak,
                 "log_max_1": False}
    
    # quantization
    if MODEL_TYPE != "flux2":
        setup_pipe_to_calibrate(MODEL_TYPE, pipe)
    QuantModel_args = dict(model=pipe.unet if MODEL_TYPE != "flux2" else pipe.transformer,
                            wq_params=wq_params,
                            aq_params=aq_params,
                            softmax_aq_params=softmax_aq_params,
                            aq_mode=[QMODE.NORMAL.value, QMODE.QDIFF.value],
                            tib_recon=opt.tib_recon,)
    cali_model_args = dict(use_aq=opt.use_aq,
                    path=os.path.join(outpath, "cali_ckpt.pth"),
                    running_stat=opt.running_stat,
                    interval=interval,
                    tib_recon=opt.tib_recon,
                    w_cali_data=w_cali_data,
                    a_cali_data = a_cali_data,
                    iters=opt.iters,
                    batch_size=8,
                    w=0.01,
                    asym=opt.asym,
                    warmup=0.2,
                    opt_mode=RLOSS.MSE,
                    multi_gpu=False,
                    no_recon=opt.no_recon,
                    resume_w=opt.resume_w)
                           
    if opt.multi_gpu == False: # single gpu
        qnn = QuantModel(**QuantModel_args).to('cuda').eval()
        cali_model(qnn=qnn,**cali_model_args)

    else: # multi-gpu
        raise NotImplementedError("Multi-gpu is not supported yet")
    
        logger.info("Using multi-gpu")
        kwargs = dict(iters=opt.iters,
                            batch_size=8,
                            w=0.01,
                            asym=opt.asym,
                            warmup=0.2,
                            opt_mode=RLOSS.MSE,
                            wq_params=wq_params,
                            aq_params=aq_params,
                            softmax_a_bit=opt.softmax_a_bit,
                            aq_mode=opt.q_mode,
                            multi_gpu=ngpus_per_node > 1,
                            t2i_log_quant=opt.t2i_log_quant,
                            t2i_real_time=opt.t2i_real_time,
                            t2i_start_peak=opt.t2i_start_peak,
                        )
                            #no_grad_ckpt=opt.no_grad_ckpt)
                            
        mp.spawn(cali_model_multi, args=(opt.dist_backend,
                            opt.world_size,
                            opt.dist_url,
                            opt.rank,
                            ngpus_per_node,
                            pipe.unet,
                            opt.use_aq,
                            os.path.join(outpath, "cali_ckpt.pth"),
                            w_cali_data,
                            a_cali_data,
                            interval,
                            opt.running_stat,
                            opt.qdiff,
                            kwargs), nprocs=ngpus_per_node)
        
        
    print("\n================ FULL QUANT AUDIT ================\n")

    device = next(qnn.parameters()).device

    # -------------------------------------------------
    # 1. Robust input construction (Flux2-safe)
    # -------------------------------------------------
    hidden_size = None

    # Try common attributes
    for attr in ["hidden_size", "dim", "model_dim", "query_dim", "inner_dim"]:
        if hasattr(qnn, attr):
            hidden_size = getattr(qnn, attr)
            print(f"[INFO] Found hidden_size via qnn.{attr}: {hidden_size}")
            break

    # Fallback: infer from any Linear-like module
    if hidden_size is None:
        for m in qnn.modules():
            if isinstance(m, torch.nn.Linear):
                hidden_size = m.in_features
                print(f"[INFO] Inferred hidden_size from Linear: {hidden_size}")
                break

    # Final fallback (safe default for diffusion models)
    if hidden_size is None:
        hidden_size = 768
        print("[WARN] Using fallback hidden_size = 768")

    x = torch.randn(1, 77, hidden_size, device=device)

    print("\n[INFO] Input shape:", x.shape)
    print("[INFO] Device:", device)

    # -------------------------------------------------
    # 2. Structural quantisation audit
    # -------------------------------------------------
    linear_total = 0
    linear_quant = 0
    norm_total = 0
    norm_quant = 0
    aqtizer_total = 0

    for m in qnn.modules():
        if isinstance(m, torch.nn.Linear):
            linear_total += 1
            if hasattr(m, "aqtizer"):
                linear_quant += 1

        if isinstance(m, torch.nn.RMSNorm):
            norm_total += 1
            if hasattr(m, "aqtizer"):
                norm_quant += 1

        if hasattr(m, "aqtizer"):
            aqtizer_total += 1

    print("\n===== STRUCTURE REPORT =====")
    print(f"Linear quantised: {linear_quant}/{linear_total}")
    print(f"RMSNorm quantised: {norm_quant}/{norm_total}")
    print(f"Aqtizer attachments: {aqtizer_total}")

    # -------------------------------------------------
    # 3. Module type breakdown (VERY important for Flux2)
    # -------------------------------------------------
    print("\n===== MODULE TYPE DISTRIBUTION =====")

    counts = defaultdict(int)
    for m in qnn.modules():
        counts[type(m).__name__] += 1

    for k, v in sorted(counts.items(), key=lambda x: -x[1])[:25]:
        print(f"{k:40s}: {v}")

    # -------------------------------------------------
    # 4. Attention coverage sanity check
    # -------------------------------------------------
    attn_total = 0
    attn_quant = 0

    for name, m in qnn.named_modules():
        if "attn" in name.lower():
            attn_total += 1

            # consider "quantised" if ANY projection is patched
            has_quant = False
            for attr in ["to_q", "to_k", "to_v", "to_qkv_mlp_proj"]:
                if hasattr(m, attr) and hasattr(getattr(m, attr), "aqtizer"):
                    has_quant = True

            if has_quant:
                attn_quant += 1

    print("\n===== ATTENTION REPORT =====")
    print(f"Attention blocks total: {attn_total}")
    print(f"Attention blocks quantised: {attn_quant}")

    # -------------------------------------------------
    # 5. Forward sanity check
    # -------------------------------------------------
    qnn.eval()

    print("\n===== FORWARD PASS TEST =====")

    try:
        with torch.no_grad():
            x = torch.randn(1, 77, hidden_size, device=device)

            # Flux2 models often require kwargs safety
            out = qnn(
                x,
                encoder_hidden_states=None,
                timestep=None,
                y=None
            )

        print("Forward: OK")
        print("Output shape:", getattr(out, "shape", type(out)))

    except Exception as e:
        print("Forward FAILED:")
        print(e)

    print("\n================ END AUDIT ================\n")
    print("\n================ FLUX2 ATTENTION VARIANT AUDIT ================\n")

    # -------------------------------------------------
    # 1. Identify all attention module types
    # -------------------------------------------------
    attn_counts = defaultdict(int)
    attn_quant = defaultdict(int)

    def is_quantised_attn(m):
        # robust definition: ANY projection or norm has aqtizer
        for attr in ["to_q", "to_k", "to_v", "to_out",
                    "norm_q", "norm_k",
                    "add_q_proj", "add_k_proj", "add_v_proj",
                    "to_add_out"]:
            if hasattr(m, attr):
                sub = getattr(m, attr)

                # ModuleList case (Flux2 uses this sometimes)
                if isinstance(sub, torch.nn.ModuleList):
                    for s in sub:
                        if hasattr(s, "aqtizer"):
                            return True

                if hasattr(sub, "aqtizer"):
                    return True

        return False


    # -------------------------------------------------
    # 2. Scan model
    # -------------------------------------------------
    for name, m in qnn.named_modules():
        cls = type(m).__name__

        # only focus on Flux2 attention-related modules
        if "Attention" in cls or "attention" in name.lower():
            attn_counts[cls] += 1

            if is_quantised_attn(m):
                attn_quant[cls] += 1


    # -------------------------------------------------
    # 3. Report per variant
    # -------------------------------------------------
    print("===== ATTENTION VARIANT COVERAGE =====\n")

    all_keys = set(attn_counts.keys()) | set(attn_quant.keys())

    for k in sorted(all_keys):
        total = attn_counts[k]
        quant = attn_quant[k]
        ratio = (quant / total * 100) if total > 0 else 0.0

        print(f"{k:40s} : {quant:4d}/{total:4d}  ({ratio:5.1f}%)")

    # -------------------------------------------------
    # 4. Global summary
    # -------------------------------------------------
    total_attn = sum(attn_counts.values())
    total_quant = sum(attn_quant.values())

    print("\n===== GLOBAL SUMMARY =====")
    print(f"Total attention modules     : {total_attn}")
    print(f"Quantised attention modules : {total_quant}")
    print(f"Coverage                    : {100*total_quant/total_attn:.2f}%")

    # -------------------------------------------------
    # 5. Find unquantised hotspots (IMPORTANT)
    # -------------------------------------------------
    print("\n===== FIRST 50 UNQUANTISED ATTENTION MODULES =====")

    count = 0
    for name, m in qnn.named_modules():
        cls = type(m).__name__

        if "Attention" in cls or "attention" in name.lower():
            if not is_quantised_attn(m):
                print(f"{name:80s} -> {cls}")
                count += 1

            if count >= 50:
                break

    print("\n================ END AUDIT ================\n")
    
    hits = {"quant": 0, "orig": 0}

    def hook(name):
        def f(m, inp, out):
            hits["quant" if "Quant" in type(m).__name__ else "orig"] += 1
        return f

    for m in qnn.modules():
        if "attn" in type(m).__name__.lower():
            m.register_forward_hook(hook(type(m).__name__))
            break
        
    _ = qnn(x)
    print(hits)
if __name__ == "__main__":
    main()
