# COMP6258 Coursework — Distribution-Aware Group Quantisation of Diffusion Models

This repository implements a post-training quantization (PTQ) pipeline for text-to-image diffusion models, inspired by **DGQ (Distribution-aware Group Quantization)**. The goal is to reduce memory/compute cost while preserving both **image quality** and **text-image alignment**, especially in low-bit (<8-bit) activation settings.

> Core idea: (1) preserve **activation outliers** using group-wise quantization, and (2) quantize **attention scores** with a distribution-aware strategy (log quantization + special handling of the `<start>` token + prompt-dependent scaling).

## Features
- [ ] Activation quantization with outlier-preserving group quantization (pixel-wise / channel-wise)
- [ ] Attention-aware quantization for (cross-)attention scores
- [ ] Stable Diffusion inference with quantized activations/weights (TODO specify modules)
- [ ] Evaluation: (FID / IS / CLIP) or lightweight proxy metrics + qualitative grids

## Method (high level)
1. **Outlier-preserving group quantization (activations)**  
   - Detect whether outliers concentrate along **channels** or **pixels** and group accordingly.  
   - Quantize each group with its own scale/zero-point.
2. **Attention-aware quantization (attention scores)**  
   - Apply **logarithmic quantization** to preserve small values on a log scale.  
   - Keep the `<start>` token attention path in full precision.  
   - Use **dynamic scaling per prompt** using max attention value (excluding `<start>`).

## Results
### Qualitative
| Full precision | Baseline PTQ | Ours (DGQ-style) |
|---|---|---|
| (image) | (image) | (image) |

### Quantitative (TODO)
- Model: `...`
- Steps: `...`
- Calibration prompts: `...`
- Bit-width: W?A? (groups=?)

| Setting | FID ↓ | CLIP ↑ | Notes |
|---|---:|---:|---|
| FP32 | TODO | TODO | reference |
| W8A8 | TODO | TODO |  |
| W8A6 | TODO | TODO |  |

## Setup
### Environment
- Python: `3.x`
- PyTorch: `x.x`
- CUDA: `x.x` (optional)

```bash
pip install -r requirements.txt


