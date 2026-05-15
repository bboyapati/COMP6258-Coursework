# COMP6258 Coursework — Scaling Attention-Based Models through Distribution-Aware Group Quantisation (DGQ)
## COMP6258 Reproducibility Challenge — University of Southampton
**Authors:** Anuoluwa Adeleye · Anthony Cheung · Bharath C. Boyapati · Swayem Kandangwa  
**Paper:** [DGQ: Distribution-Aware Group Quantization for 
Text-to-Image Diffusion Models](https://openreview.net/forum?id=ZyNEr7Xw5L) 
(Ryu et al., ICLR 2025)
---
## Overview

This repository contains the code, results, and report for our reproducibility study of DGQ [Ryu et al., 2025]. We reproduce the paper's core quantitative claims on Stable Diffusion v1.4 and extend the distributional analysis to attention-based YOLO architectures and vision-language models.

# CNN_Model_Quantizer 

A fully contained ipynb environment that downloads a YOLOv8 model, calibrates our activation groupings and quantizes model weights and activations. It then wraps the model allowing for ultralytics and pytorch model manipulation. 

To control the number of groups, you need to create a new json file by running the cell under **"Create and Save activations outliers and groupings"** and updating **"n_groups"**.

The WrapLayers function should be updated to the intended weight and activation quantization bits.
