import math
import torch
from torch import Tensor
import sys
import os

_LOG_EPS: float = 1e-8

def compute_log_scale(x: Tensor) -> Tensor:
    """
    Derive the log-quantizer scale from an observed tensor.

      s = max(A[:,1:])

    For cross-attention, the caller is responsible for passing only the non-<start> tokens before calling this

    Parameters
    ----------
    x : Tensor  — positive-valued tensor (post-Softmax attention scores)

    Returns
    -------
    scale : 0-dim Tensor
    """
    return x.max()

def log_quantise(x: Tensor, scale: Tensor, bit_width: int) -> Tensor:
    """
    Map positive floating-point values to integer codes

        x_q = clamp( round( -log2(x / s) ),  0,  2^b - 1 )

    Parameters
    ----------
    x         : Tensor  — positive input values
    scale     : Tensor  — broadcastable against x
    bit_width : int

    Returns
    -------
    x_q : Tensor  — integer codes in [0, 2^b - 1], stored as float
    """
    assert bit_width >= 1
    q_max = 2 ** bit_width - 1
    x_safe = x.clamp(min=_LOG_EPS) # Guard against log2(0) = -inf
    x_q = torch.clamp(torch.round(-torch.log2(x_safe / scale)), min=0.0, max=float(q_max))
    return x_q

def log_dequantise(x_q: Tensor, scale: Tensor) -> Tensor:
    """
    Reconstruct floating-point values from log codes

        x_dq = s * 2^(-x_q)

    Parameters
    ----------
    x_q   : Tensor  — integer codes (stored as float)
    scale : Tensor  — broadcastable against x_q

    Returns
    -------
    x_dq : Tensor  — reconstructed positive floating-point values
    """
    return scale * torch.pow(2.0, -x_q)

def log_fake_quantise(x: Tensor, scale: Tensor, bit_width: int) -> Tensor:
    """
    Simulate log quantisation in floating-point (fake-quant)

    Applies log_quantise then log_dequantise so the tensor stays in float32 but carries the quantisation error
    Used during PTQ calibration

    Parameters
    ----------
    x         : Tensor  — positive input values
    scale     : Tensor  — log-quantiser scale (typically max of x)
    bit_width : int

    Returns
    -------
    x_dq : Tensor  — fake-quantised tensor
    """
    x_q = log_quantise(x, scale, bit_width)
    x_dq = log_dequantise(x_q, scale)
    return x_dq

class logQuantiser(torch.nn.Module):
  """
  Stateful logarithmic quantiser

  Two operating modes
  -------------------
  Dynamic : scale = max(x) recomputed each call
            Used for cross-attention - prompt-specific scale

  Static  : scale fixed once during calibration via set_scale()
            Used for self-attention - more stable distribution

  The caller (attention module) is responsible for slicing out the <start> token before calling this
  """

  def __init__(self, bit_width: int, dynamic: bool = False):
    """
    Parameters
    ----------
    bit_width : int - target bit_width b
    dynamic   : bool - True for dynamic mode, False for static mode
    """
    super().__init__()
    self.bit_width = bit_width
    self.dynamic = dynamic

    # Calibrated scale for static mode stored as a buffer so it travels with the module via state_dict
    self.register_buffer("_scale_buf", torch.zeros(1))
    self._scale_set: bool = False

  def set_scale(self, scale: Tensor) -> None:
        """Store a calibrated scale for static mode."""
        self._scale_buf.data = scale.clone().float().reshape(1)
        self._scale_set      = True

  def set_dynamic(self, dynamic: bool) -> None:
        self.dynamic = dynamic

  @property
  def is_ready(self) -> bool:
        """True if the quantiser can run without error."""
        return self.dynamic or self._scale_set

  def forward(self, x: Tensor) -> Tensor:
    """
    Fake-quantise x using log quantisation

    Dynamic mode: scale = max(x)
    Static mode: scale = self._scale_buf

    Parameters
    ----------
    x : Tensor  — positive input values (caller must have removeed the <start> token for cross-attention)

    Returns
    -------
    x_q : Tensor  — fake log-quantised tensor, same shape as x
    """
    if not self.is_ready:
            raise RuntimeError(
                "LogQuantizer.forward() called before calibration. "
                "Either set dynamic=True or call set_scale() first."
            )
    if self.dynamic:
      scale = compute_log_scale(x)
    else:
      scale = self._scale_buf.to(x.device)
    return log_fake_quantise(x, scale, self.bit_width)

    def extra_repr(self) -> str:
        mode  = "dynamic" if self.dynamic else "static"
        s_val = f"{self._scale_buf.item():.6f}" if self._scale_set else "not set"
        return f"bit_width={self.bit_width}, mode={mode}, scale={s_val}"