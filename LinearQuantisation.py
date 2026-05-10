import math
import torch
from torch import Tensor

def compute_scale_zeropoint(x_min: Tensor, x_max: Tensor, bit_width: int) -> tuple[Tensor, Tensor]:
    """
    Compute quantisation scale and zero-point — Equation 6.

        s = (x_max - x_min) / 2^b
        z = x_min

    Parameters
    ----------
    x_min, x_max : Tensor  — observed min/max (scalar or batched)
    bit_width    : int     — target bit-width b

    Returns
    -------
    scale      : Tensor
    zero_point : Tensor  (float, not rounded to int)
    """
    assert bit_width >= 1, "bit_width must be at least 1"

    n_levels = 2 ** bit_width

    # Clamp to avoid division by zero for constant tensors
    range_ = (x_max - x_min).clamp(min=1e-8)

    scale      = range_ / n_levels   # s — Eq. 6
    zero_point = x_min               # z — Eq. 6

    return scale, zero_point

def linear_quantise(x: Tensor, scale: Tensor, zero_point: Tensor, bit_width: int) -> Tensor:
    """
    Quantise x to integer codes in [0, 2^b - 1] — Equation 1 (left).

        x_q = clamp( round( (x - z) / s ),  0,  2^b - 1 )

    Parameters
    ----------
    x          : Tensor  — floating-point input
    scale      : Tensor  — broadcastable against x
    zero_point : Tensor  — broadcastable against x  (= x_min)
    bit_width  : int

    Returns
    -------
    x_q : Tensor  — integer codes stored as float32
    """
    q_min = 0
    q_max = 2 ** bit_width - 1

    # FIX: shift by zero_point BEFORE dividing by scale
    # Original had: round(x / scale) + zero_point  — wrong convention
    x_q = torch.clamp(
        torch.round((x - zero_point) / scale),
        min=q_min,
        max=q_max,
    )
    return x_q

def linear_dequantise(x_q: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
    """
    Dequantise integer codes back to floating-point — Equation 1 (right).

        x_dq = s * x_q + z

    Parameters
    ----------
    x_q        : Tensor  — integer codes (stored as float32)
    scale      : Tensor  — broadcastable against x_q
    zero_point : Tensor  — broadcastable against x_q

    Returns
    -------
    x_dq : Tensor  — reconstructed floating-point values
    """
    return scale * x_q + zero_point

#
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
s, z = compute_scale_zeropoint(x.min(), x.max(), bit_width=8)

x_q  = linear_quantise(x, s, z, bit_width=8)
x_dq = linear_dequantise(x_q, s, z)

print(f"scale      = {s.item():.5f}")
print(f"zero_point = {z.item():.5f}")
print(f"x          = {x.tolist()}")
print(f"x_q        = {x_q.tolist()}")
print(f"x_dq       = {[round(v, 4) for v in x_dq.tolist()]}")
print(f"max error  = {(x - x_dq).abs().max().item():.6f}  (should be < scale = {s.item():.5f})")

def fake_quantise(x: Tensor, scale: Tensor, zero_point: Tensor, bit_width: int) -> Tensor:
    """
    Simulate quantisation in floating-point (fake-quant).

    Applies linear_quantise then linear_dequantise so the tensor stays in
    float32 but carries the quantisation error. This is inserted into the
    forward pass during PTQ calibration to measure reconstruction quality.

    Note: use_ste is intentionally omitted here — pure PTQ does not need
    gradient flow through the rounding step. STE will be added in the
    BRECQ weight-optimisation module.

    Parameters
    ----------
    x          : Tensor  — input activations or weights
    scale      : Tensor  — broadcastable scale
    zero_point : Tensor  — broadcastable zero-point
    bit_width  : int

    Returns
    -------
    x_dq : Tensor  — fake-quantised tensor, same shape and dtype as x
    """
    x_q  = linear_quantise(x, scale, zero_point, bit_width)
    x_dq = linear_dequantise(x_q, scale, zero_point)
    return x_dq

class LinearQuantiser(torch.nn.Module):
    """
    Stateful linear quantiser for PTQ.

    Calibrate once with set_params(), then call forward() to fake-quantise.
    Scale and zero_point are registered as buffers (not parameters) because
    pure PTQ does not update them via gradients.

    Attributes
    ----------
    bit_width  : int
    scale      : Tensor (buffer) — fixed after calibration
    zero_point : Tensor (buffer) — fixed after calibration
    """

    def __init__(self, bit_width: int):
        super().__init__()
        self.bit_width = bit_width
        self.register_buffer("scale",      torch.zeros(1))
        self.register_buffer("zero_point", torch.zeros(1))

        self._calibrated: bool = False

    def set_params(self, scale: Tensor, zero_point: Tensor) -> None:
        """
        Store calibrated scale and zero-point in the registered buffers.
        Uses .data.copy_() so the buffer stays a buffer (not replaced).
        """
        # Copy into the existing buffer instead of reassigning
        self.scale.data      = scale.clone().float().reshape(self.scale.shape)
        self.zero_point.data = zero_point.clone().float().reshape(self.zero_point.shape)
        self._calibrated     = True

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def forward(self, x: Tensor) -> Tensor:
        """
        Fake-quantise x using the calibrated scale and zero-point.

        FIX: removed use_ste parameter — fake_quantise no longer accepts it.
        STE support will be added in the BRECQ module.
        """
        if not self._calibrated:
            raise RuntimeError(
                "LinearQuantizer.forward() called before calibration. "
                "Call set_params(scale, zero_point) first."
            )

        return fake_quantise(x, self.scale, self.zero_point, self.bit_width)

    def extra_repr(self) -> str:
        s = self.scale.item()   if self.scale.numel()      == 1 else "(...)"
        z = self.zero_point.item() if self.zero_point.numel() == 1 else "(...)"
        return f"bit_width={self.bit_width}, calibrated={self._calibrated}, scale={s:.5f}, zero_point={z:.5f}"