import torch
from enum import Enum
from quant.quant_layer import QMODE, QuantLayer, lp_loss
from quant.adaptive_rounding import AdaRoundQuantizer
from typing import Union
import logging
logger = logging.getLogger(__name__)

RLOSS = Enum('RLOSS', ('RELAXATION', 'MSE', 'FISHER_DIAG', 'FISHER_FULL', 'NONE'))
print_freq = 2000

class LinearTempDecay:
    def __init__(self, 
                 t_max: int, 
                 rel_start_decay: float = 0.2, 
                 start_b: int = 10, 
                 end_b: int = 2
                 ) -> None:
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t) -> float:
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
