"""Learning-rate style schedule for GRL strength (Ganin & Lempitsky)."""

from __future__ import annotations

import numpy as np


def ganin_grl_lambda(epoch: int, max_epochs: int) -> float:
    """Sigmoid ramp from ~0 (epoch 0) to ~1 (last epoch)."""
    if max_epochs <= 1:
        return 1.0
    p = float(epoch) / float(max_epochs - 1)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
