"""
Log-space 4D gain multipliers with per-step clipped delta.
Controls: [alpha_p, alpha_d, beta_x, beta_v]
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class GainBounds:
    lo: np.ndarray  # shape (4,)
    hi: np.ndarray  # shape (4,)

class GainParam4:
    """
    y = log(g). Keep y within [log(lo), log(hi)] and clip each step by a fraction
    of the box size to avoid instability. Values() returns exp(y) clipped to [lo,hi].
    """
    def __init__(self, lo, hi):
        self.lo = np.array(lo, dtype=float)
        self.hi = np.array(hi, dtype=float)
        assert self.lo.shape == (4,) and self.hi.shape == (4,), "expect 4D bounds"
        assert np.all(self.hi > self.lo), "hi must be > lo"

        self.y_lo = np.log(self.lo)
        self.y_hi = np.log(self.hi)
        self.y = 0.5 * (self.y_lo + self.y_hi)  # start in the middle

    def reset_mid(self):
        self.y = 0.5 * (self.y_lo + self.y_hi)

    def step(self, dy, frac_step: float = 0.05):
        """Apply delta in log-space with per-dim clipping by box fraction."""
        dy = np.asarray(dy, float)
        max_step = frac_step * (self.y_hi - self.y_lo)
        dy = np.clip(dy, -max_step, max_step)
        self.y = np.clip(self.y + dy, self.y_lo, self.y_hi)

    def values(self):
        """Return gains g in original space, clipped to [lo,hi]."""
        return np.clip(np.exp(self.y), self.lo, self.hi)

    def normalized(self):
        """Return normalized position in the log-box ∈ [0,1]^4."""
        return (self.y - self.y_lo) / (self.y_hi - self.y_lo + 1e-12)