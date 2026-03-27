from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ImuState:
    R_wb: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    p_wb: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    v_wb: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def copy(self) -> "ImuState":
        return ImuState(
            R_wb=self.R_wb.copy(),
            p_wb=self.p_wb.copy(),
            v_wb=self.v_wb.copy(),
            b_g=self.b_g.copy(),
            b_a=self.b_a.copy(),
        )
