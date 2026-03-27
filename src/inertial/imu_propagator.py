from __future__ import annotations

import numpy as np

from src.inertial.imu_state import ImuState
from src.io.euroc_loader import ImuSample


def _hat(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def so3_exp(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) + _hat(phi)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    H = _hat(phi)
    return np.eye(3, dtype=np.float64) + a * H + b * (H @ H)


class ImuPropagator:
    def __init__(self, gravity_world: np.ndarray) -> None:
        self.g = gravity_world.astype(np.float64)

    def propagate(self, state: ImuState, imu_seq: list[ImuSample]) -> ImuState:
        if not imu_seq:
            return state

        s = state.copy()
        for i in range(1, len(imu_seq)):
            prev = imu_seq[i - 1]
            curr = imu_seq[i]
            dt = curr.timestamp_s - prev.timestamp_s
            if dt <= 0.0:
                continue

            omega = prev.gyro_rad_s - s.b_g
            acc = prev.accel_m_s2 - s.b_a

            s.R_wb = s.R_wb @ so3_exp(omega * dt)
            a_world = self.g + s.R_wb @ acc
            s.p_wb = s.p_wb + s.v_wb * dt + 0.5 * a_world * dt * dt
            s.v_wb = s.v_wb + a_world * dt

        return s
