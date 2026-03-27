from __future__ import annotations

import numpy as np

from src.inertial.imu_state import ImuState


class LooseVioFusion:
    def __init__(self, alpha_rotation: float = 0.15, alpha_position: float = 0.05) -> None:
        self.alpha_rotation = float(np.clip(alpha_rotation, 0.0, 1.0))
        self.alpha_position = float(np.clip(alpha_position, 0.0, 1.0))

    def update_with_visual_delta(self, state: ImuState, dR_vis: np.ndarray, dt_vis_dir: np.ndarray) -> ImuState:
        out = state.copy()

        # Light correction that nudges IMU-predicted orientation toward visual increment.
        dR_blend = (1.0 - self.alpha_rotation) * np.eye(3) + self.alpha_rotation * dR_vis
        u, _, vt = np.linalg.svd(dR_blend)
        dR_proj = u @ vt
        out.R_wb = out.R_wb @ dR_proj

        # Monocular translation has unknown scale; use direction only as weak correction.
        if np.linalg.norm(dt_vis_dir) > 1e-12 and np.linalg.norm(out.v_wb) > 1e-6:
            v_dir = out.v_wb / np.linalg.norm(out.v_wb)
            t_dir = dt_vis_dir / np.linalg.norm(dt_vis_dir)
            blend_dir = (1.0 - self.alpha_position) * v_dir + self.alpha_position * t_dir
            out.v_wb = np.linalg.norm(out.v_wb) * (blend_dir / np.linalg.norm(blend_dir))

        return out
