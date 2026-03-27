from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator


class CameraCalibration(BaseModel):
    width: int
    height: int
    intrinsics: list[float] = Field(description="[fx, fy, cx, cy]")
    distortion_coeffs: list[float] = Field(default_factory=list)

    @field_validator("intrinsics")
    @classmethod
    def _validate_intrinsics(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("intrinsics must be [fx, fy, cx, cy]")
        return v

    @property
    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def dist(self) -> np.ndarray:
        if not self.distortion_coeffs:
            return np.zeros((4, 1), dtype=np.float64)
        return np.asarray(self.distortion_coeffs, dtype=np.float64).reshape(-1, 1)


class ImuCalibration(BaseModel):
    gyroscope_noise_density: float = 0.0
    gyroscope_random_walk: float = 0.0
    accelerometer_noise_density: float = 0.0
    accelerometer_random_walk: float = 0.0
    update_rate: float = 200.0


class RigCalibration(BaseModel):
    cam0: CameraCalibration
    imu0: ImuCalibration = Field(default_factory=ImuCalibration)


@dataclass
class FrameConventions:
    gravity_world: np.ndarray


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_euroc_cam_calibration(cam_yaml_path: Path) -> CameraCalibration:
    data = _read_yaml(cam_yaml_path)
    intr = data.get("intrinsics", [])
    dist = data.get("distortion_coefficients", [])
    resolution = data.get("resolution", [0, 0])
    return CameraCalibration(
        width=int(resolution[0]),
        height=int(resolution[1]),
        intrinsics=[float(x) for x in intr],
        distortion_coeffs=[float(x) for x in dist],
    )


def load_euroc_imu_calibration(imu_yaml_path: Path) -> ImuCalibration:
    data = _read_yaml(imu_yaml_path)
    return ImuCalibration(
        gyroscope_noise_density=float(data.get("gyroscope_noise_density", 0.0)),
        gyroscope_random_walk=float(data.get("gyroscope_random_walk", 0.0)),
        accelerometer_noise_density=float(data.get("accelerometer_noise_density", 0.0)),
        accelerometer_random_walk=float(data.get("accelerometer_random_walk", 0.0)),
        update_rate=float(data.get("update_rate", 200.0)),
    )


def default_frame_conventions() -> FrameConventions:
    # z-up world frame with gravity downward.
    return FrameConventions(gravity_world=np.array([0.0, 0.0, -9.81], dtype=np.float64))
