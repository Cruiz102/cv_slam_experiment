from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraFrame:
    timestamp_s: float
    image_path: Path


@dataclass
class ImuSample:
    timestamp_s: float
    gyro_rad_s: np.ndarray
    accel_m_s2: np.ndarray


@dataclass
class EurocSequence:
    frames: list[CameraFrame]
    imu: list[ImuSample]


def _ns_to_s(t_ns: int) -> float:
    return float(t_ns) * 1e-9


def load_euroc_cam0_frames(seq_root: Path) -> list[CameraFrame]:
    csv_path = seq_root / "mav0" / "cam0" / "data.csv"
    img_dir = seq_root / "mav0" / "cam0" / "data"
    frames: list[CameraFrame] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            t_ns = int(row[0])
            filename = row[1].strip()
            frames.append(CameraFrame(timestamp_s=_ns_to_s(t_ns), image_path=img_dir / filename))

    frames.sort(key=lambda x: x.timestamp_s)
    return frames


def load_euroc_imu0(seq_root: Path) -> list[ImuSample]:
    csv_path = seq_root / "mav0" / "imu0" / "data.csv"
    samples: list[ImuSample] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            t_ns = int(row[0])
            wx, wy, wz = (float(row[1]), float(row[2]), float(row[3]))
            ax, ay, az = (float(row[4]), float(row[5]), float(row[6]))
            samples.append(
                ImuSample(
                    timestamp_s=_ns_to_s(t_ns),
                    gyro_rad_s=np.array([wx, wy, wz], dtype=np.float64),
                    accel_m_s2=np.array([ax, ay, az], dtype=np.float64),
                )
            )

    samples.sort(key=lambda x: x.timestamp_s)
    return samples


def load_euroc_sequence(seq_root: Path) -> EurocSequence:
    frames = load_euroc_cam0_frames(seq_root)
    imu = load_euroc_imu0(seq_root)
    if not frames:
        raise ValueError("No camera frames found")
    if not imu:
        raise ValueError("No IMU samples found")
    return EurocSequence(frames=frames, imu=imu)


def load_gray_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def imu_between(samples: list[ImuSample], t0: float, t1: float) -> list[ImuSample]:
    # Linear scan is fine for the first implementation.
    return [s for s in samples if t0 < s.timestamp_s <= t1]
