from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_trajectory_csv(out_path: Path, timestamps: list[float], positions: list[np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("timestamp_s,x,y,z\n")
        for t, p in zip(timestamps, positions):
            f.write(f"{t:.9f},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")


def read_trajectory_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts: list[float] = []
    ps: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row["timestamp_s"]))
            ps.append(np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64))
    return np.asarray(ts, dtype=np.float64), np.asarray(ps, dtype=np.float64)


def read_euroc_groundtruth(seq_root: Path) -> tuple[np.ndarray, np.ndarray]:
    gt_path = seq_root / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    ts: list[float] = []
    ps: list[np.ndarray] = []

    with gt_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            t_s = float(int(row[0]) * 1e-9)
            px, py, pz = float(row[1]), float(row[2]), float(row[3])
            ts.append(t_s)
            ps.append(np.array([px, py, pz], dtype=np.float64))

    if not ts:
        raise ValueError(f"No ground-truth rows found in {gt_path}")
    return np.asarray(ts, dtype=np.float64), np.asarray(ps, dtype=np.float64)


def align_by_nearest_timestamp(
    est_t: np.ndarray,
    est_p: np.ndarray,
    gt_t: np.ndarray,
    gt_p: np.ndarray,
    max_dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    aligned_est: list[np.ndarray] = []
    aligned_gt: list[np.ndarray] = []

    for t, p in zip(est_t, est_p):
        idx = int(np.argmin(np.abs(gt_t - t)))
        dt = abs(float(gt_t[idx] - t))
        if dt <= max_dt:
            aligned_est.append(p)
            aligned_gt.append(gt_p[idx])

    if len(aligned_est) < 3:
        raise ValueError("Not enough timestamp matches; increase max_dt or check clock sync")

    return np.asarray(aligned_est, dtype=np.float64), np.asarray(aligned_gt, dtype=np.float64)


def umeyama_align(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    # Returns src transformed into dst frame.
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = np.sum(src_c * src_c) / src.shape[0]
        scale = np.trace(np.diag(D) @ S) / max(var_src, 1e-12)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return (scale * (R @ src.T)).T + t


def ate_metrics(est_p: np.ndarray, gt_p: np.ndarray) -> dict[str, float]:
    e = np.linalg.norm(est_p - gt_p, axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(e * e))),
        "ate_mean_m": float(np.mean(e)),
        "ate_median_m": float(np.median(e)),
        "ate_max_m": float(np.max(e)),
    }


def rpe_rmse(est_p: np.ndarray, gt_p: np.ndarray, delta: int = 1) -> float:
    if len(est_p) <= delta:
        return float("nan")
    e: list[float] = []
    for i in range(len(est_p) - delta):
        de = est_p[i + delta] - est_p[i]
        dg = gt_p[i + delta] - gt_p[i]
        e.append(float(np.linalg.norm(de - dg)))
    arr = np.asarray(e, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))
