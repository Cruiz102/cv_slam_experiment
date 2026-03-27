from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.trajectory_metrics import (
    align_by_nearest_timestamp,
    ate_metrics,
    read_euroc_groundtruth,
    read_trajectory_csv,
    rpe_rmse,
    umeyama_align,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark estimated trajectory against EuRoC ground truth")
    p.add_argument("--seq", type=Path, required=True, help="EuRoC sequence root")
    p.add_argument("--est", type=Path, required=True, help="Estimated trajectory CSV (timestamp_s,x,y,z)")
    p.add_argument("--max-sync-dt", type=float, default=0.01, help="Max timestamp difference for alignment in seconds")
    p.add_argument("--no-scale-align", action="store_true", help="Disable scale alignment (useful for stereo/metric estimates)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    est_t, est_p = read_trajectory_csv(args.est)
    gt_t, gt_p = read_euroc_groundtruth(args.seq)

    est_m, gt_m = align_by_nearest_timestamp(est_t, est_p, gt_t, gt_p, max_dt=args.max_sync_dt)
    est_aligned = umeyama_align(est_m, gt_m, with_scale=not args.no_scale_align)

    ate = ate_metrics(est_aligned, gt_m)
    rpe1 = rpe_rmse(est_aligned, gt_m, delta=1)
    rpe10 = rpe_rmse(est_aligned, gt_m, delta=10)

    print("Benchmark Results")
    print("-----------------")
    print(f"matches: {len(est_aligned)}")
    print(f"ATE RMSE (m):   {ate['ate_rmse_m']:.4f}")
    print(f"ATE Mean (m):   {ate['ate_mean_m']:.4f}")
    print(f"ATE Median (m): {ate['ate_median_m']:.4f}")
    print(f"ATE Max (m):    {ate['ate_max_m']:.4f}")
    print(f"RPE RMSE d=1 (m):  {rpe1:.4f}")
    print(f"RPE RMSE d=10 (m): {rpe10:.4f}")


if __name__ == "__main__":
    main()
