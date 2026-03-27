#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def open_camera(camera_id: int, camera_path: str, width: int, height: int) -> tuple[cv2.VideoCapture, str]:
    """Open camera using V4L2 first, then CAP_ANY fallback."""
    candidates: list[object] = []
    if camera_path:
        candidates.append(camera_path)
    candidates.extend([camera_id, f"/dev/video{camera_id}"])

    for cand in candidates:
        for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
            cap = cv2.VideoCapture(cand, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, f"{cand} (backend={backend})"
            cap.release()

    raise RuntimeError("Unable to open camera with provided id/path")


def next_index(out_dir: Path, prefix: str) -> int:
    """Find next numeric index based on existing files."""
    max_idx = -1
    for p in out_dir.glob(f"{prefix}_*.jpg"):
        stem = p.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[-1])
            max_idx = max(max_idx, idx)
        except ValueError:
            continue
    return max_idx + 1


def blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian: higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def save_frame(frame: np.ndarray, out_dir: Path, prefix: str, idx: int) -> Path:
    out_path = out_dir / f"{prefix}_{idx:06d}.jpg"
    cv2.imwrite(str(out_path), frame)
    return out_path


def draw_hud(
    frame: np.ndarray,
    saved_count: int,
    auto_mode: bool,
    auto_interval_s: float,
    sharpness: float,
    sharpness_min: float,
    show_grid: bool,
) -> np.ndarray:
    view = frame.copy()

    if show_grid:
        h, w = view.shape[:2]
        for frac in (1 / 3, 2 / 3):
            x = int(w * frac)
            y = int(h * frac)
            cv2.line(view, (x, 0), (x, h), (80, 80, 80), 1, cv2.LINE_AA)
            cv2.line(view, (0, y), (w, y), (80, 80, 80), 1, cv2.LINE_AA)

    status = "AUTO" if auto_mode else "MANUAL"
    color = (0, 200, 255) if auto_mode else (0, 255, 120)
    sharp_color = (0, 255, 120) if sharpness >= sharpness_min else (0, 80, 255)

    cv2.putText(view, f"Mode: {status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(view, f"Saved: {saved_count}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(view, f"Auto interval: {auto_interval_s:.1f}s", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(view, f"Sharpness: {sharpness:.1f} (min {sharpness_min:.1f})", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sharp_color, 2)

    controls = "s:save a:auto g:grid +/-:interval q:quit"
    cv2.putText(view, controls, (12, view.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 2)
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture relocalization map images with OpenCV.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-path", type=str, default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output-dir", type=str, default="data/mapping_images")
    parser.add_argument("--prefix", type=str, default="map")
    parser.add_argument("--auto-interval", type=float, default=1.0, help="Seconds between auto-captures.")
    parser.add_argument("--sharpness-min", type=float, default=80.0, help="Blur threshold for warning only.")
    parser.add_argument("--start-auto", action="store_true", help="Start in auto-capture mode.")
    parser.add_argument("--show-grid", action="store_true", help="Show rule-of-thirds overlay grid.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap, src_info = open_camera(args.camera_id, args.camera_path, args.width, args.height)
    print(f"Opened camera source: {src_info}")
    print(f"Saving images to: {out_dir}")

    idx = next_index(out_dir, args.prefix)
    auto_mode = bool(args.start_auto)
    show_grid = bool(args.show_grid)
    auto_interval_s = max(0.2, float(args.auto_interval))
    sharpness_min = max(1.0, float(args.sharpness_min))
    last_auto_save_t = 0.0
    saved_count = 0

    print("Controls:")
    print("  s: save image now")
    print("  a: toggle auto-capture")
    print("  g: toggle grid")
    print("  + / - : increase/decrease auto interval")
    print("  q: quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed")
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = blur_score(gray)

            now = time.monotonic()
            if auto_mode and (now - last_auto_save_t) >= auto_interval_s:
                path = save_frame(frame, out_dir, args.prefix, idx)
                idx += 1
                saved_count += 1
                last_auto_save_t = now
                if sharp < sharpness_min:
                    print(f"Saved (blurry warning): {path} | sharpness={sharp:.1f}")
                else:
                    print(f"Saved: {path} | sharpness={sharp:.1f}")

            hud = draw_hud(
                frame,
                saved_count=saved_count,
                auto_mode=auto_mode,
                auto_interval_s=auto_interval_s,
                sharpness=sharp,
                sharpness_min=sharpness_min,
                show_grid=show_grid,
            )
            cv2.imshow("Relocalization Map Capture", hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = save_frame(frame, out_dir, args.prefix, idx)
                idx += 1
                saved_count += 1
                if sharp < sharpness_min:
                    print(f"Saved (blurry warning): {path} | sharpness={sharp:.1f}")
                else:
                    print(f"Saved: {path} | sharpness={sharp:.1f}")
            elif key == ord("a"):
                auto_mode = not auto_mode
                print(f"Auto-capture: {'ON' if auto_mode else 'OFF'}")
                last_auto_save_t = time.monotonic()
            elif key == ord("g"):
                show_grid = not show_grid
                print(f"Grid overlay: {'ON' if show_grid else 'OFF'}")
            elif key in (ord("+"), ord("=")):
                auto_interval_s = min(10.0, auto_interval_s + 0.2)
                print(f"Auto interval: {auto_interval_s:.1f}s")
            elif key == ord("-"):
                auto_interval_s = max(0.2, auto_interval_s - 0.2)
                print(f"Auto interval: {auto_interval_s:.1f}s")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Total saved this session: {saved_count}")


if __name__ == "__main__":
    main()
