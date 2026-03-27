#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def open_camera(camera_id: int, camera_path: str, width: int, height: int) -> cv2.VideoCapture:
    candidates: list[object] = [camera_path, camera_id, f"/dev/video{camera_id}"]
    for cand in candidates:
        cap = cv2.VideoCapture(cand, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"Opened camera source: {cand}")
                return cap
        cap.release()
    raise RuntimeError("Unable to open camera")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture manual reference images for DA3 keyframe memory.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-path", type=str, default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output-dir", type=str, default="data/reference_memory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.camera_id, args.camera_path, args.width, args.height)

    print("Controls: [s] save reference image, [q] quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed")
                continue

            view = frame.copy()
            cv2.putText(view, "s: save  q: quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Reference Capture", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_path = out_dir / f"ref_{ts}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"Saved: {out_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
