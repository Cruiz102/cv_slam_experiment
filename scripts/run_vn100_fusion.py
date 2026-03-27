from __future__ import annotations

import argparse
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from src.inertial.madgwick import MadgwickAHRS
from src.inertial.vn100_serial import VN100Serial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read VN100 serial stream and run orientation fusion (Madgwick)")
    p.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--beta", type=float, default=0.08, help="Madgwick gain")
    p.add_argument("--window-sec", type=float, default=20.0, help="Time window for plots")
    p.add_argument("--configure-vnymr", action="store_true", help="Send VN command to configure async output to VNYMR")
    return p.parse_args()


def configure_vnymr(sensor: VN100Serial) -> None:
    # This command configures async output type to VNYMR and async rate divisor to 1.
    # Depending on firmware, you may need to adjust this with Control Center.
    sensor.send_ascii_command("VNWRG,06,14")
    sensor.send_ascii_command("VNWRG,07,1")


def main() -> None:
    args = parse_args()

    sensor = VN100Serial(port=args.port, baudrate=args.baud)
    if args.configure_vnymr:
        configure_vnymr(sensor)
        time.sleep(0.2)

    filt = MadgwickAHRS(beta=args.beta)

    plt.ion()
    fig, (ax_r, ax_p, ax_y) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle("VN100 IMU Fusion (Madgwick)")

    t_hist: deque[float] = deque()
    r_hist: deque[float] = deque()
    p_hist: deque[float] = deque()
    y_hist: deque[float] = deque()

    start = time.monotonic()
    last_t = time.monotonic()

    try:
        while True:
            pkt = sensor.read_vnymr()
            if pkt is None:
                plt.pause(0.001)
                continue

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt <= 0.0 or dt > 0.5:
                dt = 1.0 / 100.0

            gyro_rad_s = np.radians(pkt.gyro_xyz_deg_s)
            q = filt.update(gyro_rad_s=gyro_rad_s, accel=pkt.accel_xyz, mag=pkt.mag_xyz, dt=dt)
            roll, pitch, yaw = filt.euler_deg()

            t_rel = now - start
            t_hist.append(t_rel)
            r_hist.append(float(roll))
            p_hist.append(float(pitch))
            y_hist.append(float(yaw))

            while t_hist and (t_rel - t_hist[0]) > args.window_sec:
                t_hist.popleft()
                r_hist.popleft()
                p_hist.popleft()
                y_hist.popleft()

            ax_r.cla()
            ax_p.cla()
            ax_y.cla()

            t_vals = np.array(t_hist, dtype=np.float64)
            ax_r.plot(t_vals, np.array(r_hist), color="tab:red")
            ax_p.plot(t_vals, np.array(p_hist), color="tab:green")
            ax_y.plot(t_vals, np.array(y_hist), color="tab:blue")

            ax_r.set_ylabel("Roll (deg)")
            ax_p.set_ylabel("Pitch (deg)")
            ax_y.set_ylabel("Yaw (deg)")
            ax_y.set_xlabel("Time (s)")

            ax_r.grid(True, alpha=0.3)
            ax_p.grid(True, alpha=0.3)
            ax_y.grid(True, alpha=0.3)

            fig.canvas.draw_idle()
            plt.pause(0.001)

            if int(t_rel * 10) % 10 == 0:
                print(
                    f"rpy_fused=({roll:+7.2f}, {pitch:+7.2f}, {yaw:+7.2f}) deg | "
                    f"rpy_vn100=({pkt.roll_deg:+7.2f}, {pkt.pitch_deg:+7.2f}, {pkt.yaw_deg:+7.2f}) deg"
                )

    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()
        plt.close(fig)


if __name__ == "__main__":
    main()
