from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import serial


@dataclass
class VnymrPacket:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    mag_xyz: np.ndarray
    accel_xyz: np.ndarray
    gyro_xyz_deg_s: np.ndarray


class VN100Serial:
    def __init__(self, port: str, baudrate: int = 115200, timeout_s: float = 0.2) -> None:
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()

    @staticmethod
    def _checksum(payload: str) -> str:
        cs = 0
        for c in payload:
            cs ^= ord(c)
        return f"{cs:02X}"

    def send_ascii_command(self, body_without_dollar_and_star: str) -> None:
        cs = self._checksum(body_without_dollar_and_star)
        msg = f"${body_without_dollar_and_star}*{cs}\r\n"
        self.ser.write(msg.encode("ascii"))

    def read_vnymr(self) -> VnymrPacket | None:
        line = self.ser.readline().decode("ascii", errors="ignore").strip()
        if not line.startswith("$VNYMR"):
            return None

        try:
            payload = line[1:]
            if "*" in payload:
                data_part, _ = payload.split("*", 1)
            else:
                data_part = payload

            tokens = data_part.split(",")
            if len(tokens) < 13:
                return None

            yaw = float(tokens[1])
            pitch = float(tokens[2])
            roll = float(tokens[3])
            mag = np.array([float(tokens[4]), float(tokens[5]), float(tokens[6])], dtype=np.float64)
            acc = np.array([float(tokens[7]), float(tokens[8]), float(tokens[9])], dtype=np.float64)
            gyro = np.array([float(tokens[10]), float(tokens[11]), float(tokens[12])], dtype=np.float64)

            return VnymrPacket(
                yaw_deg=yaw,
                pitch_deg=pitch,
                roll_deg=roll,
                mag_xyz=mag,
                accel_xyz=acc,
                gyro_xyz_deg_s=gyro,
            )
        except (ValueError, IndexError):
            return None
