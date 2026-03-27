from __future__ import annotations

import numpy as np


class MadgwickAHRS:
    def __init__(self, sample_freq_hz: float = 100.0, beta: float = 0.1) -> None:
        self.sample_freq_hz = float(sample_freq_hz)
        self.beta = float(beta)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z

    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < eps:
            return v
        return v / n

    def update(self, gyro_rad_s: np.ndarray, accel: np.ndarray, mag: np.ndarray, dt: float) -> np.ndarray:
        q1, q2, q3, q4 = self.q
        gx, gy, gz = gyro_rad_s
        ax, ay, az = accel
        mx, my, mz = mag

        if dt <= 0.0:
            return self.q

        a = self._normalize(np.array([ax, ay, az], dtype=np.float64))
        m = self._normalize(np.array([mx, my, mz], dtype=np.float64))
        if np.linalg.norm(a) < 1e-9 or np.linalg.norm(m) < 1e-9:
            return self.q

        ax, ay, az = a
        mx, my, mz = m

        _2q1mx = 2.0 * q1 * mx
        _2q1my = 2.0 * q1 * my
        _2q1mz = 2.0 * q1 * mz
        _2q2mx = 2.0 * q2 * mx

        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _2q4 = 2.0 * q4
        _2q1q3 = 2.0 * q1 * q3
        _2q3q4 = 2.0 * q3 * q4

        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q1q4 = q1 * q4
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q2q4 = q2 * q4
        q3q3 = q3 * q3
        q3q4 = q3 * q4
        q4q4 = q4 * q4

        hx = mx * q1q1 - _2q1my * q4 + _2q1mz * q3 + mx * q2q2 + _2q2 * my * q3 + _2q2 * mz * q4 - mx * q3q3 - mx * q4q4
        hy = _2q1mx * q4 + my * q1q1 - _2q1mz * q2 + _2q2mx * q3 - my * q2q2 + my * q3q3 + _2q3 * mz * q4 - my * q4q4
        _2bx = np.sqrt(hx * hx + hy * hy)
        _2bz = -_2q1mx * q3 + _2q1my * q2 + mz * q1q1 + _2q2mx * q4 - mz * q2q2 + _2q3 * my * q4 - mz * q3q3 + mz * q4q4
        _4bx = 2.0 * _2bx
        _4bz = 2.0 * _2bz

        s1 = (
            -_2q3 * (2.0 * q2q4 - _2q1q3 - ax)
            + _2q2 * (2.0 * q1q2 + _2q3q4 - ay)
            - _2bz * q3 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx)
            + (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
            + _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        )
        s2 = (
            _2q4 * (2.0 * q2q4 - _2q1q3 - ax)
            + _2q1 * (2.0 * q1q2 + _2q3q4 - ay)
            - 4.0 * q2 * (1.0 - 2.0 * q2q2 - 2.0 * q3q3 - az)
            + _2bz * q4 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx)
            + (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
            + (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        )
        s3 = (
            -_2q1 * (2.0 * q2q4 - _2q1q3 - ax)
            + _2q4 * (2.0 * q1q2 + _2q3q4 - ay)
            - 4.0 * q3 * (1.0 - 2.0 * q2q2 - 2.0 * q3q3 - az)
            + (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx)
            + (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
            + (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        )
        s4 = (
            _2q2 * (2.0 * q2q4 - _2q1q3 - ax)
            + _2q3 * (2.0 * q1q2 + _2q3q4 - ay)
            + (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx)
            + (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
            + _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        )

        s = self._normalize(np.array([s1, s2, s3, s4], dtype=np.float64))

        q_dot = 0.5 * np.array(
            [
                -q2 * gx - q3 * gy - q4 * gz,
                q1 * gx + q3 * gz - q4 * gy,
                q1 * gy - q2 * gz + q4 * gx,
                q1 * gz + q2 * gy - q3 * gx,
            ],
            dtype=np.float64,
        ) - self.beta * s

        self.q = self._normalize(self.q + q_dot * dt)
        return self.q

    def euler_deg(self) -> np.ndarray:
        q1, q2, q3, q4 = self.q
        roll = np.arctan2(2.0 * (q1 * q2 + q3 * q4), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        pitch = np.arcsin(np.clip(2.0 * (q1 * q3 - q4 * q2), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (q1 * q4 + q2 * q3), 1.0 - 2.0 * (q3 * q3 + q4 * q4))
        return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))
