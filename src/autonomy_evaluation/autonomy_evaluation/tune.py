#!/usr/bin/env python3
"""
tune.py  —  ESEKF-12 parameter optimiser
Fixes applied vs previous version:
  1. GNSS converted to ENU relative to first fix, NOT absolute globe metres.
  2. EKF initialised from GT position so GNSS measurement residuals start near zero.
  3. Optuna penalty thresholds derived from the actual baseline (not hard-coded 10 deg).
  4. Wheel yaw-rate sign / wheelbase match carla_node patch (WHEEL_YAW_SIGN=-1, wb=2.875).
  5. Compass & GNSS yaw sign inversions confirmed from debug audit.
  6. VO events loaded from vo_aligned.csv and fed into simulation.
  7. search space bounded to [0.05x, 20x] of baseline instead of fixed log range.
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CONSTANTS  (must match carla_node & C++ EKF exactly)
# ──────────────────────────────────────────────────────────────
DEFAULT_RUN_DIR = os.path.join(os.getcwd(), "eval_temp")
EARTH_RADIUS    = 6378137.0
MATCH_TOL       = 0.05          # seconds for merge_asof
SETTLING_TIME   = 5.0           # seconds to skip after start

RAD = np.pi / 180.0

# From debug audit:
#   GNSS Heading  → yaw sign inversion + ~0 offset  ✓
#   MAG Compass   → yaw sign inversion + ~0 offset  ✓
#   Wheel yaw-rate sign is negative (vehicle turns opposite to steer convention)
GNSS_YAW_SIGN    = -1.0
COMPASS_SIGN     = -1.0
WHEEL_YAW_SIGN   = -1.0
WHEELBASE        = 2.875        # Tesla Model 3 in CARLA (metres)
TRACK_WIDTH      = 1.580        # Tesla Model 3 in CARLA (metres)
MAX_STEER_RAD    = 1.22         # carla_node: steer_norm * 1.22

# Lever-arm offsets (front GNSS antenna)
GNSS_FRONT_OFFSET = 1.11        # metres forward of vehicle centre

# ──────────────────────────────────────────────────────────────
# MATH UTILITIES
# ──────────────────────────────────────────────────────────────
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def quaternion_to_euler(q):
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z))
    return wrap(yaw), wrap(pitch), wrap(roll)

def euler_to_quaternion(yaw, pitch, roll):
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])

def small_angle_quaternion(d):
    t = np.linalg.norm(d)
    if t < 1e-10:
        q = np.array([1.0, 0.5*d[0], 0.5*d[1], 0.5*d[2]])
    else:
        q = np.array([np.cos(t/2), *(d/t * np.sin(t/2))])
    return q / np.linalg.norm(q)

def quaternion_multiply(a, b):
    w1,x1,y1,z1 = a
    w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 + y1*w2 + z1*x2 - x1*z2,  # ← fixed
        w1*z2 + z1*w2 + x1*y2 - y1*x2   # ← fixed
    ])

# ──────────────────────────────────────────────────────────────
# ESEKF-12  (position · attitude · velocity · gyro-bias)
# ──────────────────────────────────────────────────────────────
class ESEKF12:
    def __init__(self, p):
        self.x = np.zeros(13)
        self.x[3] = 1.0          # qw = 1
        self.P = np.eye(12)
        self.initialized = False

        self.Q = np.diag([
            p["q_pos_x"], p["q_pos_y"], p["q_pos_z"],
            p["q_att_x"], p["q_att_y"], p["q_att_z"],
            p["q_vel_x"], p["q_vel_y"], p["q_vel_z"],
            p["q_bias_x"],p["q_bias_y"],p["q_bias_z"]
        ])
        self.R_gnss      = np.diag([p["r_gnss_x"],    p["r_gnss_y"],    p["r_gnss_z"]])
        self.R_gnss_yaw  = np.array([[p["r_gnss_yaw"]]])
        self.R_compass   = np.array([[p["r_compass"]]])
        self.R_wheel_vel = np.array([[p["r_wheel_vel"]]])
        self.R_wheel_yaw = np.array([[p["r_wheel_yaw"]]])
        self.R_nhc       = np.diag([5.0, 5.0])
        self.R_vo        = np.diag([p["r_vo_x"], p["r_vo_y"], p["r_vo_z"]])
        self.R_slam      = np.diag([p["r_slam_x"], p["r_slam_y"], p["r_slam_z"]])
        self.R_slam_yaw  = np.array([[p["r_slam_yaw"]]])

    def initialize_state(self, x, y, z, v, yaw, pitch=0.0, roll=0.0):
        q = euler_to_quaternion(yaw, pitch, roll)
        self.x = np.array([x, y, z, *q, v, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.initialized = True

    def yaw(self):
        return quaternion_to_euler(self.x[3:7])[0]

    def update(self, residual, H, R, angle=False):
        y = np.asarray(residual, dtype=float).reshape(-1, 1)
        if angle:
            y[0, 0] = wrap(y[0, 0])
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        dx = (K @ y).ravel()
        self.x[:3]   += dx[:3]
        self.x[3:7]   = quaternion_multiply(self.x[3:7], small_angle_quaternion(dx[3:6]))
        self.x[3:7]  /= np.linalg.norm(self.x[3:7])
        self.x[7:10] += dx[6:9]
        self.x[10:13]+= dx[9:12]
        I = np.eye(12) - K @ H
        self.P = I @ self.P @ I.T + K @ R @ K.T

    def predict(self, dt, ax, ay, az, gx, gy, gz):
        if not self.initialized or dt <= 0:
            return
        q  = self.x[3:7]
        bg = self.x[10:13]
        omega = np.array([gx, gy, gz]) - bg
        qn = quaternion_multiply(q, small_angle_quaternion(omega * dt))
        qn /= np.linalg.norm(qn)
        yaw = quaternion_to_euler(qn)[0]
        c, s = np.cos(yaw), np.sin(yaw)
        aw = np.array([ax*c - ay*s,  ax*s + ay*c,  0.0])
        self.x[:3]   += self.x[7:10] * dt
        self.x[3:7]   = qn
        self.x[7:10] += aw * dt
        F = np.eye(12)
        F[:3,  6:9]  = np.eye(3) * dt
        F[3:6, 9:12] = -np.eye(3) * dt
        F[6, 5] = (-ax*s - ay*c) * dt
        F[7, 5] = ( ax*c - ay*s) * dt
        self.P = F @ self.P @ F.T + self.Q * max(dt, 1e-3)

# ──────────────────────────────────────────────────────────────
# BASELINE PARAMETERS  (from C++ EKF — tuning multiplies these)
# ──────────────────────────────────────────────────────────────
BASE_Q = [
    0.002434079267, 0.3016930045, 6.763067319e-07,  # q_pos
    0.00189155875, 3.906643206e-05, 1.019373728,     # q_att
    1.499787478e-05, 0.0007853073166, 5.974407434e-05, # q_vel
    2.84056179e-07, 0.0003355011837, 0.0002529653963  # q_bias
]

BASE_R = {
    "r_gnss_x": 0.00157071002,
    "r_gnss_y": 0.05853417464,
    "r_gnss_z": 0.5900535799,
    "r_gnss_yaw": 0.3782675436,
    "r_compass": 1523.666788,
    "r_wheel_vel": 22.27563706,
    "r_wheel_yaw": 0.5215293569,
    "r_vo_x": 3.092217883,
    "r_vo_y": 0.02518252328,
    "r_vo_z": 0.008813331433,
    "r_slam_x": 108.2642282,
    "r_slam_y": 79.49254327,
    "r_slam_z": 316.5679796,
    "r_slam_yaw": 1940.664139,
}

BASE_PARAMS = {}
for i, k in enumerate("xyz"):
    BASE_PARAMS[f"q_pos_{k}"]  = BASE_Q[i]
    BASE_PARAMS[f"q_att_{k}"]  = BASE_Q[i+3]
    BASE_PARAMS[f"q_vel_{k}"]  = BASE_Q[i+6]
    BASE_PARAMS[f"q_bias_{k}"] = BASE_Q[i+9]
BASE_PARAMS.update(BASE_R)

# ──────────────────────────────────────────────────────────────
# EKF SIMULATION
# ──────────────────────────────────────────────────────────────
def run_ekf_simulation(events, params, gt_init):
    """
    gt_init : dict with keys x, y, z, yaw  — taken from the first GT row.
              Used to initialise EKF position so GNSS residuals start near zero.
    """
    ekf = ESEKF12(params)
    results = []

    last_imu_t     = None
    last_wheel_t   = None
    wheel_yaw      = 0.0
    wheel_init     = False

    # GNSS ENU origin — set on first GNSS event so all subsequent fixes are relative
    gnss_ox = None
    gnss_oy = None
    gnss_oz = None

    for t, typ, row in events:

        # ── IMU ──────────────────────────────────────────────
        if typ == "imu":
            if last_imu_t is None:
                last_imu_t = t
                continue
            dt = min(t - last_imu_t, 0.1)
            last_imu_t = t
            if ekf.initialized:
                ekf.predict(dt,
                    row["Accel_X"], row["Accel_Y"], row["Accel_Z"],
                    row["Gyro_X"],  row["Gyro_Y"],  row["Gyro_Z"])
                results.append((t, ekf.x[0], ekf.x[1], ekf.x[2], ekf.yaw()))

        # ── GNSS ─────────────────────────────────────────────
        elif typ == "gnss":
            # Raw absolute metres from (0°,0°) on the globe
            fx_abs = row["Front_Lon"] * RAD * EARTH_RADIUS
            fy_abs = row["Front_Lat"] * RAD * EARTH_RADIUS
            fz_abs = row["Front_Alt"]
            rx_abs = row["Rear_Lon"]  * RAD * EARTH_RADIUS
            ry_abs = row["Rear_Lat"]  * RAD * EARTH_RADIUS
            rz_abs = row["Rear_Alt"]

            # Lock ENU origin on first fix
            if gnss_ox is None:
                gnss_ox = fx_abs
                gnss_oy = fy_abs
                gnss_oz = fz_abs

            # ENU-relative positions
            fx = fx_abs - gnss_ox
            fy = fy_abs - gnss_oy
            fz = fz_abs - gnss_oz
            rx = rx_abs - gnss_ox
            ry = ry_abs - gnss_oy
            rz = rz_abs - gnss_oz

            # Heading from dual-antenna baseline
            raw_yaw = np.arctan2(fy - ry, fx - rx)

            # Vehicle-centre position (lever-arm removal, same as C++ EKF)
            veh_x = fx - GNSS_FRONT_OFFSET * np.cos(raw_yaw)
            veh_y = fy - GNSS_FRONT_OFFSET * np.sin(raw_yaw)
            veh_z = (fz + rz) / 2.0 - 1.5

            # Apply signed heading (confirmed by debug audit)
            gnss_yaw = wrap(GNSS_YAW_SIGN * raw_yaw)

            # ── Initialise EKF from GT, not GNSS ──────────
            # This eliminates the ~335m GNSS-vs-GT offset at t=0.
            if not ekf.initialized:
                ekf.initialize_state(
                    gt_init["x"], gt_init["y"], gt_init["z"],
                    0.0, gt_init["yaw"]
                )
                # Do NOT lock origin yet — wait until first update below
                continue

            # ── GNSS position update ──────────────────────
            # Offset GNSS ENU to GT world frame using the difference recorded
            # when the EKF was initialised.  We compute this offset once.
            if not hasattr(ekf, "_gnss_to_gt_ox"):
                ekf._gnss_to_gt_ox = gt_init["x"] - veh_x
                ekf._gnss_to_gt_oy = gt_init["y"] - veh_y
                ekf._gnss_to_gt_oz = gt_init["z"] - veh_z

            meas_x = veh_x + ekf._gnss_to_gt_ox
            meas_y = veh_y + ekf._gnss_to_gt_oy
            meas_z = veh_z + ekf._gnss_to_gt_oz

            H = np.zeros((3, 12)); H[:3, :3] = np.eye(3)
            ekf.update([meas_x - ekf.x[0],
                        meas_y - ekf.x[1],
                        meas_z - ekf.x[2]], H, ekf.R_gnss)

            # Dual-antenna yaw update
            H = np.zeros((1, 12)); H[0, 5] = 1.0
            ekf.update([wrap(gnss_yaw - ekf.yaw())], H, ekf.R_gnss_yaw, True)

        # ── MAGNETOMETER ─────────────────────────────────────
        elif typ == "mag" and ekf.initialized:
            raw_yaw   = quaternion_to_euler([row["Qw"], row["Qx"], row["Qy"], row["Qz"]])[0]
            raw_deg   = np.degrees(raw_yaw) % 360.0
            cal_deg   = ((360.0 - raw_deg) - 45.0) % 360.0   # same formula as C++ & metric_viz
            cal_yaw   = wrap(COMPASS_SIGN * np.radians(cal_deg))
            H = np.zeros((1, 12)); H[0, 5] = 1.0
            ekf.update([wrap(cal_yaw - ekf.yaw())], H, ekf.R_compass, True)

        # ── WHEEL ODOMETRY ───────────────────────────────────
        elif typ == "wheel" and ekf.initialized:
            speed = float(row["Odom_Velocity"])
            steer = float(row.get("Steering_Angle", 0.0))
            yaw   = ekf.yaw()
            c, s  = np.cos(yaw), np.sin(yaw)

            # Speed update
            v_est = ekf.x[7]*c + ekf.x[8]*s
            H = np.zeros((1, 12)); H[0, 6] = c; H[0, 7] = s
            ekf.update([speed - v_est], H, ekf.R_wheel_vel)

            # Non-holonomic constraint (lateral & vertical velocity ~ 0)
            H = np.zeros((2, 12)); H[0, 7] = 1.0; H[1, 8] = 1.0
            ekf.update([-ekf.x[8], -ekf.x[9]], H, ekf.R_nhc)

            # Kinematic yaw-rate from steering angle
            if not wheel_init:
                wheel_yaw  = yaw
                wheel_init = True
                last_wheel_t = t
            else:
                dt = t - last_wheel_t
                if 0 < dt < 0.2:
                    yaw_rate  = WHEEL_YAW_SIGN * (speed / WHEELBASE) * np.tan(steer) / TRACK_WIDTH
                    wheel_yaw = wrap(wheel_yaw + yaw_rate * dt)
                    H = np.zeros((1, 12)); H[0, 5] = 1.0
                    ekf.update([wrap(wheel_yaw - yaw)], H, ekf.R_wheel_yaw, True)
                last_wheel_t = t

        # ── VISUAL ODOMETRY ──────────────────────────────────
        elif typ == "vo" and ekf.initialized:
            H = np.zeros((3, 12)); H[:3, :3] = np.eye(3)
            ekf.update([row["VO_X"] - ekf.x[0],
                        row["VO_Y"] - ekf.x[1],
                        row["VO_Z"] - ekf.x[2]], H, ekf.R_vo)

        # ── LiDAR SLAM ───────────────────────────────────────
        elif typ == "slam" and ekf.initialized:
            H = np.zeros((3, 12)); H[:3, :3] = np.eye(3)
            ekf.update([row["SLAM_X"] - ekf.x[0],
                        row["SLAM_Y"] - ekf.x[1],
                        row["SLAM_Z"] - ekf.x[2]], H, ekf.R_slam)
            H = np.zeros((1, 12)); H[0, 5] = 1.0
            slam_yaw = wrap(float(row["SLAM_Yaw"]))
            ekf.update([wrap(slam_yaw - ekf.yaw())], H, ekf.R_slam_yaw, True)

    return pd.DataFrame(results, columns=["Timestamp", "X", "Y", "Z", "Yaw"])

# ──────────────────────────────────────────────────────────────
# METRIC CALCULATION
# ──────────────────────────────────────────────────────────────
def calculate_metrics(sim, gt):
    if sim.empty:
        return np.inf, np.inf

    gt  = gt.sort_values("Timestamp").drop_duplicates("Timestamp")
    sim = sim.sort_values("Timestamp")

    merged = pd.merge_asof(sim, gt, on="Timestamp",
                           direction="nearest", tolerance=MATCH_TOL
                           ).dropna(subset=["GT_X", "GT_Y", "GT_Z", "GT_Yaw"])

    if len(merged) < 10:
        return np.inf, np.inf

    # Skip settling window using GT timeline (not sim timeline)
    start = gt["Timestamp"].min() + SETTLING_TIME
    merged = merged[merged["Timestamp"] >= start]
    if merged.empty:
        return np.inf, np.inf

    pos = np.sqrt(
        (merged["X"].to_numpy() - merged["GT_X"].to_numpy())**2 +
        (merged["Y"].to_numpy() - merged["GT_Y"].to_numpy())**2 +
        (merged["Z"].to_numpy() - merged["GT_Z"].to_numpy())**2
    )
    yaw = wrap(merged["Yaw"].to_numpy() - merged["GT_Yaw"].to_numpy())

    rmse_pos = float(np.sqrt(np.mean(pos**2)))
    rmse_yaw = float(np.sqrt(np.mean(np.degrees(yaw)**2)))
    return rmse_pos, rmse_yaw

def evaluate_stored_baseline(ekf_csv):
    sim = pd.DataFrame({
        "Timestamp": ekf_csv["GT_Timestamp"],
        "X":   ekf_csv["EKF_X"],
        "Y":   ekf_csv["EKF_Y"],
        "Z":   ekf_csv["EKF_Z"],
        "Yaw": ekf_csv["EKF_Yaw"],
    })
    gt = ekf_csv[["GT_Timestamp","GT_X","GT_Y","GT_Z","GT_Yaw"]].rename(
        columns={"GT_Timestamp":"Timestamp"})
    return calculate_metrics(sim, gt)

# ──────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE
# ──────────────────────────────────────────────────────────────
def suggest_params(trial):
    """Each param explored in [0.05×, 20×] of baseline on log scale."""
    return {
        k: trial.suggest_float(k, v * 0.05, v * 20.0, log=True)
        for k, v in BASE_PARAMS.items()
    }

def objective(trial, events, gt, gt_init, base_pos, base_yaw):
    try:
        params = suggest_params(trial)
        sim    = run_ekf_simulation(events, params, gt_init)
        pos, yaw = calculate_metrics(sim, gt)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 1e9

    trial.set_user_attr("rmse_pos", pos)
    trial.set_user_attr("rmse_yaw", yaw)

    if not np.isfinite(pos) or not np.isfinite(yaw):
        return 1e9

    # ── Penalty thresholds derived from actual baseline ────
    # Reject solutions that are catastrophically worse (3× baseline),
    # but never reject based on an absolute number that's smaller than the baseline.
    pos_limit = max(base_pos * 3.0, 15.0)
    yaw_limit = max(base_yaw * 3.0, 30.0)

    if yaw > yaw_limit:
        return 1e6 + yaw
    if pos > pos_limit:
        return 1e5 + pos

    # Normalised score — equal weight on pos and yaw improvement
    score = (
        0.5 * (pos / max(base_pos, 1e-6)) +
        0.5 * (yaw / max(base_yaw, 1e-6))
    )
    return score

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--trials",  type=int, default=2)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    run_dir     = os.path.abspath(args.run_dir)
    sensor_dir  = os.path.join(run_dir, "sensors")
    aligned_dir = os.path.join(run_dir, "aligned")

    print("📊 Loading evaluation datasets...")

    # ── Load raw sensors ──────────────────────────────────
    imu   = pd.read_csv(os.path.join(sensor_dir, "imu.csv"))
    gnss  = pd.read_csv(os.path.join(sensor_dir, "gnss.csv"))
    mag   = pd.read_csv(os.path.join(sensor_dir, "magnetometer.csv"))
    wheel = pd.read_csv(os.path.join(sensor_dir, "wheel_odom.csv"))

    # ── Load aligned outputs ──────────────────────────────
    ekf_csv = pd.read_csv(os.path.join(aligned_dir, "ekf_aligned.csv"))

    # VO — use VO timestamps (not GT timestamps) to avoid compounding alignment error
    vo_path = os.path.join(aligned_dir, "vo_aligned.csv")
    if os.path.exists(vo_path):
        vo = pd.read_csv(vo_path)
        if "VO_Timestamp" in vo.columns:
            vo["Timestamp"] = pd.to_numeric(vo["VO_Timestamp"], errors="coerce")
        else:
            vo["Timestamp"] = pd.to_numeric(vo["GT_Timestamp"], errors="coerce")
        vo = vo.rename(columns={"VO_X": "VO_X", "VO_Y": "VO_Y", "VO_Z": "VO_Z"})
    else:
        vo = pd.DataFrame()

    # SLAM — use SLAM timestamps
    slam_path = os.path.join(aligned_dir, "slam_aligned.csv")
    if os.path.exists(slam_path):
        slam = pd.read_csv(slam_path)
        if "SLAM_Timestamp" in slam.columns:
            slam["Timestamp"] = pd.to_numeric(slam["SLAM_Timestamp"], errors="coerce")
        else:
            slam["Timestamp"] = pd.to_numeric(slam["GT_Timestamp"], errors="coerce")
    else:
        slam = pd.DataFrame()

    # ── Build GT table ────────────────────────────────────
    gt = ekf_csv[["GT_Timestamp","GT_X","GT_Y","GT_Z","GT_Yaw","GT_Velocity"]].rename(
        columns={"GT_Timestamp": "Timestamp"})
    gt = gt.sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)

    gt_start = gt["Timestamp"].min()
    gt_end   = gt["Timestamp"].max()

    # GT initial state used to seed EKF — avoids 335m GNSS offset
    gt_init = {
        "x":   float(gt["GT_X"].iloc[0]),
        "y":   float(gt["GT_Y"].iloc[0]),
        "z":   float(gt["GT_Z"].iloc[0]),
        "yaw": float(gt["GT_Yaw"].iloc[0]),
    }

    print(f"  GT origin  : X={gt_init['x']:.2f}  Y={gt_init['y']:.2f}  "
          f"Z={gt_init['z']:.2f}  Yaw={math.degrees(gt_init['yaw']):.1f}°")

    # ── Coerce timestamps ─────────────────────────────────
    for df in [imu, gnss, mag, wheel]:
        df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")

    # ── Trim all sensors to GT window ─────────────────────
    def trim(df):
        return df[(df["Timestamp"] >= gt_start) &
                  (df["Timestamp"] <= gt_end)
                  ].dropna(subset=["Timestamp"]).copy()

    imu, gnss, mag, wheel = map(trim, [imu, gnss, mag, wheel])
    if not vo.empty:
        vo = trim(vo)
    if not slam.empty:
        slam = trim(slam)

    # ── Build chronological event queue ───────────────────
    events = []
    for df, typ in [
        (imu,   "imu"),
        (gnss,  "gnss"),
        (mag,   "mag"),
        (wheel, "wheel"),
        (vo,    "vo"),
        (slam,  "slam"),
    ]:
        if not df.empty:
            events.extend(
                (float(r["Timestamp"]), typ, r)
                for r in df.to_dict("records")
            )

    events.sort(key=lambda x: x[0])
    print(f"🔄 Sensor events : {len(events)}")
    print(f"🔄 GT window     : {gt_start:.3f} → {gt_end:.3f} s\n")

    # ── Evaluate stored baseline (real C++ EKF result) ────
    stored_pos, stored_yaw = evaluate_stored_baseline(ekf_csv)
    print("========== STORED BASELINE (C++ EKF) ==========")
    print(f"  Position RMSE : {stored_pos:.4f} m")
    print(f"  Yaw RMSE      : {stored_yaw:.4f} deg")
    print("================================================\n")

    # ── Evaluate Python simulation at BASE_PARAMS ─────────
    sim_base = run_ekf_simulation(events, BASE_PARAMS, gt_init)
    base_pos, base_yaw = calculate_metrics(sim_base, gt)
    print("========== SIMULATION BASELINE (BASE_PARAMS) ==========")
    print(f"  Position RMSE : {base_pos:.4f} m")
    print(f"  Yaw RMSE      : {base_yaw:.4f} deg")
    print("========================================================\n")

    if not np.isfinite(base_pos) or not np.isfinite(base_yaw):
        print("❌ Simulation baseline returned inf/nan.  Check sensor data.")
        print("   Most likely cause: GNSS origin or GT-init mismatch.")
        return 1

    # ── Optuna study ──────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=min(20, args.trials)
        )
    )

    # Seed with baseline so first trial is always a known-good point
    study.enqueue_trial(BASE_PARAMS)

    bar = tqdm(total=args.trials, dynamic_ncols=True,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | {postfix}")

    for _ in range(args.trials):
        study.optimize(
            lambda trial: objective(trial, events, gt, gt_init, base_pos, base_yaw),
            n_trials=1,
            show_progress_bar=False
        )
        best = study.best_trial
        bar.set_postfix_str(
            f"pos {best.user_attrs.get('rmse_pos', np.inf):.3f} m | "
            f"yaw {best.user_attrs.get('rmse_yaw', np.inf):.2f} deg"
        )
        bar.update(1)

    bar.close()

    best_params = study.best_params
    tuned = run_ekf_simulation(events, best_params, gt_init)
    tuned_pos, tuned_yaw = calculate_metrics(tuned, gt)

    # ── Print results ─────────────────────────────────────
    print("\n========== FINAL TUNED PARAMETERS ==========")
    print("\nQ  (process noise):")
    for grp, label in [("pos","pos"), ("att","att"), ("vel","vel"), ("bias","bias")]:
        vals = [best_params[f"q_{grp}_{k}"] for k in "xyz"]
        print(f"  q_{label:<5} = [{vals[0]:.10g}, {vals[1]:.10g}, {vals[2]:.10g}]")

    print("\nR  (measurement noise):")
    print(f"  r_gnss_pos   = [{best_params['r_gnss_x']:.10g}, "
          f"{best_params['r_gnss_y']:.10g}, {best_params['r_gnss_z']:.10g}]")
    print(f"  r_gnss_yaw   = {best_params['r_gnss_yaw']:.10g}")
    print(f"  r_compass    = {best_params['r_compass']:.10g}")
    print(f"  r_wheel_vel  = {best_params['r_wheel_vel']:.10g}")
    print(f"  r_wheel_yaw  = {best_params['r_wheel_yaw']:.10g}")
    print(f"  r_vo_pos     = [{best_params['r_vo_x']:.10g}, "
          f"{best_params['r_vo_y']:.10g}, {best_params['r_vo_z']:.10g}]")
    print(f"  r_slam_pos   = [{best_params['r_slam_x']:.10g}, "
          f"{best_params['r_slam_y']:.10g}, {best_params['r_slam_z']:.10g}]")
    print(f"  r_slam_yaw   = {best_params['r_slam_yaw']:.10g}")

    print("\n========== RESULTS ==========")
    print(f"  Stored C++ EKF  pos RMSE : {stored_pos:.4f} m")
    print(f"  Stored C++ EKF  yaw RMSE : {stored_yaw:.4f} deg")
    print(f"  Sim baseline    pos RMSE : {base_pos:.4f} m")
    print(f"  Sim baseline    yaw RMSE : {base_yaw:.4f} deg")
    print(f"  Tuned           pos RMSE : {tuned_pos:.4f} m")
    print(f"  Tuned           yaw RMSE : {tuned_yaw:.4f} deg")
    print(f"  Objective score          : {study.best_value:.6f}")

    def delta(a, b, label, unit):
        if b < a:
            print(f"  ✅ {label} improved by {a-b:.4f} {unit}")
        else:
            print(f"  ⚠️  {label} worsened by {b-a:.4f} {unit}")

    delta(base_pos, tuned_pos, "Position", "m")
    delta(base_yaw, tuned_yaw, "Yaw",      "deg")

    # Save tuned trajectory for manual inspection
    out_csv = os.path.join(run_dir, "tuned_ekf.csv")
    tuned.to_csv(out_csv, index=False)
    print(f"\n📄 Tuned trajectory saved → {out_csv}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())