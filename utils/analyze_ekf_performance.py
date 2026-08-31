#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def load_csv(path):
    if not os.path.exists(path):
        print(f"[!] Log file not found at: {path}")
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df.sort_values("Timestamp").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="/home/ubuntu/AV6/repo-dev/eval_plots/tuning_data/gt_data.csv")
    ap.add_argument("--ekf", default="/home/ubuntu/AV6/repo-dev/eval_plots/tuning_data/ekf_data.csv")
    ap.add_argument("--warmup", type=float, default=2.5)
    args = ap.parse_args()

    gt_df = load_csv(args.gt)
    ekf_df = load_csv(args.ekf)

    if gt_df is None or ekf_df is None:
        print("\n[❌] Missing files! You might need to run the CARLA simulation one more time.")
        sys.exit(1)

    # 1. Warm-up crop
    t_start = max(gt_df["Timestamp"].min(), ekf_df["Timestamp"].min())
    ekf_df = ekf_df[ekf_df["Timestamp"] >= t_start + args.warmup].copy()

    # 2. Synchronize
    merged = pd.merge_asof(ekf_df, gt_df, on="Timestamp", direction="nearest").dropna(subset=["Loc_X", "Loc_Y", "Est_X", "Est_Y"])

    # 3. Calculate Real Errors
    merged["err_x"] = merged["Est_X"] - merged["Loc_X"]
    merged["err_y"] = merged["Est_Y"] - merged["Loc_Y"]
    merged["pos_error_2d"] = np.hypot(merged["err_x"], merged["err_y"])
    
    # EKF is logged in radians ('Est_Yaw'), GT is logged in degrees ('Yaw_Degrees')
    merged["yaw_error_rad"] = merged.apply(
        lambda row: wrap_angle(row["Est_Yaw"] - np.radians(row["Yaw_Degrees"])), axis=1
    )
    merged["yaw_error_deg"] = np.degrees(merged["yaw_error_rad"])

    rmse_pos = np.sqrt(np.mean(merged["pos_error_2d"]**2))
    rmse_yaw = np.sqrt(np.mean(merged["yaw_error_deg"]**2))

    print("\n" + "="*50)
    print("      TRUE EKF VS GROUND TRUTH METRICS")
    print("="*50)
    print(f"Data points matched : {len(merged)}")
    print(f"Position RMSE (2D)  : {rmse_pos:.4f} meters")
    print(f"Yaw RMSE            : {rmse_yaw:.4f} degrees")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()