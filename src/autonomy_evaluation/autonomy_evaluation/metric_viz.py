#!/usr/bin/env python3
import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy import signal

NUM_CLASSES = 30
CAM_CAPS = {
    'front_left': 250.0,
    'rear': 150.0,
    'side_left': 80.0,
    'side_right': 80.0
}
EARTH_RADIUS = 6378137.0

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def wrap_deg(a_rad): 
    return (np.degrees(a_rad) + 360.0) % 360.0

def quaternion_to_euler(w, x, y, z):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw, pitch, roll

def plot_wrapped_line(ax, x, y, **kwargs):
    x, y = np.array(x), np.array(y)
    dy = np.abs(np.diff(y))
    splits = np.where(dy > 180)[0] + 1
    x_split = np.insert(x, splits, np.nan)
    y_split = np.insert(y, splits, np.nan)
    ax.plot(x_split, y_split, **kwargs)

def angle_error(est, gt):
    return wrap_angle(est - gt)

def safe_float(x):
    return float(x) if np.isfinite(x) else 0.0

def load_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f'Warning: failed to read {path}: {e}')
        return pd.DataFrame()

def save_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def compute_perception_metrics(run_dir):
    perception_dir = os.path.join(run_dir, 'perception')
    results = {}
    manifest_path = os.path.join(perception_dir, 'manifest.csv')
    manifest = load_csv(manifest_path)
    for cam in CAM_CAPS:
        results[cam] = {'mIoU': 0.0, 'RMSE': 0.0, 'delta_1_25': 0.0, 'max_depth_cap': CAM_CAPS[cam], 'frames_evaluated': 0}
    if manifest.empty:
        return results
    
    grouped = {cam: [] for cam in CAM_CAPS}
    for _, row in manifest.iterrows():
        cam = str(row['Camera'])
        rel_path = str(row['File'])
        if cam in grouped:
            grouped[cam].append(os.path.join(run_dir, rel_path))
            
    for cam in CAM_CAPS:
        rmse_values, d1_values, miou_values = [], [], []
        for path in grouped[cam]:
            if not os.path.exists(path): continue
            try:
                data = np.load(path)
                pred_d, gt_d = data['pred_depth'].astype(np.float32), data['gt_depth'].astype(np.float32)
                pred_s, gt_s = data['pred_seg'].astype(np.uint8), data['gt_seg'].astype(np.uint8)
                
                if pred_d.shape != gt_d.shape or pred_s.shape != gt_s.shape: continue
                
                max_depth = CAM_CAPS[cam]
                mask = (gt_d >= 1.0) & (gt_d <= max_depth) & (pred_d >= 1.0) & (pred_d <= max_depth)
                if np.any(mask):
                    p, g = pred_d[mask], gt_d[mask]
                    rmse_values.append(float(np.sqrt(np.mean((p - g) ** 2))))
                    ratio = np.maximum(p / g, g / p)
                    d1_values.append(float((ratio < 1.25).mean() * 100.0))
                
                valid_s = (gt_s != 255) & (gt_s < NUM_CLASSES) & (pred_s < NUM_CLASSES)
                if np.any(valid_s):
                    gt_valid, pred_valid = gt_s[valid_s].astype(np.int32), pred_s[valid_s].astype(np.int32)
                    cm = np.bincount(NUM_CLASSES * gt_valid + pred_valid, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
                    intersection = np.diag(cm)
                    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
                    ious = [intersection[i] / union[i] for i in range(1, NUM_CLASSES) if union[i] > 0]
                    if ious: miou_values.append(float(np.mean(ious)))
            except Exception:
                continue
                
        results[cam]['mIoU'] = safe_float(np.mean(miou_values)) if miou_values else 0.0
        results[cam]['RMSE'] = safe_float(np.mean(rmse_values)) if rmse_values else 0.0
        results[cam]['delta_1_25'] = safe_float(np.mean(d1_values)) if d1_values else 0.0
        results[cam]['frames_evaluated'] = len(rmse_values)
    return results

def build_localization_dataframe(run_dir):
    aligned_dir = os.path.join(run_dir, 'aligned')
    ekf = load_csv(os.path.join(aligned_dir, 'ekf_aligned.csv'))
    slam = load_csv(os.path.join(aligned_dir, 'slam_aligned.csv'))
    vo = load_csv(os.path.join(aligned_dir, 'vo_aligned.csv'))
    
    if ekf.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    ekf = ekf.drop_duplicates(subset=['GT_Timestamp']).sort_values('GT_Timestamp').reset_index(drop=True)
    merged = ekf.copy()
    
    if not slam.empty:
        slam = slam.sort_values('GT_Timestamp').drop_duplicates('GT_Timestamp').reset_index(drop=True)
        slam_cols = ['GT_Timestamp','SLAM_X','SLAM_Y','SLAM_Z','SLAM_Yaw'] if 'SLAM_X' in slam.columns else []
        if slam_cols:
            merged = pd.merge_asof(merged, slam[slam_cols], on='GT_Timestamp', direction='nearest', tolerance=0.05)
            
    if not vo.empty:
        vo = vo.sort_values('GT_Timestamp').drop_duplicates('GT_Timestamp').reset_index(drop=True)
        vo_cols = ['GT_Timestamp','VO_X','VO_Y','VO_Z'] if 'VO_X' in vo.columns else []
        if vo_cols:
            merged = pd.merge_asof(merged, vo[vo_cols], on='GT_Timestamp', direction='nearest', tolerance=0.05)
            
    return merged.reset_index(drop=True), ekf, slam, vo

def calculate_metrics(merged):
    if merged.empty: return {}, merged
    
    settling_time = 0.0
    min_time = merged['GT_Timestamp'].min()
    merged = merged[merged['GT_Timestamp'] >= min_time + settling_time].copy().reset_index(drop=True)
    if merged.empty: return {}, merged
    
    start_time = merged['GT_Timestamp'].iloc[0]
    merged['Time'] = merged['GT_Timestamp'] - start_time
    
    # Base Position Error
    merged['error_x'] = merged['EKF_X'] - merged['GT_X']
    merged['error_y'] = merged['EKF_Y'] - merged['GT_Y']
    merged['error_z'] = merged['EKF_Z'] - merged['GT_Z']
    merged['pos_error'] = np.sqrt(merged['error_x'] ** 2 + merged['error_y'] ** 2 + merged['error_z'] ** 2)
    
    merged['yaw_error'] = merged.apply(lambda r: angle_error(r['EKF_Yaw'], r['GT_Yaw']), axis=1)
    merged['pitch_error'] = merged.apply(lambda r: angle_error(r['EKF_Pitch'], -r['GT_Pitch']), axis=1)
    merged['roll_error'] = merged.apply(lambda r: angle_error(r['EKF_Roll'], -r['GT_Roll']), axis=1)
    
    # ---------------------------------------------------------
    # NEW CORNER LOGIC (3.5 Degree Threshold)
    # ---------------------------------------------------------
    t_gt = merged['GT_Timestamp'].to_numpy()
    dt_gt = np.diff(t_gt)
    dt_gt[dt_gt <= 0.0] = 0.01
    
    dyaw_rad = np.diff(np.unwrap(merged['GT_Yaw']))
    yaw_rate_deg = np.abs(np.degrees(dyaw_rad) / dt_gt)
    
    # Append 0.0 to pad the size back to N (so it matches GT shape)
    yaw_rate_deg = np.append(yaw_rate_deg, 0.0) 
    merged['is_corner'] = yaw_rate_deg >= 3.5
    corner_mask = merged['is_corner']
    # ---------------------------------------------------------
    
    if 'SLAM_X' in merged.columns:
        valid = merged['SLAM_X'].notna()
        merged.loc[valid, 'slam_pos_error'] = np.sqrt((merged.loc[valid, 'SLAM_X'] - merged.loc[valid, 'GT_X'])**2 + (merged.loc[valid, 'SLAM_Y'] - merged.loc[valid, 'GT_Y'])**2 + (merged.loc[valid, 'SLAM_Z'] - merged.loc[valid, 'GT_Z'])**2)
    if 'VO_X' in merged.columns:
        valid = merged['VO_X'].notna()
        merged.loc[valid, 'vo_pos_error'] = np.sqrt((merged.loc[valid, 'VO_X'] - merged.loc[valid, 'GT_X'])**2 + (merged.loc[valid, 'VO_Y'] - merged.loc[valid, 'GT_Y'])**2 + (merged.loc[valid, 'VO_Z'] - merged.loc[valid, 'GT_Z'])**2)

    t, dt = merged['Time'].to_numpy(), np.where(merged['Time'].diff().fillna(0.01) > 0.01, merged['Time'].diff().fillna(0.01), 0.01)
    speed = merged['GT_Velocity'].astype(float)
    
    if len(speed) >= 21:
        smoothed = speed.rolling(11, center=True, min_periods=1).mean()
        try:
            smoothed = pd.Series(signal.savgol_filter(smoothed.to_numpy(), 21, 3), index=merged.index)
        except Exception:
            smoothed = pd.Series(smoothed, index=merged.index)
    else:
        smoothed = speed.rolling(20, center=True, min_periods=1).mean()
        
    smoothed = smoothed.ffill().bfill()
    merged['Long_Accel'] = smoothed.diff() / dt
    merged['Long_Jerk'] = merged['Long_Accel'].diff() / dt
    merged['Lat_Accel'] = smoothed * np.radians(merged['GT_Yaw'].diff() / dt)
    merged['Lat_Jerk'] = merged['Lat_Accel'].diff() / dt

    metrics = {
        'duration_s': safe_float(merged['Time'].max()), 'samples': int(len(merged)),
        'rmse_pos_3d': safe_float(np.sqrt(np.mean(merged['pos_error'] ** 2))),
        'rmse_x': safe_float(np.sqrt(np.mean(merged['error_x'] ** 2))), 'rmse_y': safe_float(np.sqrt(np.mean(merged['error_y'] ** 2))), 'rmse_z': safe_float(np.sqrt(np.mean(merged['error_z'] ** 2))),
        'max_error_3d': safe_float(merged['pos_error'].max()), 'mean_error_3d': safe_float(merged['pos_error'].mean()),
        'rmse_yaw_deg': safe_float(np.sqrt(np.mean(np.degrees(merged['yaw_error']) ** 2))),
        'rmse_pitch_deg': safe_float(np.sqrt(np.mean(np.degrees(merged['pitch_error']) ** 2))),
        'rmse_roll_deg': safe_float(np.sqrt(np.mean(np.degrees(merged['roll_error']) ** 2))),
        'avg_speed': safe_float(merged['GT_Velocity'].mean()),
        'max_jerk': safe_float(merged['Long_Jerk'].abs().dropna().max()) if merged['Long_Jerk'].notna().any() else 0.0
    }
    
    # Save isolated corner metrics
    metrics['rmse_pos_3d_corner'] = safe_float(np.sqrt(np.mean(merged.loc[corner_mask, 'pos_error'] ** 2))) if corner_mask.any() else 0.0
    metrics['rmse_yaw_deg_corner'] = safe_float(np.sqrt(np.mean(np.degrees(merged.loc[corner_mask, 'yaw_error']) ** 2))) if corner_mask.any() else 0.0
    
    if 'slam_pos_error' in merged.columns and not merged['slam_pos_error'].dropna().empty:
        metrics['slam_rmse_3d'] = safe_float(np.sqrt(np.mean(merged['slam_pos_error'].dropna() ** 2)))
    if 'vo_pos_error' in merged.columns and not merged['vo_pos_error'].dropna().empty:
        metrics['vo_rmse_3d'] = safe_float(np.sqrt(np.mean(merged['vo_pos_error'].dropna() ** 2)))
        
    for col in ['Bias_Gx','Bias_Gy','Bias_Gz']:
        if col not in merged.columns: merged[col] = 0.0
    metrics['final_bias_gx'] = safe_float(merged['Bias_Gx'].iloc[-1])
    metrics['final_bias_gy'] = safe_float(merged['Bias_Gy'].iloc[-1])
    metrics['final_bias_gz'] = safe_float(merged['Bias_Gz'].iloc[-1])
    
    return metrics, merged

def generate_plots(run_dir, out_dir, merged, ekf_full):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="darkgrid", palette="deep")
    
    gt_origin_x = ekf_full['GT_X'].iloc[0]
    gt_origin_y = ekf_full['GT_Y'].iloc[0]
    gt_origin_z = ekf_full['GT_Z'].iloc[0]
    gt_start_time = ekf_full['GT_Timestamp'].iloc[0]
    
    planning_dir = os.path.join(run_dir, 'planning')
    sensor_dir = os.path.join(run_dir, 'sensors')
    
    global_path = []
    if os.path.exists(os.path.join(planning_dir, 'global_path.json')):
        try:
            with open(os.path.join(planning_dir, 'global_path.json')) as f: global_path = json.load(f)
        except: pass
    local_df = load_csv(os.path.join(planning_dir, 'local_path.csv'))
    target_df = load_csv(os.path.join(planning_dir, 'target_speed.csv'))
    
    imu = load_csv(os.path.join(sensor_dir, 'imu.csv'))
    wheel = load_csv(os.path.join(sensor_dir, 'wheel_odom.csv'))
    gnss = load_csv(os.path.join(sensor_dir, 'gnss.csv'))
    mag = load_csv(os.path.join(sensor_dir, 'magnetometer.csv'))
    alt = load_csv(os.path.join(sensor_dir, 'altimeter.csv'))

    if not imu.empty: imu = imu[imu['Timestamp'] >= gt_start_time].copy()
    if not wheel.empty: wheel = wheel[wheel['Timestamp'] >= gt_start_time].copy()
    if not gnss.empty: gnss = gnss[gnss['Timestamp'] >= gt_start_time].copy()
    if not mag.empty: mag = mag[mag['Timestamp'] >= gt_start_time].copy()
    if not alt.empty: alt = alt[alt['Timestamp'] >= gt_start_time].copy()

    if 'EKF_Vz' in merged.columns and 'EKF_Vy' in merged.columns and 'EKF_Vx' in merged.columns:
        ekf_speed = np.sqrt(merged['EKF_Vx']**2 + merged['EKF_Vy']**2 + merged['EKF_Vz']**2)
    elif 'EKF_Vx' in merged.columns and 'EKF_Vy' in merged.columns:
        ekf_speed = np.sqrt(merged['EKF_Vx']**2 + merged['EKF_Vy']**2)
    else:
        ekf_speed = merged.get('EKF_Vx', pd.Series(np.zeros(len(merged))))

    if 'GT_Vz' in merged.columns and 'GT_Vy' in merged.columns and 'GT_Vx' in merged.columns:
        gt_speed = np.sqrt(merged['GT_Vx']**2 + merged['GT_Vy']**2 + merged['GT_Vz']**2)
    else:
        gt_speed = merged['GT_Velocity']

    is_corner = merged['is_corner'].to_numpy()

    # =========================================================================
    # PLOT 1: Trajectory (Spatial Mapping + Position Error vs Time)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(24, 10)) # Increased Width
    axes[0].plot(merged['GT_X'], merged['GT_Y'], 'k--', linewidth=3, label='Ground Truth', zorder=4)
    axes[0].plot(merged['EKF_X'], merged['EKF_Y'], color='blue', linewidth=3, label='EKF Fused Trajectory', zorder=5)

    offset_x, offset_y = 0.0, 0.0
    gx_corr, gy_corr, gnss_times = None, None, None
    if not gnss.empty:
        lat0, lon0 = gnss['Front_Lat'].iloc[0], gnss['Front_Lon'].iloc[0]
        gx_raw = np.radians(gnss['Front_Lon'] - lon0) * math.cos(math.radians(lat0)) * EARTH_RADIUS
        gy_raw = np.radians(gnss['Front_Lat'] - lat0) * EARTH_RADIUS
        gnss_times = gnss['Timestamp'].to_numpy()
        interp_yaw = np.interp(gnss_times, ekf_full['GT_Timestamp'].to_numpy(), ekf_full['GT_Yaw'].to_numpy())
        interp_yaw_enu = interp_yaw * -1.0
        GNSS_FRONT_OFFSET = 1.11 
        gx_corr = gx_raw - (GNSS_FRONT_OFFSET * np.cos(interp_yaw_enu))
        gy_corr = gy_raw - (GNSS_FRONT_OFFSET * np.sin(interp_yaw_enu))
        offset_x = gt_origin_x - gx_corr.iloc[0] if isinstance(gx_corr, pd.Series) else gt_origin_x - gx_corr[0]
        offset_y = gt_origin_y - gy_corr.iloc[0] if isinstance(gy_corr, pd.Series) else gt_origin_y - gy_corr[0]
        axes[0].scatter(gx_corr + offset_x, gy_corr + offset_y, c='purple', s=25, alpha=0.6, label='Raw GNSS (Offset Corrected)')

    if 'SLAM_X' in merged.columns and merged['SLAM_X'].notna().any():
        axes[0].scatter(merged['SLAM_X'], merged['SLAM_Y'], c='red', s=15, alpha=0.4, label='Raw SLAM')
    if 'VO_X' in merged.columns and merged['VO_X'].notna().any():
        axes[0].scatter(merged['VO_X'], merged['VO_Y'], c='teal', s=25, alpha=0.6, label='Raw VO')

    wheel_times, wheel_err = None, None
    if not wheel.empty:
        w_df = wheel.copy()
        w_df['dt'] = w_df['Timestamp'].diff().fillna(0.0).clip(0.0, 0.1)
        w_gt_yaw = np.interp(w_df['Timestamp'], ekf_full['GT_Timestamp'], ekf_full['GT_Yaw'])
        wheel_dx = (w_df['Odom_Velocity'] * np.cos(w_gt_yaw) * w_df['dt']).cumsum()
        wheel_dy = (w_df['Odom_Velocity'] * np.sin(w_gt_yaw) * w_df['dt']).cumsum()
        w_df['Wheel_X'] = wheel_dx + gt_origin_x
        w_df['Wheel_Y'] = gt_origin_y - wheel_dy 
        axes[0].plot(w_df['Wheel_X'], w_df['Wheel_Y'], color='saddlebrown', linestyle=':', linewidth=2, alpha=0.8, label='Wheel Odom Pos (Isolated)')

        i_gt_x = np.interp(w_df['Timestamp'], merged['GT_Timestamp'], merged['GT_X'])
        i_gt_y = np.interp(w_df['Timestamp'], merged['GT_Timestamp'], merged['GT_Y'])
        wheel_times = w_df['Timestamp'] - gt_start_time
        wheel_err = np.sqrt((w_df['Wheel_X'] - i_gt_x)**2 + (w_df['Wheel_Y'] - i_gt_y)**2)

    axes[0].set_title("Decoupled Sensors and Tuned Trajectory")
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)"); axes[0].legend(loc="best"); axes[0].axis("equal")

    axes[1].plot(merged['Time'], merged['pos_error'], color='blue', linewidth=3, label='EKF Pos Error')
    
    if gnss_times is not None:
        i_gt_x_gnss = np.interp(gnss_times, merged['GT_Timestamp'], merged['GT_X'])
        i_gt_y_gnss = np.interp(gnss_times, merged['GT_Timestamp'], merged['GT_Y'])
        gnss_err = np.sqrt(((gx_corr + offset_x) - i_gt_x_gnss)**2 + ((gy_corr + offset_y) - i_gt_y_gnss)**2)
        axes[1].scatter(gnss_times - gt_start_time, gnss_err, c='purple', s=15, alpha=0.6, label='GNSS Error')

    if 'slam_pos_error' in merged.columns and merged['slam_pos_error'].notna().any():
        axes[1].plot(merged['Time'], merged['slam_pos_error'], color='red', alpha=0.6, label='SLAM Error')
    if 'vo_pos_error' in merged.columns and merged['vo_pos_error'].notna().any():
        axes[1].plot(merged['Time'], merged['vo_pos_error'], color='teal', alpha=0.6, label='VO Error')
    if wheel_times is not None:
        axes[1].plot(wheel_times, wheel_err, color='saddlebrown', linestyle=':', linewidth=2, alpha=0.8, label='Wheel Pos Error (Isolated)')

    axes[1].fill_between(merged['Time'], -5, 40, where=is_corner, color='gold', alpha=0.2, label='Corner Region', zorder=0)
    axes[1].set_title("Position Error vs Time (Decoupled Sensors)")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Position Error (m)")
    axes[1].legend(loc="best"); axes[1].set_ylim(0, 30)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, '1_trajectory_and_error.png'), dpi=150); plt.close()

    # =========================================================================
    # PLOT 2: Altitude (Z-Axis) Tracking and Errors 
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(24, 8)) # Increased Width
    
    axes[0].plot(merged['Time'], merged['GT_Z'], color='black', linestyle='--', linewidth=3, label='Ground Truth Z')
    axes[0].plot(merged['Time'], merged['EKF_Z'], color='blue', linewidth=2, label='EKF Z', alpha=0.8)
    
    if not alt.empty:
        axes[0].scatter(alt['Timestamp'] - gt_start_time, alt['Alt_Z'], color='orange', s=15, alpha=0.6, label='Altimeter Z')
    if 'SLAM_Z' in merged.columns and merged['SLAM_Z'].notna().any():
        axes[0].plot(merged['Time'], merged['SLAM_Z'], color='red', alpha=0.6, label='SLAM Z')
    if 'VO_Z' in merged.columns and merged['VO_Z'].notna().any():
        axes[0].plot(merged['Time'], merged['VO_Z'], color='teal', alpha=0.6, label='VO Z')
    if not gnss.empty:
        gnss_alt_corr = gnss['Front_Alt'] - gnss['Front_Alt'].iloc[0] + gt_origin_z
        axes[0].scatter(gnss['Timestamp'] - gt_start_time, gnss_alt_corr, color='purple', s=15, alpha=0.4, label='GNSS Z (Offset)')
        
    axes[0].set_title('Altitude Tracking (Z-Axis)'); axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Altitude (m)')
    axes[0].legend(loc='best')
    
    axes[1].plot(merged['Time'], merged['EKF_Z'] - merged['GT_Z'], color='blue', linewidth=2, label='EKF Z Error')
    if not alt.empty:
        i_gt_z_alt = np.interp(alt['Timestamp'], merged['GT_Timestamp'], merged['GT_Z'])
        axes[1].scatter(alt['Timestamp'] - gt_start_time, alt['Alt_Z'] - i_gt_z_alt, color='orange', s=15, alpha=0.6, label='Altimeter Error')
    if 'SLAM_Z' in merged.columns and merged['SLAM_Z'].notna().any():
        axes[1].plot(merged['Time'], merged['SLAM_Z'] - merged['GT_Z'], color='red', alpha=0.6, label='SLAM Z Error')
    if 'VO_Z' in merged.columns and merged['VO_Z'].notna().any():
        axes[1].plot(merged['Time'], merged['VO_Z'] - merged['GT_Z'], color='teal', alpha=0.6, label='VO Z Error')
    if not gnss.empty:
        i_gt_z_gnss = np.interp(gnss['Timestamp'], merged['GT_Timestamp'], merged['GT_Z'])
        axes[1].scatter(gnss['Timestamp'] - gt_start_time, gnss_alt_corr - i_gt_z_gnss, color='purple', s=15, alpha=0.4, label='GNSS Z Error')

    axes[1].fill_between(merged['Time'], -15, 15, where=is_corner, color='gold', alpha=0.2, label='Corner')
    axes[1].axhline(0, c='k', ls='--', alpha=0.5)
    axes[1].set_title('Altitude Error vs Ground Truth'); axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Error (m)')
    axes[1].set_ylim(-10, 10); axes[1].legend(loc='best')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, '2_altitude_tracking.png'), dpi=120); plt.close()

    # =========================================================================
    # PLOT 3: Orientation (Yaw, Pitch, Roll) Tracking & Error
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(24, 18)) # Increased Width
    
    # Visual inversion corrections for Plot 3 only
    gt_yaw_plot = merged['GT_Yaw']
    gt_pitch_plot = merged['GT_Pitch']
    gt_roll_plot = merged['GT_Roll']
    
    # ---- ROW 1: YAW ----
    plot_wrapped_line(axes[0,0], merged['Time'], wrap_deg(gt_yaw_plot), color='k', linestyle='--', linewidth=2.5, label='Ground Truth Yaw (Negated)')
    plot_wrapped_line(axes[0,0], merged['Time'], wrap_deg(merged['EKF_Yaw']), color='blue', linewidth=2, label='EKF Yaw')
    
    if not mag.empty:
        m_df = mag.copy()
        def calc_mag_yaw(r):
            raw_yaw_rad = math.atan2(2.0*(r['Qw']*r['Qz']+r['Qx']*r['Qy']), 1.0-2.0*(r['Qy']**2+r['Qz']**2))
            raw_deg = math.degrees(raw_yaw_rad) % 360.0
            corr_deg = ((360.0 - raw_deg) - 45.0) % 360.0
            return wrap_angle(-1.0 * math.radians(corr_deg))
        m_df['Compass'] = m_df.apply(calc_mag_yaw, axis=1)
        axes[0,0].scatter(m_df["Timestamp"] - gt_start_time, wrap_deg(m_df['Compass']), marker='x', color='orange', s=20, alpha=0.8, label='Mag Yaw (Calibrated)')
        
        i_gt_yaw_mag = np.interp(m_df["Timestamp"], merged['GT_Timestamp'], gt_yaw_plot)
        mag_err = (np.degrees(m_df['Compass'] - i_gt_yaw_mag) + 180.0) % 360.0 - 180.0
        axes[0,1].scatter(m_df["Timestamp"] - gt_start_time, mag_err, marker='x', color='orange', s=20, alpha=0.6, label='Mag Error')

    if not gnss.empty and 'Rear_Lat' in gnss.columns:
        gnss_df = gnss.copy()
        fx_rad = np.radians(gnss_df['Front_Lon']) * np.cos(np.radians(gnss_df['Front_Lat']))
        rx_rad = np.radians(gnss_df['Rear_Lon']) * np.cos(np.radians(gnss_df['Rear_Lat']))
        fy_rad = np.radians(gnss_df['Front_Lat'])
        ry_rad = np.radians(gnss_df['Rear_Lat'])
        gnss_raw = np.arctan2(fy_rad - ry_rad, fx_rad - rx_rad)
        gnss_yaw = wrap_deg(-1.0 * gnss_raw)
        axes[0,0].scatter(gnss_df["Timestamp"] - gt_start_time, gnss_yaw, marker='D', color='magenta', s=25, alpha=0.8, label='GNSS Dual Yaw')
        
        i_gt_yaw_gnss = np.interp(gnss_df["Timestamp"], merged['GT_Timestamp'], gt_yaw_plot)
        gnss_y_err = (np.degrees((-1.0 * gnss_raw) - i_gt_yaw_gnss) + 180.0) % 360.0 - 180.0
        axes[0,1].scatter(gnss_df["Timestamp"] - gt_start_time, gnss_y_err, marker='D', color='magenta', s=25, alpha=0.6, label='GNSS Error')

    if 'SLAM_Yaw' in merged.columns and merged['SLAM_Yaw'].notna().any():
        plot_wrapped_line(axes[0,0], merged['Time'], wrap_deg(merged['SLAM_Yaw']), color='red', alpha=0.6, label='SLAM Yaw')
        slam_yaw_err = merged.apply(lambda r: angle_error(r['SLAM_Yaw'], -r['GT_Yaw']) if pd.notna(r['SLAM_Yaw']) else np.nan, axis=1)
        plot_wrapped_line(axes[0,1], merged['Time'], np.degrees(slam_yaw_err), color='red', alpha=0.6, label='SLAM Yaw Error')

    if not imu.empty:
        imu_df = imu.copy()
        imu_yaws, imu_pitches, imu_rolls = [], [], []
        for _, r in imu_df.iterrows():
            y, p, ro = quaternion_to_euler(r.get('Qw',1), r.get('Qx',0), r.get('Qy',0), r.get('Qz',0))
            # FIX: Adding 45-degree offset to IMU Yaw due to ENU mismatch
            imu_yaws.append(y + math.radians(45.0))
            imu_pitches.append(p); imu_rolls.append(ro)
            
        imu_df['IMU_Yaw'], imu_df['IMU_Pitch'], imu_df['IMU_Roll'] = imu_yaws, imu_pitches, imu_rolls
        
        axes[0,0].scatter(imu_df["Timestamp"] - gt_start_time, wrap_deg(np.array(imu_df['IMU_Yaw'])), marker='.', color='gray', s=5, alpha=0.4, label='Raw IMU Yaw (+45deg)')
        i_gt_yaw_imu = np.interp(imu_df["Timestamp"], merged['GT_Timestamp'], gt_yaw_plot)
        imu_y_err = (np.degrees(np.array(imu_df['IMU_Yaw']) - i_gt_yaw_imu) + 180.0) % 360.0 - 180.0
        axes[0,1].scatter(imu_df["Timestamp"] - gt_start_time, imu_y_err, marker='.', color='gray', s=5, alpha=0.4, label='IMU Yaw Error')

    if not wheel.empty:
        w_df_yaw = wheel.copy()
        w_dt = w_df_yaw['Timestamp'].diff().fillna(0.01)
        
        # FIX: Multiplied Integrated Wheel Yaw by -1.0
        integrated_wheel_yaw = -1.0 * np.cumsum(w_df_yaw['Steering_Angle'] * w_dt) + ekf_full['GT_Yaw'].iloc[0]
        axes[0,0].plot(w_df_yaw["Timestamp"] - gt_start_time, wrap_deg(integrated_wheel_yaw), color='saddlebrown', linestyle=':', label='Wheel Integrated Yaw (* -1)')
        
        i_gt_yaw_wheel = np.interp(w_df_yaw["Timestamp"], merged['GT_Timestamp'], gt_yaw_plot)
        w_yaw_err = (np.degrees(integrated_wheel_yaw - i_gt_yaw_wheel) + 180.0) % 360.0 - 180.0
        plot_wrapped_line(axes[0,1], w_df_yaw["Timestamp"] - gt_start_time, w_yaw_err, color='saddlebrown', linestyle=':', label='Wheel Yaw Error')

    axes[0,0].set_title("Yaw Tracking (0 - 360 Degrees)"); axes[0,0].set_ylabel("Yaw (Degrees)")
    axes[0,0].set_yticks(np.arange(0, 361, 90)); axes[0,0].set_ylim(-10, 370); axes[0,0].legend(loc="best")
    
    # Align EKF Error plot against the visually negated GT Yaw
    ekf_yaw_err_plot = merged.apply(lambda r: angle_error(r['EKF_Yaw'], -r['GT_Yaw']), axis=1)
    plot_wrapped_line(axes[0,1], merged['Time'], np.degrees(ekf_yaw_err_plot), color='blue', linewidth=2, label='EKF Error')
    axes[0,1].fill_between(merged['Time'], -180, 180, where=is_corner, color='gold', alpha=0.2, label='Corner Region', zorder=0)
    axes[0,1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0,1].set_title("Yaw Error vs Ground Truth"); axes[0,1].set_ylabel("Error (Degrees)"); axes[0,1].legend(loc="best")

    # ---- ROW 2: PITCH ----
    axes[1,0].plot(merged['Time'], np.degrees(gt_pitch_plot), 'k--', linewidth=2.5, label='Ground Truth Pitch (Negated)')
    axes[1,0].plot(merged['Time'], np.degrees(merged['EKF_Pitch']), 'blue', linewidth=2, label='EKF Pitch')
    if not imu.empty:
        axes[1,0].scatter(imu_df["Timestamp"] - gt_start_time, np.degrees(imu_df['IMU_Pitch']), marker='.', color='gray', s=5, alpha=0.4, label='Raw IMU Pitch')
    axes[1,0].set_title("Pitch Tracking"); axes[1,0].set_ylabel("Pitch (Degrees)"); axes[1,0].legend(loc="best")

    axes[1,1].plot(merged['Time'], np.degrees(merged['pitch_error']), color='blue', linewidth=2, label='EKF Pitch Error')
    axes[1,1].fill_between(merged['Time'], -90, 90, where=is_corner, color='gold', alpha=0.2, label='Corner Region', zorder=0)
    axes[1,1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1,1].set_title("Pitch Error"); axes[1,1].set_ylabel("Error (Degrees)"); axes[1,1].legend(loc="best")

    # ---- ROW 3: ROLL ----
    axes[2,0].plot(merged['Time'], np.degrees(gt_roll_plot), 'k--', linewidth=2.5, label='Ground Truth Roll (Negated)')
    axes[2,0].plot(merged['Time'], np.degrees(merged['EKF_Roll']), 'blue', linewidth=2, label='EKF Roll')
    if not imu.empty:
        axes[2,0].scatter(imu_df["Timestamp"] - gt_start_time, np.degrees(imu_df['IMU_Roll']), marker='.', color='gray', s=5, alpha=0.4, label='Raw IMU Roll')
    axes[2,0].set_title("Roll Tracking"); axes[2,0].set_xlabel("Time (s)"); axes[2,0].set_ylabel("Roll (Degrees)"); axes[2,0].legend(loc="best")

    axes[2,1].plot(merged['Time'], np.degrees(merged['roll_error']), color='blue', linewidth=2, label='EKF Roll Error')
    axes[2,1].fill_between(merged['Time'], -90, 90, where=is_corner, color='gold', alpha=0.2, label='Corner Region', zorder=0)
    axes[2,1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[2,1].set_title("Roll Error"); axes[2,1].set_xlabel("Time (s)"); axes[2,1].set_ylabel("Error (Degrees)"); axes[2,1].legend(loc="best")

    plt.tight_layout(); plt.savefig(os.path.join(out_dir, '3_orientation_analysis.png'), dpi=150); plt.close()

    # PLOT 4: Gyro Bias
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for idx, (col, color, title) in enumerate([('Bias_Gx', '#9467bd', 'Gyro Bias X'), ('Bias_Gy', '#e377c2', 'Gyro Bias Y'), ('Bias_Gz', '#17becf', 'Gyro Bias Z')]):
        if col in merged.columns:
            axes[idx].plot(merged['Time'], merged[col], color=color, linewidth=2, alpha=0.8)
            axes[idx].axhline(0, color='black', linestyle='--', alpha=0.5)
            axes[idx].set_title(title); axes[idx].set_xlabel('Time (s)'); axes[idx].set_ylabel('Bias (rad/s)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, '4_gyro_bias_convergence.png'), dpi=120); plt.close()

    # PLOT 5: Jerk
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax1.plot(merged['Time'], merged['Long_Jerk'], color='#ff7f0e', alpha=0.9, linewidth=2)
    ax1.axhline(3, color='green', linestyle='--', alpha=0.6, label='Comfort Limit')
    ax1.axhline(-3, color='green', linestyle='--', alpha=0.6)
    ax1.set_title('Longitudinal Jerk Over Time'); ax1.set_ylim(-15, 15); ax1.legend()
    h = ax2.hist2d(merged['Lat_Jerk'].fillna(0).clip(-10, 10), merged['Long_Jerk'].fillna(0).clip(-10, 10), bins=40, cmap='mako', range=[[-10,10],[-10,10]])
    fig.colorbar(h[3], ax=ax2, label='Frequency')
    ax2.set_title('2D Jerk Heatmap'); plt.tight_layout(); plt.savefig(os.path.join(out_dir, '5_jerk_analysis.png'), dpi=120); plt.close()

    # PLOT 6: Covariance
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    scatter = ax1.scatter(merged['EKF_X'], merged['EKF_Y'], c=merged['pos_error'], cmap='flare_r', s=30, zorder=3)
    ax1.plot(merged['GT_X'], merged['GT_Y'], 'k--', alpha=0.4, label='Ground Truth', zorder=2)
    fig.colorbar(scatter, ax=ax1, label='3D Position Error (m)')
    ax1.set_title('EKF 2D Error Map'); ax1.set_aspect('equal', adjustable='datalim')
    
    ax2.plot(merged['EKF_X'], merged['EKF_Y'], color=(237/255, 59/255, 178/255), label='Estimated Path', linewidth=2, zorder=2)
    for _, row in merged.iloc[::30].iterrows():
        if row.get('P_X', 0) > 0 and row.get('P_Y', 0) > 0:
            ellipse = Ellipse((row['EKF_X'], row['EKF_Y']), 4*np.sqrt(row['P_X']), 4*np.sqrt(row['P_Y']), angle=0, alpha=0.3, color="#098931", zorder=1)
            ax2.add_patch(ellipse)
    ax2.set_title('Position Covariance Ellipses (2σ)'); ax2.set_aspect('equal', adjustable='datalim')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, '6_spatial_error_and_covariance.png'), dpi=120); plt.close()

    # =========================================================================
    # PLOT 7: Speed Profile Tracking and Error
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(24, 8)) # Increased Width
    
    axes[0].plot(merged['Time'], gt_speed, color='black', linestyle='--', linewidth=3, label='GT Speed')
    axes[0].plot(merged['Time'], ekf_speed, color='blue', linewidth=2, label='EKF Speed', alpha=0.9)
    if not wheel.empty:
        axes[0].plot(wheel['Timestamp'] - gt_start_time, wheel['Odom_Velocity'], color='brown', linestyle=':', linewidth=2, alpha=0.8, label='Wheel Speed')
    if not target_df.empty:
        target_times = target_df['Timestamp'] - gt_start_time
        valid_targets = target_times >= 0
        axes[0].plot(target_times[valid_targets], target_df['Target_Speed'][valid_targets], color='purple', linestyle=':', linewidth=2, label='VGP Target Speed', alpha=0.9)
        
    axes[0].fill_between(merged['Time'], ekf_speed, gt_speed, alpha=0.2, color='gray')
    axes[0].set_title('Speed Profile Tracking'); axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Speed (m/s)')
    axes[0].legend(loc='best')
    
    axes[1].plot(merged['Time'], ekf_speed - gt_speed, color='blue', linewidth=2, label='EKF Speed Error')
    if not wheel.empty:
        i_gt_speed_w = np.interp(wheel['Timestamp'], merged['GT_Timestamp'], gt_speed)
        axes[1].plot(wheel['Timestamp'] - gt_start_time, wheel['Odom_Velocity'] - i_gt_speed_w, color='brown', linestyle=':', linewidth=2, alpha=0.8, label='Wheel Speed Error')
        
    axes[1].fill_between(merged['Time'], -10, 10, where=is_corner, color='gold', alpha=0.2, label='Corner Region')
    axes[1].axhline(0, c='k', ls='--', alpha=0.5)
    axes[1].set_title('Speed Error vs Ground Truth'); axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Error (m/s)')
    axes[1].legend(loc='best')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, '7_speed_tracking.png'), dpi=120); plt.close()

    # PLOT 8: Planning
    fig8 = plt.figure(figsize=(14, 10)); ax8 = fig8.add_subplot(111)
    ax8.plot(merged['GT_X'], merged['GT_Y'], color='black', linestyle='--', label='Ground Truth Executed', linewidth=2, zorder=3)
    ax8.plot(merged['EKF_X'], merged['EKF_Y'], color='#1f77b4', linestyle='-', label='EKF Executed', linewidth=2, zorder=4)
    if global_path:
        gp = pd.DataFrame(global_path)
        if not gp.empty and 'X' in gp.columns: ax8.plot(gp['X'], gp['Y'], color='purple', linestyle='-', linewidth=4, label='Mission Global Path', alpha=0.5, zorder=2)
    
    if not local_df.empty and 'Path_JSON' in local_df.columns:
        for idx, row in local_df.iloc[::25].iterrows():
            try:
                path = json.loads(row['Path_JSON'])
                t_norm = row['Timestamp'] - merged['GT_Timestamp'].iloc[0]
                if t_norm < 0: continue
                ekf_state = merged.iloc[(merged['Time'] - t_norm).abs().argmin()]
                ex, ey, eyaw = ekf_state['EKF_X'], ekf_state['EKF_Y'], ekf_state['EKF_Yaw']
                gx_list, gy_list = [], []
                for p in path:
                    gx = ex + p['X'] * math.cos(eyaw) + p['Y'] * math.sin(eyaw)
                    gy = ey + p['X'] * math.sin(eyaw) - p['Y'] * math.cos(eyaw)
                    gx_list.append(gx); gy_list.append(gy)
                ax8.plot(gx_list, gy_list, color='orange', alpha=0.6, linewidth=1.5, zorder=5)
            except: pass
        handles, labels = ax8.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='orange', alpha=0.6, lw=1.5))
        labels.append('Local Lattice Plan Traces')
        ax8.legend(handles=handles, labels=labels, loc='best')
    else:
        ax8.legend(loc='best')
    ax8.set_title('Path Planning & Execution Comparison')
    ax8.set_xlabel('X (m)'); ax8.set_ylabel('Y (m)'); ax8.set_aspect('equal', adjustable='box')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, '8_path_planning_comparison.png'), dpi=120); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='eval_temp')
    parser.add_argument('--debug', action='store_true', help='Include and calculate perception metrics')
    args = parser.parse_args()
    
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(os.path.join(run_dir, '..', 'eval_plots'))
    
    print(f'Reading evaluation data from: {run_dir}')
    print('Calculating localization metrics...')
    
    merged, ekf, slam, vo = build_localization_dataframe(run_dir)
    if merged.empty:
        print('ERROR: No aligned EKF data found.')
        return 1
    
    metrics, merged = calculate_metrics(merged)
    save_json(os.path.join(out_dir, 'ekf_metrics.json'), metrics)
    
    perception = {}
    if args.debug:
        print('Calculating perception metrics...')
        perception = compute_perception_metrics(run_dir)
        save_json(os.path.join(run_dir, 'perception_metrics.json'), perception)
        
    print('\n========== FINAL EVALUATION ==========')
    print(f"Duration              : {metrics.get('duration_s', 0.0):.2f} s")
    print(f"Samples               : {metrics.get('samples', 0)}")
    print(f"3D Position RMSE      : {metrics.get('rmse_pos_3d', 0.0):.3f} m")
    
    # NEW PRINT STATEMENTS FOR CORNER ISOLATION
    if metrics.get('rmse_pos_3d_corner', 0.0) > 0:
        print(f"Corner 3D Pos RMSE    : {metrics.get('rmse_pos_3d_corner', 0.0):.3f} m")
        print(f"Corner Yaw RMSE       : {metrics.get('rmse_yaw_deg_corner', 0.0):.3f} deg")
        
    print(f"X RMSE                : {metrics.get('rmse_x', 0.0):.3f} m")
    print(f"Y RMSE                : {metrics.get('rmse_y', 0.0):.3f} m")
    print(f"Z RMSE                : {metrics.get('rmse_z', 0.0):.3f} m")
    print(f"Mean 3D Error         : {metrics.get('mean_error_3d', 0.0):.3f} m")
    print(f"Max 3D Error          : {metrics.get('max_error_3d', 0.0):.3f} m")
    print(f"Yaw RMSE              : {metrics.get('rmse_yaw_deg', 0.0):.3f} deg")
    print(f"Pitch RMSE            : {metrics.get('rmse_pitch_deg', 0.0):.3f} deg")
    print(f"Roll RMSE             : {metrics.get('rmse_roll_deg', 0.0):.3f} deg")
    print(f"Average Speed         : {metrics.get('avg_speed', 0.0):.3f} m/s")
    print(f"Max Longitudinal Jerk : {metrics.get('max_jerk', 0.0):.3f} m/s^3")
    if 'slam_rmse_3d' in metrics: print(f"SLAM 3D RMSE          : {metrics['slam_rmse_3d']:.3f} m")
    if 'vo_rmse_3d' in metrics: print(f"VO 3D RMSE            : {metrics['vo_rmse_3d']:.3f} m")
    
    if args.debug:
        print('\n---------- PERCEPTION ----------')
        print(f"{'CAMERA':<15} | {'mIoU':<10} | {'RMSE(m)':<10} | {'d1.25(%)':<10} | {'Frames':<8}")
        print('-' * 70)
        for cam, data in perception.items():
            print(f"{cam:<15} | {data['mIoU']:<10.4f} | {data['RMSE']:<10.2f} | {data['delta_1_25']:<10.2f} | {data['frames_evaluated']:<8}")
        print('=======================================\n')

    print('Numerical metrics are saved.')
    print('Generating plots...')
    generate_plots(run_dir, out_dir, merged, ekf)
    print(f'Plots saved to: {out_dir}')
    print('Evaluation complete.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())