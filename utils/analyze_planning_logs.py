#!/usr/bin/env python3
"""
analyze_planning_logs.py

Offline comparison of GT (CARLA-autopilot-driven) trajectory against what your
planning stack WOULD have chosen, using the CSV logs added to:
  - lattice_planner_node.cpp  -> /tmp/gt_trajectory_log.csv (from odom_callback)
                              -> /tmp/lattice_planner_log.csv
  - mission_planner_node.py   -> /tmp/mission_planner_log.csv

Usage:
    python analyze_planning_logs.py \
        --gt /tmp/gt_trajectory_log.csv \
        --lattice /tmp/lattice_planner_log.csv \
        --mission /tmp/mission_planner_log.csv \
        --outdir ./planning_report
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path, required_cols):
    if not os.path.exists(path):
        print(f"[!] Missing log file: {path} -- skipping that part of the analysis.")
        return None
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[!] {path} is missing expected columns {missing} -- "
              f"check the CSV header matches what was added to the node. Skipping.")
        return None
    if len(df) == 0:
        print(f"[!] {path} is empty -- did the run actually produce data?")
        return None
    return df.sort_values("t_sec").reset_index(drop=True)


def nearest_gt_distance(gt_t, gt_x, gt_y, query_t, query_x, query_y,
                         fwd_window=(0.0, 20.0)):
    """For one (query_t, query_x, query_y), find the closest GT point in time
    window [query_t + fwd_window[0], query_t + fwd_window[1]] and return the
    spatial distance to it. This answers: 'did the car ever actually get near
    where the planner wanted it, shortly after this decision was made?'"""
    mask = (gt_t >= query_t + fwd_window[0]) & (gt_t <= query_t + fwd_window[1])
    if not np.any(mask):
        return np.nan
    dx = gt_x[mask] - query_x
    dy = gt_y[mask] - query_y
    d = np.hypot(dx, dy)
    return float(np.min(d))


def estimate_gt_yaw_rate(gt_df):
    t = gt_df["t_sec"].values
    yaw = np.unwrap(gt_df["yaw"].values)
    dt = np.diff(t)
    dt[dt < 1e-3] = 1e-3
    dyaw = np.diff(yaw) / dt
    yaw_rate = np.zeros(len(t))
    yaw_rate[1:] = dyaw
    return np.abs(yaw_rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="/tmp/gt_trajectory_log.csv")
    ap.add_argument("--lattice", default="/tmp/lattice_planner_log.csv")
    ap.add_argument("--mission", default="/tmp/mission_planner_log.csv")
    ap.add_argument("--outdir", default="./planning_report")
    ap.add_argument("--turn-yaw-rate-thresh", type=float, default=0.08,
                     help="rad/s above which a GT moment is considered 'turning'")
    ap.add_argument("--target-dist", type=float, default=12.0,
                     help="must match target_dist_ in lattice_planner_node.cpp")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    gt = load_csv(args.gt, ["t_sec", "x", "y", "yaw"])
    lat = load_csv(args.lattice, ["t_sec", "ego_x", "ego_y", "ego_yaw", "tgt_offset",
                                   "chosen_offset", "chosen_cost", "blocked_count",
                                   "straight_blocked", "wp_x", "wp_y"])
    mis = load_csv(args.mission, ["t_sec", "wp_index", "x", "y"])

    if gt is None:
        print("Can't do anything without the GT trajectory log. Aborting.")
        sys.exit(1)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PLANNING-VS-GROUND-TRUTH ANALYSIS")
    report_lines.append("=" * 70)
    report_lines.append(f"GT samples: {len(gt)}  (span {gt['t_sec'].min():.1f}s - {gt['t_sec'].max():.1f}s)")

    gt_t = gt["t_sec"].values
    gt_x = gt["x"].values
    gt_y = gt["y"].values
    gt_yaw_rate = estimate_gt_yaw_rate(gt)

    fig_rows = 1 + (lat is not None) + (mis is not None)
    fig, axes = plt.subplots(fig_rows, 1, figsize=(11, 4 * fig_rows))
    if fig_rows == 1:
        axes = [axes]
    ax_i = 0

    # ---------------------------------------------------------------
    # LATTICE PLANNER vs GT
    # ---------------------------------------------------------------
    if lat is not None:
        # [FIX]: Ignore the first 2.5 seconds to allow EKF and Planners to initialize, 
        # preventing massive startup errors from skewing the mean/RMSE.
        lat_start = lat["t_sec"].min()
        lat = lat[lat["t_sec"] >= lat_start + 2.5].copy()

        report_lines.append("")
        report_lines.append("-" * 70)
        report_lines.append("LATTICE PLANNER")
        report_lines.append("-" * 70)
        report_lines.append(f"Cycles logged (after 2.5s warmup crop): {len(lat)}")

        # World-frame predicted waypoint from the ego-relative (base_link) wp_x, wp_y
        # ego_yaw from lattice log is in EKF negated convention, correct before applying rotation
        cos_y = np.cos(lat["ego_yaw"].values)   # cos is even, unchanged
        sin_y = -np.sin(lat["ego_yaw"].values)  # negate to get sin(true_yaw)
        wp_world_x = lat["ego_x"].values + lat["wp_x"].values * cos_y - lat["wp_y"].values * sin_y
        wp_world_y = lat["ego_y"].values + lat["wp_x"].values * sin_y + lat["wp_y"].values * cos_y
        lat = lat.assign(wp_world_x=wp_world_x, wp_world_y=wp_world_y)

        errors = np.full(len(lat), np.nan)
        
        match_window_end = 20.0 
        gt_t_max = gt_t.max()
        for i, row in lat.iterrows():
            if row["t_sec"] > gt_t_max - match_window_end:
                continue
            errors[i] = nearest_gt_distance(gt_t, gt_x, gt_y, row["t_sec"],
                                             row["wp_world_x"], row["wp_world_y"],
                                             fwd_window=(0.0, match_window_end))
        lat = lat.assign(gt_match_error=errors)

        gt_idx = np.searchsorted(gt_t, lat["t_sec"].values)
        gt_idx = np.clip(gt_idx, 0, len(gt_t) - 1)
        lat = lat.assign(local_yaw_rate=gt_yaw_rate[gt_idx])
        lat = lat.assign(is_turning=lat["local_yaw_rate"] > args.turn_yaw_rate_thresh)

        valid = lat.dropna(subset=["gt_match_error"])
        if len(valid) > 0:
            rmse_lat = np.sqrt(np.mean(valid['gt_match_error']**2))
            report_lines.append(f"Cross-track error (planned waypoint vs nearest actual GT position within {match_window_end}s):")
            report_lines.append(f"  RMSE={rmse_lat:.2f}m  "
                                 f"mean={valid['gt_match_error'].mean():.2f}m  "
                                 f"median={valid['gt_match_error'].median():.2f}m  "
                                 f"p90={valid['gt_match_error'].quantile(0.9):.2f}m  "
                                 f"max={valid['gt_match_error'].max():.2f}m")

            straight = valid[~valid["is_turning"]]
            turning = valid[valid["is_turning"]]
            if len(straight) > 0:
                report_lines.append(f"  on STRAIGHT segments: mean error={straight['gt_match_error'].mean():.2f}m "
                                     f"(n={len(straight)})")
            if len(turning) > 0:
                report_lines.append(f"  on TURNING segments:  mean error={turning['gt_match_error'].mean():.2f}m "
                                     f"(n={len(turning)})")
                if len(straight) > 0 and turning['gt_match_error'].mean() > 1.5 * straight['gt_match_error'].mean():
                    report_lines.append("  >>> Error concentrates on turns. Points at the turn-handling "
                                         "chain (curvature gating / lane-centering suppression / lattice "
                                         "reference spline at the intersection), not general lane-keeping.")
        else:
            report_lines.append("  No lattice rows had a matching GT point within the forward window -- "
                                 "check that t_sec uses the same clock in both logs (sim time vs wall time).")

        # straight-candidate false-block check
        sb = lat[lat["straight_blocked"] == True]
        report_lines.append("")
        report_lines.append(f"Straight (delta=0) candidate was blocked in {len(sb)}/{len(lat)} "
                             f"cycles ({100.0*len(sb)/max(len(lat),1):.1f}%).")
        if len(sb) > 0:
            sb_valid = sb.dropna(subset=["gt_match_error"]) if "gt_match_error" in sb else sb
            false_positive_like = 0
            for _, row in sb.iterrows():
                straight_x = row["ego_x"] + args.target_dist * np.cos(row["ego_yaw"])
                straight_y = row["ego_y"] + args.target_dist * np.sin(row["ego_yaw"])
                
                d = nearest_gt_distance(gt_t, gt_x, gt_y, row["t_sec"], straight_x, straight_y,
                                         fwd_window=(0.5, 20.0))
                if not np.isnan(d) and d < 2.5:
                    false_positive_like += 1
            if len(sb) > 0:
                pct = 100.0 * false_positive_like / len(sb)
                report_lines.append(f"  Of those, {false_positive_like}/{len(sb)} ({pct:.0f}%) were cases "
                                     f"where GT later drove within 2.5m of the spot the straight candidate "
                                     f"was aiming for --")
                report_lines.append(f"  i.e. nothing was actually there. If this percentage is "
                                     f"high, suspect a stale/misaligned /slam/local_costmap (timing or frame "
                                     f"mismatch between when the grid was captured and the current ego pose), "
                                     f"not a real obstacle-avoidance failure.")

        worst = lat.dropna(subset=["gt_match_error"]).sort_values("gt_match_error", ascending=False).head(25)
        worst_path = os.path.join(args.outdir, "worst_offenders.csv")
        worst.to_csv(worst_path, index=False)
        report_lines.append("")
        report_lines.append(f"Top 25 worst cycles written to {worst_path} -- use their t_sec values to "
                             f"scrub directly to those moments in your recorded run.")

        ax = axes[ax_i]; ax_i += 1
        ax.plot(lat["t_sec"], lat["gt_match_error"], '.', ms=3, label="cross-track error (m)")
        blocked_t = lat[lat["straight_blocked"] == True]["t_sec"]
        if len(blocked_t) > 0:
            ax.scatter(blocked_t, np.zeros(len(blocked_t)), marker='x', color='red',
                       label="straight candidate blocked", zorder=5)
        ax.set_xlabel("t_sec"); ax.set_ylabel("error (m)")
        ax.set_title("Lattice planner: cross-track error to GT, and false-block markers")
        ax.legend(loc="upper right")

    # ---------------------------------------------------------------
    # MISSION PLANNER vs GT
    # ---------------------------------------------------------------
    if mis is not None:
        report_lines.append("")
        report_lines.append("-" * 70)
        report_lines.append("MISSION PLANNER (global route)")
        report_lines.append("-" * 70)

        # [FIX]: To avoid the "12m init offset", evaluate from the GT trajectory
        # looking for the nearest mission waypoint. This naturally ignores parts 
        # of the global path that the ego hasn't reached yet or started behind.
        wp_errors = []
        mis_x = mis["x"].values
        mis_y = mis["y"].values
        
        for gx, gy in zip(gt_x, gt_y):
            # Find closest mission planner waypoint for this GT point
            d = np.min(np.hypot(mis_x - gx, mis_y - gy))
            wp_errors.append(d)

        if not wp_errors:
            report_lines.append("  No GT data could be compared against the global path.")
        else:
            rmse_mis = np.sqrt(np.mean(np.array(wp_errors)**2))
            report_lines.append(f"Distance from driven GT to nearest global waypoint (N={len(wp_errors)} GT points):")
            report_lines.append(f"  RMSE={rmse_mis:.2f}m  "
                                 f"mean={np.mean(wp_errors):.2f}m  "
                                 f"min={np.min(wp_errors):.2f}m  "
                                 f"max={np.max(wp_errors):.2f}m")

        ax = axes[ax_i]; ax_i += 1
        # [FIX]: Plot mission points smaller and put GT on top (zorder)
        ax.plot(mis["x"], mis["y"], '.', ms=1, color='orange', label="mission planner global path", zorder=1)
        ax.plot(gt_x, gt_y, '-', lw=2, color='black', label="GT (autopilot) trajectory", zorder=5)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal")
        ax.set_title("Top-down: GT vs global path")
        ax.legend(loc="best")

    # always include a raw top-down overlay with lattice waypoints if we have them
    ax = axes[ax_i]; ax_i += 1
    if lat is not None:
        # [FIX]: Plot lattice points smaller and put GT on top (zorder)
        ax.plot(lat["wp_world_x"], lat["wp_world_y"], '.', ms=1, color='magenta',
                label="lattice planned waypoints (world frame)", zorder=1)
    ax.plot(gt_x, gt_y, '-', lw=2, color='black', label="GT (autopilot) trajectory", zorder=5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal")
    ax.set_title("Top-down: GT vs lattice planned waypoints")
    ax.legend(loc="best")

    fig.tight_layout()
    plot_path = os.path.join(args.outdir, "error_over_time.png")
    fig.savefig(plot_path, dpi=130)

    report_lines.append("")
    report_lines.append(f"Plot written to {plot_path}")
    report_lines.append("=" * 70)

    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\n[full report written to {args.outdir}/]")


if __name__ == "__main__":
    main()