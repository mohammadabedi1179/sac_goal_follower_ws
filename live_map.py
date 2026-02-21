#!/usr/bin/env python3
"""
Animate robot trajectories from episode_stats.json.

Shows:
- Robot trajectory
- Robot heading
- Ground-truth obstacle positions
- Final goal position
"""

import argparse
import json
import os
import math
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# Helpers
# ============================================================

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def vec2(p):
    if p is None or not isinstance(p, (list, tuple)) or len(p) < 2:
        return (np.nan, np.nan)
    return (safe_float(p[0]), safe_float(p[1]))


# ============================================================
# Robot Path Loader
# ============================================================

def load_robot_path_xy_yaw(ep: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Supports:
      - ep["robot_path"] = [[x,y,yaw], ...]
      - ep["robot_traj"] = [(x,y), ...]
    """

    rp = ep.get("robot_path", None)
    if rp is None:
        rp = ep.get("robot_traj", None)

    if not isinstance(rp, list) or len(rp) < 2:
        return None

    pts = []
    for p in rp:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue

        x, y = safe_float(p[0]), safe_float(p[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        yaw = safe_float(p[2]) if len(p) >= 3 else np.nan
        pts.append([x, y, yaw])

    if len(pts) < 2:
        return None

    arr = np.asarray(pts, dtype=float)

    # Estimate yaw if missing
    if np.any(~np.isfinite(arr[:, 2])):
        dx = np.diff(arr[:, 0])
        dy = np.diff(arr[:, 1])
        est = np.arctan2(dy, dx)

        yaws = np.full(arr.shape[0], np.nan)
        yaws[1:] = est
        if len(est) > 0:
            yaws[0] = est[0]

        arr[:, 2] = np.where(np.isfinite(arr[:, 2]), arr[:, 2], yaws)

    return arr


# ============================================================
# Ground Truth Obstacles Loader
# ============================================================

def load_gt_obstacles_path_points(ep: Dict[str, Any]) -> np.ndarray:
    gt = ep.get("gt_obstacles_path", None)
    pts = []

    if isinstance(gt, list):
        for frame in gt:
            if not isinstance(frame, dict):
                continue
            for _, v in frame.items():
                if v is None:
                    continue
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    x, y = safe_float(v[0]), safe_float(v[1])
                    if np.isfinite(x) and np.isfinite(y):
                        pts.append([x, y])

    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


# ============================================================
# Utility
# ============================================================

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def set_equal_axes(ax, pts: np.ndarray, padding: float = 1.0):
    if pts.size == 0:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal", adjustable="box")
        return

    xmin, ymin = np.min(pts[:, 0]), np.min(pts[:, 1])
    xmax, ymax = np.max(pts[:, 0]), np.max(pts[:, 1])

    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin, 1.0) + 2 * padding

    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_aspect("equal", adjustable="box")


# ============================================================
# Animation
# ============================================================

def animate_episode(ep: Dict[str, Any], outdir: str, fps=20):

    ep_id = int(safe_float(ep.get("episode"), -1))
    reason = str(ep.get("reason", ""))
    reward = safe_float(ep.get("reward"), 0.0)

    robot_xyz = load_robot_path_xy_yaw(ep)
    if robot_xyz is None:
        print(f"[skip] Episode {ep_id}: no trajectory")
        return

    robot_xy = robot_xyz[:, :2]
    robot_yaw = robot_xyz[:, 2]

    gt_obs_pts = load_gt_obstacles_path_points(ep)
    goal_final = vec2(ep.get("goal_final"))

    fig, ax = plt.subplots()
    ax.set_title(f"Episode {ep_id} | R={reward:.2f} | {reason}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)

    all_pts = [robot_xy]
    if gt_obs_pts.size:
        all_pts.append(gt_obs_pts)
    if np.isfinite(goal_final[0]):
        all_pts.append(np.array([[goal_final[0], goal_final[1]]]))

    set_equal_axes(ax, np.vstack(all_pts))

    # Draw obstacles
    if gt_obs_pts.size:
        ax.scatter(gt_obs_pts[:, 0], gt_obs_pts[:, 1],
                   s=40, alpha=0.6, label="Obstacles")

    # Draw goal
    if np.isfinite(goal_final[0]) and np.isfinite(goal_final[1]):
        ax.scatter([goal_final[0]], [goal_final[1]],
                   s=150, marker="*", label="Goal", zorder=5)

    # Animated elements
    trail_line, = ax.plot([], [], lw=2, label="Robot")
    head_line, = ax.plot([], [], lw=2)

    ax.legend()

    n = robot_xy.shape[0]

    def init():
        trail_line.set_data([], [])
        head_line.set_data([], [])
        return trail_line, head_line

    def update(i):
        trail_line.set_data(robot_xy[:i+1, 0],
                            robot_xy[:i+1, 1])

        x, y = robot_xy[i]
        yaw = robot_yaw[i]

        hx = x + 0.8 * math.cos(yaw)
        hy = y + 0.8 * math.sin(yaw)

        head_line.set_data([x, hx], [y, hy])

        return trail_line, head_line

    anim = FuncAnimation(fig, update, frames=n,
                         init_func=init, blit=True)

    ensure_dir(outdir)
    base = os.path.join(outdir, f"episode_{ep_id:06d}")

    try:
        path = base + ".mp4"
        anim.save(path, fps=fps, dpi=200)
        print(f"[ok] saved {path}")
    except Exception:
        print("[warn] ffmpeg not available, saving GIF instead.")
        path = base + ".gif"
        anim.save(path, writer="pillow", fps=fps)
        print(f"[ok] saved {path}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="anim_out")
    ap.add_argument("--step", type=int, default=100)
    args = ap.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)

    ensure_dir(args.out)

    episodes = sorted(data, key=lambda e: e.get("episode", 0))

    for ep in episodes:
        ep_id = int(safe_float(ep.get("episode"), -1))
        if ep_id % args.step != 0:
            continue
        animate_episode(ep, args.out)


if __name__ == "__main__":
    main()
