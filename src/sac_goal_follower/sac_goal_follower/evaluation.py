#!/usr/bin/env python3
"""
Evaluate a trained SB3 SAC agent on GoalFollowerEnv (ROS2 + Gazebo).

What it does:
- Runs N evaluation episodes (default 20)
- Logs per-step + per-episode metrics
- Produces a deep analysis report (Markdown) + plots + JSON/CSV outputs

Usage example:
  python3 eval_sac_20eps.py \
    --model ./models/sac_goal_follower_best.zip \
    --replay ./models/sac_goal_follower_last_replay.pkl \
    --episodes 20 \
    --outdir ./eval_out \
    --deterministic 1

Notes:
- Requires ROS2 environment running + Gazebo running + your topics/services available.
- Must be executed in the same ROS2 workspace env where detectors_msgs etc are available.
"""

import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import rclpy

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# Import your environment
from goal_env_gt import GoalFollowerEnv


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _mean_ci95(xs: np.ndarray) -> Tuple[float, float]:
    """Return (mean, halfwidth_95CI) using normal approx; fine for quick reporting."""
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return np.nan, np.nan
    m = float(np.mean(xs))
    if xs.size == 1:
        return m, np.nan
    s = float(np.std(xs, ddof=1))
    hw = 1.96 * s / np.sqrt(xs.size)
    return m, hw


@dataclass
class EpisodeResult:
    episode: int
    seed: int

    return_sum: float
    length_steps: int
    duration_sec: float

    terminated: bool
    truncated: bool
    reason: str

    robot_start: Optional[Tuple[float, float]]
    goal_start: Optional[Tuple[float, float]]
    robot_final: Optional[Tuple[float, float]]
    goal_final: Optional[Tuple[float, float]]
    min_dist: Optional[float]

    # Step-derived stats
    v_cmd_mean: float
    v_cmd_max: float
    w_cmd_mean_abs: float

    ttc_min: float
    ttc_mean: float
    ttc_frac_below_2s: float

    min_ultra_min: float
    min_obs_depth_min: float

    progress_total: float
    dist_start: float
    dist_final: float

    # Paths (optional, can be large)
    robot_traj: Optional[List[List[float]]] = None


def run_one_episode(
    env: Monitor,
    model: SAC,
    ep_i: int,
    seed: int,
    deterministic: bool = True,
    max_steps_hard_cap: int = 5000,
) -> Tuple[EpisodeResult, Dict[str, Any]]:
    """
    Runs a single episode and returns:
    - EpisodeResult (summary)
    - raw dict with per-step arrays (for deeper offline analysis)
    """
    obs, _ = env.reset(seed=seed)

    t0 = time.time()

    # Per-step logs
    rews: List[float] = []
    dists: List[float] = []
    bearings: List[float] = []
    v_cmds: List[float] = []
    w_cmds: List[float] = []
    ttcs: List[float] = []
    min_ultras: List[float] = []
    min_obs_depths: List[float] = []

    robot_traj_last: Optional[List[Tuple[float, float]]] = None

    # For progress
    dist_start = _safe_float(obs[0], default=np.nan)
    prev_dist = dist_start

    terminated = False
    truncated = False
    reason = ""

    # placeholders from info
    robot_start = None
    goal_start = None
    robot_final = None
    goal_final = None
    min_dist = None

    steps = 0
    while True:
        steps += 1
        if steps > max_steps_hard_cap:
            # hard safety stop (should not happen if time_limit works)
            truncated = True
            reason = "hard_cap"
            break

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)

        rews.append(float(reward))

        # Pull common fields from obs (your obs[0]=goal_dist, obs[1]=bearing)
        d = _safe_float(obs[0])
        b = _safe_float(obs[1])
        dists.append(d)
        bearings.append(b)

        # From info (your env provides these)
        v_cmds.append(_safe_float(info.get("v_cmd", np.nan)))
        w_cmds.append(_safe_float(info.get("w_cmd", np.nan)))
        ttcs.append(_safe_float(info.get("ttc", np.inf), default=np.inf))

        min_ultras.append(_safe_float(info.get("min_ultrasonic_distance", np.nan)))
        min_obs_depths.append(_safe_float(info.get("min_obstacle_depth", np.nan)))

        # Episode boundary
        if term or trunc:
            terminated = bool(term)
            truncated = bool(trunc)
            reason = str(info.get("reason", ""))

            robot_start = info.get("robot_start", None)
            goal_start = info.get("goal_start", None)
            robot_final = info.get("robot_final", None)
            goal_final = info.get("goal_final", None)
            min_dist = info.get("min_dist", None)

            # robot_traj is stored in env info at end as "robot_traj" (list of (x,y))
            robot_traj_last = info.get("robot_traj", None)
            break

        # progress tracking
        if np.isfinite(prev_dist) and np.isfinite(d):
            prev_dist = d

    duration = time.time() - t0
    ret = float(np.sum(rews)) if rews else 0.0
    length = int(len(rews))

    dists_arr = np.array(dists, dtype=float)
    v_arr = np.array(v_cmds, dtype=float)
    w_arr = np.array(w_cmds, dtype=float)
    ttc_arr = np.array(ttcs, dtype=float)
    min_ultra_arr = np.array(min_ultras, dtype=float)
    min_obs_arr = np.array(min_obs_depths, dtype=float)

    # progress_total: dist_start - dist_final (positive is good)
    dist_final = float(dists_arr[-1]) if dists_arr.size else np.nan
    progress_total = float(dist_start - dist_final) if (np.isfinite(dist_start) and np.isfinite(dist_final)) else np.nan

    # TTC fraction below 2s (danger zone)
    ttc_below = np.isfinite(ttc_arr) & (ttc_arr < 2.0)
    ttc_frac = float(np.mean(ttc_below)) if ttc_arr.size else np.nan

    ep = EpisodeResult(
        episode=ep_i,
        seed=seed,
        return_sum=ret,
        length_steps=length,
        duration_sec=float(duration),
        terminated=terminated,
        truncated=truncated,
        reason=reason,

        robot_start=tuple(robot_start) if isinstance(robot_start, (list, tuple)) else None,
        goal_start=tuple(goal_start) if isinstance(goal_start, (list, tuple)) else None,
        robot_final=tuple(robot_final) if isinstance(robot_final, (list, tuple)) else None,
        goal_final=tuple(goal_final) if isinstance(goal_final, (list, tuple)) else None,
        min_dist=_safe_float(min_dist, default=np.nan) if min_dist is not None else None,

        v_cmd_mean=float(np.nanmean(v_arr)) if v_arr.size else np.nan,
        v_cmd_max=float(np.nanmax(v_arr)) if v_arr.size else np.nan,
        w_cmd_mean_abs=float(np.nanmean(np.abs(w_arr))) if w_arr.size else np.nan,

        ttc_min=float(np.nanmin(ttc_arr)) if ttc_arr.size else np.inf,
        ttc_mean=float(np.nanmean(ttc_arr[np.isfinite(ttc_arr)])) if ttc_arr.size else np.nan,
        ttc_frac_below_2s=ttc_frac,

        min_ultra_min=float(np.nanmin(min_ultra_arr)) if min_ultra_arr.size else np.nan,
        min_obs_depth_min=float(np.nanmin(min_obs_arr)) if min_obs_arr.size else np.nan,

        progress_total=progress_total,
        dist_start=float(dist_start),
        dist_final=float(dist_final),

        robot_traj=[[float(x), float(y)] for (x, y) in robot_traj_last] if isinstance(robot_traj_last, list) else None,
    )

    raw = {
        "rewards": rews,
        "dists": dists,
        "bearings": bearings,
        "v_cmds": v_cmds,
        "w_cmds": w_cmds,
        "ttcs": ttcs,
        "min_ultras": min_ultras,
        "min_obs_depths": min_obs_depths,
    }
    return ep, raw


def save_plots(outdir: str, episodes: List[EpisodeResult]) -> None:
    os.makedirs(outdir, exist_ok=True)

    R = np.array([e.return_sum for e in episodes], dtype=float)
    L = np.array([e.length_steps for e in episodes], dtype=float)
    TTCmin = np.array([e.ttc_min for e in episodes], dtype=float)
    Vmean = np.array([e.v_cmd_mean for e in episodes], dtype=float)

    # Reward per episode
    plt.figure()
    plt.plot(R, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return (sum of rewards)")
    plt.title("Evaluation returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "returns.png"))
    plt.close()

    # Length per episode
    plt.figure()
    plt.plot(L, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode lengths")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lengths.png"))
    plt.close()

    # Safety: min TTC per episode
    plt.figure()
    plt.plot(TTCmin, marker="o")
    plt.axhline(2.0, linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Min TTC (s)")
    plt.title("Safety: minimum time-to-collision (lower is worse)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ttc_min.png"))
    plt.close()

    # Return vs mean speed
    plt.figure()
    plt.scatter(Vmean, R)
    plt.xlabel("Mean commanded v (m/s)")
    plt.ylabel("Return")
    plt.title("Return vs mean speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "return_vs_speed.png"))
    plt.close()


def write_report(outdir: str, episodes: List[EpisodeResult]) -> None:
    os.makedirs(outdir, exist_ok=True)

    reasons = [e.reason for e in episodes]
    success = sum(1 for r in reasons if r == "goal")
    collision = sum(1 for r in reasons if r == "collision")
    timeout = sum(1 for r in reasons if r == "timeout")
    hard_cap = sum(1 for r in reasons if r == "hard_cap")
    other = len(episodes) - (success + collision + timeout + hard_cap)

    R = np.array([e.return_sum for e in episodes], dtype=float)
    L = np.array([e.length_steps for e in episodes], dtype=float)
    P = np.array([e.progress_total for e in episodes], dtype=float)
    TTCmin = np.array([e.ttc_min for e in episodes], dtype=float)
    TTCfrac = np.array([e.ttc_frac_below_2s for e in episodes], dtype=float)
    Vmean = np.array([e.v_cmd_mean for e in episodes], dtype=float)
    Wabs = np.array([e.w_cmd_mean_abs for e in episodes], dtype=float)

    r_mean, r_ci = _mean_ci95(R)
    l_mean, l_ci = _mean_ci95(L)

    def corr(a, b) -> float:
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if np.sum(m) < 3:
            return np.nan
        return float(np.corrcoef(a[m], b[m])[0, 1])

    corr_R_L = corr(R, L)
    corr_R_TTC = corr(R, TTCmin)
    corr_R_V = corr(R, Vmean)
    corr_R_W = corr(R, Wabs)

    lines = []
    lines.append("# SAC Evaluation Report (20 Episodes)\n")
    lines.append("## Outcome summary\n")
    lines.append(f"- Episodes: {len(episodes)}\n")
    lines.append(f"- Success (goal): {success} ({success/len(episodes)*100:.1f}%)\n")
    lines.append(f"- Collision: {collision} ({collision/len(episodes)*100:.1f}%)\n")
    lines.append(f"- Timeout: {timeout} ({timeout/len(episodes)*100:.1f}%)\n")
    if hard_cap:
        lines.append(f"- Hard cap: {hard_cap} ({hard_cap/len(episodes)*100:.1f}%)\n")
    if other:
        lines.append(f"- Other: {other} ({other/len(episodes)*100:.1f}%)\n")

    lines.append("\n## Return & efficiency\n")
    lines.append(f"- Mean return: {r_mean:.3f} (± {r_ci:.3f} 95% CI)\n")
    lines.append(f"- Mean length: {l_mean:.1f} steps (± {l_ci:.1f} 95% CI)\n")
    lines.append(f"- Return ↔ Length corr: {corr_R_L:.3f}\n")

    lines.append("\n## Progress dynamics\n")
    lines.append(f"- Mean total progress (dist_start - dist_final): {float(np.nanmean(P)):.3f}\n")
    lines.append(f"- Median total progress: {float(np.nanmedian(P)):.3f}\n")

    lines.append("\n## Safety (TTC + proximity)\n")
    lines.append(f"- Mean(min TTC): {float(np.nanmean(TTCmin)):.3f} s\n")
    lines.append(f"- Median(min TTC): {float(np.nanmedian(TTCmin)):.3f} s\n")
    lines.append(f"- Mean fraction of steps TTC<2s: {float(np.nanmean(TTCfrac)):.3f}\n")
    lines.append(f"- Return ↔ minTTC corr: {corr_R_TTC:.3f} (positive means safer episodes score higher)\n")

    lines.append("\n## Control behavior\n")
    lines.append(f"- Mean v_cmd: {float(np.nanmean(Vmean)):.3f} m/s\n")
    lines.append(f"- Mean |w_cmd|: {float(np.nanmean(Wabs)):.3f} rad/s\n")
    lines.append(f"- Return ↔ mean speed corr: {corr_R_V:.3f}\n")
    lines.append(f"- Return ↔ mean turning corr: {corr_R_W:.3f}\n")

    lines.append("\n## Per-episode table\n")
    lines.append("| Ep | Reason | Return | Steps | MinDist | MinTTC | TTC<2s(frac) | v_mean | |w|_mean |\n")
    lines.append("|---:|:------:|------:|------:|--------:|------:|------------:|-------:|---------:|\n")
    for e in episodes:
        lines.append(
            f"| {e.episode:>2d} | {e.reason or '-':^6s} | {e.return_sum:>7.3f} | {e.length_steps:>5d} "
            f"| {(_safe_float(e.min_dist) if e.min_dist is not None else np.nan):>7.3f} "
            f"| {e.ttc_min:>6.3f} | {e.ttc_frac_below_2s:>12.3f} | {e.v_cmd_mean:>6.3f} | {e.w_cmd_mean_abs:>7.3f} |\n"
        )

    lines.append("\n## Interpretation notes (what these numbers usually mean)\n")
    lines.append(
        "- If success rate is low but progress_total is high, the agent moves toward the goal but fails late (often obstacle interaction / local minima).\n"
        "- Many collisions with very low minTTC and high TTC<2s fraction suggests the policy is aggressive and lacks braking/avoidance margin.\n"
        "- High |w_cmd| with mediocre progress suggests oscillation/zig-zag (can come from reward shaping, noisy obstacle features, or too-high entropy).\n"
        "- Timeouts with decent TTC but poor progress usually means the agent is indecisive or stuck in turning behavior.\n"
    )

    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to SAC .zip model")
    ap.add_argument("--replay", type=str, default="", help="Optional path to replay buffer .pkl")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="./eval_out")
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=12345)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ROS2 init (required before env construction)
    rclpy.init()

    try:
        env_raw = GoalFollowerEnv(
            cmd_topic="/follower_robot/cmd_vel",
            goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
            goal_odom_topic="/goal_marker/odom",
            dt=0.1,
            success_radius=1.5,
            time_limit=80.0,
            R_goal=1.0,
            R_collision=-1.0,
            gamma_shaping=0.97,
            step_penalty=0.01,
            ttc_threshold=2.0,
            cam_x=0.4,
            cam_y=0.0,
            use_ground_truth_geometry=True,
            smooth_alpha=0.20,
        )
        env = Monitor(env_raw)

        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model not found: {args.model}")

        model = SAC.load(args.model, env=env, device="auto")

        # Replay buffer is not needed for evaluation, but can be loaded to verify compatibility.
        if args.replay:
            if not os.path.exists(args.replay):
                raise FileNotFoundError(f"Replay buffer not found: {args.replay}")
            try:
                model.load_replay_buffer(args.replay)
                print(f"[OK] Loaded replay buffer: {args.replay}")
            except Exception as e:
                print(f"[WARN] Could not load replay buffer (continuing): {e}")

        deterministic = bool(args.deterministic)

        all_eps: List[EpisodeResult] = []
        all_raw: Dict[str, Any] = {"episodes": []}

        for i in range(1, args.episodes + 1):
            seed = args.seed0 + i
            print(f"\n=== Eval episode {i}/{args.episodes} (seed={seed}) ===")
            ep, raw = run_one_episode(env, model, i, seed, deterministic=deterministic)
            all_eps.append(ep)

            all_raw["episodes"].append(
                {
                    "summary": asdict(ep),
                    "raw": raw,
                }
            )

            print(
                f"[EP {i}] reason={ep.reason} return={ep.return_sum:.3f} "
                f"steps={ep.length_steps} minTTC={ep.ttc_min:.3f} "
                f"progress={ep.progress_total:.3f}"
            )

        # Save JSON outputs
        with open(os.path.join(args.outdir, "eval_full.json"), "w") as f:
            json.dump(all_raw, f, indent=2)

        with open(os.path.join(args.outdir, "eval_episode_summaries.json"), "w") as f:
            json.dump([asdict(e) for e in all_eps], f, indent=2)

        # Save plots + report
        save_plots(args.outdir, all_eps)
        write_report(args.outdir, all_eps)

        print(f"\n[DONE] Outputs saved to: {os.path.abspath(args.outdir)}")
        print(" - report.md")
        print(" - eval_episode_summaries.json")
        print(" - eval_full.json")
        print(" - returns.png / lengths.png / ttc_min.png / return_vs_speed.png")

    finally:
        try:
            env_raw.close()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
