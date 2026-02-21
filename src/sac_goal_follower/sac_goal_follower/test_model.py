#!/usr/bin/env python3
"""
SAC Model Evaluator — Test a trained model over N episodes.

Loads a saved SAC model (default: best model) and runs deterministic evaluation
episodes. Logs per-episode stats and prints a summary report.

Usage:
  ros2 run your_package test_model
  ros2 run your_package test_model --ros-args \
      -p model_name:="sac_goal_follower_best.zip" \
      -p num_episodes:=50 \
      -p deterministic:=true

The script saves results to:
  logs/test_results.json         — per-episode stats (same format as training)
  logs/test_report.txt           — human-readable summary report
"""

import os
import sys
import json
import time
import math
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node

from stable_baselines3 import SAC
from goal_env_gt import GoalFollowerEnv


class SACTester(Node):
    def __init__(self):
        super().__init__("sac_tester")

        # ── Parameters ──
        self.declare_parameter("model_name", "sac_goal_follower_best.zip")
        self.declare_parameter("num_episodes", 50)
        self.declare_parameter("deterministic", True)
        self.declare_parameter("time_limit", 80.0)

        self.model_name = str(self.get_parameter("model_name").value)
        self.num_episodes = int(self.get_parameter("num_episodes").value)
        self.deterministic = bool(self.get_parameter("deterministic").value)
        self.time_limit = float(self.get_parameter("time_limit").value)

        # ── Environment (same config as training) ──
        self.env = GoalFollowerEnv(
            cmd_topic="/follower_robot/cmd_vel",
            goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
            goal_odom_topic="/goal_marker/odom",
            dt=0.1,
            success_radius=1.5,
            time_limit=self.time_limit,
            R_goal=50.0,
            R_collision=-50.0,
            gamma_shaping=0.97,
            step_penalty=0.005,
            ttc_threshold=2.0,
            cam_x=0.4,
            cam_y=0.0,
            use_ground_truth_geometry=True,
            smooth_alpha=0.70,
        )

        # ── Paths ──
        base_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(base_dir, "results")
        self.log_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def _load_model(self):
        model_path = self.model_name
        if not os.path.isabs(model_path):
            model_path = os.path.join(self.model_dir, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = SAC.load(model_path, env=self.env, device="auto")
        self.get_logger().info(f"[TEST] Loaded model: {model_path}")
        self.get_logger().info(f"[TEST] Deterministic: {self.deterministic}")
        self.get_logger().info(f"[TEST] Episodes to run: {self.num_episodes}")
        return model

    def _dist(self, a, b):
        if a is None or b is None:
            return float("nan")
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def run_test(self):
        model = self._load_model()
        results = []

        total_start = time.time()

        for ep in range(1, self.num_episodes + 1):
            obs, _info = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            ep_start = time.time()

            # Per-step tracking
            robot_path = []
            goal_path = []
            gt_obstacles_path = []
            min_ultra_ep = float("inf")
            min_obs_ep = float("inf")
            shield_interventions = 0
            shield_levels_count = Counter()
            ttc_values = []

            while not done:
                action, _states = model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                ep_reward += reward
                ep_steps += 1

                # Track per-step data
                rp = info.get("step_robot_pose", None)
                if rp is not None and len(rp) == 3:
                    robot_path.append([float(rp[0]), float(rp[1]), float(rp[2])])

                gp = info.get("step_goal_pose", None)
                if gp is not None and len(gp) == 2:
                    goal_path.append([float(gp[0]), float(gp[1])])

                gt_obs = info.get("step_gt_obstacles_world", None)
                if gt_obs is not None and isinstance(gt_obs, dict):
                    frame = {}
                    for k, v in gt_obs.items():
                        if v is not None:
                            frame[str(k)] = [float(v[0]), float(v[1])]
                    gt_obstacles_path.append(frame)

                # Safety stats
                mu = info.get("min_ultrasonic_distance", float("inf"))
                mo = info.get("min_obstacle_depth", float("inf"))
                if mu < min_ultra_ep:
                    min_ultra_ep = mu
                if mo < min_obs_ep:
                    min_obs_ep = mo

                sl = info.get("shield_level", "none")
                shield_levels_count[sl] += 1
                if sl != "none":
                    shield_interventions += 1

                ttc_val = info.get("ttc", None)
                if ttc_val is not None and np.isfinite(ttc_val):
                    ttc_values.append(float(ttc_val))

            ep_time = time.time() - ep_start

            # Build episode record
            ep_data = {
                "episode": ep,
                "reward": float(ep_reward),
                "length": ep_steps,
                "time": float(ep_time),
                "reason": info.get("reason", "unknown"),
                "robot_start": info.get("robot_start", None),
                "goal_start": info.get("goal_start", None),
                "robot_final": info.get("robot_final", None),
                "goal_final": info.get("goal_final", None),
                "min_dist": info.get("min_dist", None),
                "min_obstacle_depth": float(min_obs_ep) if np.isfinite(min_obs_ep) else None,
                "min_ultrasonic_distance": float(min_ultra_ep) if np.isfinite(min_ultra_ep) else None,
                "ttc": info.get("ttc", None),
                "v_cmd": info.get("v_cmd", None),
                "w_cmd": info.get("w_cmd", None),
                "v_policy": info.get("v_policy", None),
                "w_policy": info.get("w_policy", None),
                "shield_level": info.get("shield_level", None),
                "shield_interventions_ep": shield_interventions,
                "shield_levels_count": dict(shield_levels_count),
                "obstacles_world_slots": info.get("obstacles_world_slots", []),
                "start_goal_distance": self._dist(info.get("robot_start"), info.get("goal_start")),
                # Trajectories for every episode (it's only 50)
                "robot_path": robot_path,
                "goal_path": goal_path,
                "gt_obstacles_path": gt_obstacles_path,
                # Per-episode TTC stats
                "ttc_min": float(min(ttc_values)) if ttc_values else None,
                "ttc_mean": float(np.mean(ttc_values)) if ttc_values else None,
            }

            results.append(ep_data)

            reason = ep_data["reason"]
            self.get_logger().info(
                f"[TEST] Ep {ep}/{self.num_episodes}: "
                f"R={ep_reward:.2f} | len={ep_steps} | {reason} | "
                f"shield={shield_interventions} | "
                f"min_obs={min_obs_ep:.2f} | min_ultra={min_ultra_ep:.2f} | "
                f"time={ep_time:.1f}s"
            )

        total_time = time.time() - total_start

        # ── Save results ──
        results_path = os.path.join(self.log_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.get_logger().info(f"[TEST] Saved results to {results_path}")

        # ── Generate report ──
        report = self._generate_report(results, total_time)
        report_path = os.path.join(self.log_dir, "test_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        self.get_logger().info(f"[TEST] Saved report to {report_path}")

        # Print to console
        print("\n" + report)

        return results

    def _generate_report(self, results, total_time):
        n = len(results)
        reasons = [r["reason"] for r in results]
        cnt = Counter(reasons)

        rewards = np.array([r["reward"] for r in results])
        lengths = np.array([r["length"] for r in results])
        times = np.array([r["time"] for r in results])

        successes = sum(1 for r in reasons if r == "goal")
        collisions = sum(1 for r in reasons if r == "collision")
        timeouts = sum(1 for r in reasons if r == "timeout")

        shield_counts = [r.get("shield_interventions_ep", 0) for r in results]
        shield_rates = [r.get("shield_interventions_ep", 0) / max(r["length"], 1) for r in results]

        min_obs_vals = [r["min_obstacle_depth"] for r in results if r.get("min_obstacle_depth") is not None]
        min_ultra_vals = [r["min_ultrasonic_distance"] for r in results if r.get("min_ultrasonic_distance") is not None]

        start_goal_dists = [r.get("start_goal_distance", float("nan")) for r in results]
        start_goal_dists = [d for d in start_goal_dists if np.isfinite(d)]

        # Min dist to goal (episode-level)
        min_dists = [r["min_dist"] for r in results if r.get("min_dist") is not None]

        # Reward by reason
        reward_by_reason = {}
        for r in results:
            reason = r["reason"]
            if reason not in reward_by_reason:
                reward_by_reason[reason] = []
            reward_by_reason[reason].append(r["reward"])

        lines = []
        lines.append("=" * 60)
        lines.append("  SAC MODEL EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Model: {self.model_name}")
        lines.append(f"Episodes: {n}")
        lines.append(f"Deterministic: {self.deterministic}")
        lines.append(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        lines.append("")

        lines.append("── OUTCOMES ──")
        lines.append(f"  Success (goal):     {successes:>4d}  ({100*successes/n:.1f}%)")
        lines.append(f"  Collision:          {collisions:>4d}  ({100*collisions/n:.1f}%)")
        lines.append(f"  Timeout:            {timeouts:>4d}  ({100*timeouts/n:.1f}%)")
        lines.append("")

        lines.append("── REWARD ──")
        lines.append(f"  Mean:   {np.mean(rewards):.2f}")
        lines.append(f"  Std:    {np.std(rewards):.2f}")
        lines.append(f"  Median: {np.median(rewards):.2f}")
        lines.append(f"  Min:    {np.min(rewards):.2f}")
        lines.append(f"  Max:    {np.max(rewards):.2f}")
        lines.append("")

        for reason in sorted(reward_by_reason.keys()):
            vals = reward_by_reason[reason]
            lines.append(f"  Reward ({reason}): mean={np.mean(vals):.2f} | std={np.std(vals):.2f} | n={len(vals)}")
        lines.append("")

        lines.append("── EPISODE LENGTH ──")
        lines.append(f"  Mean:   {np.mean(lengths):.1f} steps")
        lines.append(f"  Median: {np.median(lengths):.1f} steps")
        lines.append(f"  Min:    {np.min(lengths)} | Max: {np.max(lengths)}")
        lines.append("")

        lines.append("── NAVIGATION EFFICIENCY ──")
        if min_dists:
            lines.append(f"  min_dist to goal: mean={np.mean(min_dists):.2f}m | median={np.median(min_dists):.2f}m")
        if start_goal_dists:
            lines.append(f"  Start-goal dist:  mean={np.mean(start_goal_dists):.2f}m")
        # Success episodes: steps per meter
        goal_eps = [r for r in results if r["reason"] == "goal"]
        if goal_eps:
            efficiencies = []
            for r in goal_eps:
                d = r.get("start_goal_distance", float("nan"))
                if np.isfinite(d) and d > 0:
                    efficiencies.append(r["length"] / d)
            if efficiencies:
                lines.append(f"  Steps/meter (successes): mean={np.mean(efficiencies):.1f} | median={np.median(efficiencies):.1f}")
        lines.append("")

        lines.append("── SAFETY ──")
        if min_obs_vals:
            lines.append(f"  min_obstacle_depth:      mean={np.mean(min_obs_vals):.3f}m | min={np.min(min_obs_vals):.3f}m")
        if min_ultra_vals:
            lines.append(f"  min_ultrasonic_distance: mean={np.mean(min_ultra_vals):.3f}m | min={np.min(min_ultra_vals):.3f}m")
        lines.append(f"  Shield interventions/ep: mean={np.mean(shield_counts):.1f} | max={np.max(shield_counts)}")
        lines.append(f"  Shield rate (frac):      mean={np.mean(shield_rates):.3f} | max={np.max(shield_rates):.3f}")
        lines.append("")

        lines.append("── PER-EPISODE DETAILS ──")
        for r in results:
            shld = r.get("shield_interventions_ep", 0)
            min_o = r.get("min_obstacle_depth")
            min_u = r.get("min_ultrasonic_distance")
            lines.append(
                f"  ep={r['episode']:>3d} | R={r['reward']:>8.2f} | len={r['length']:>4d} | "
                f"{r['reason']:<10s} | shield={shld:>3d} | "
                f"min_obs={min_o if min_o is not None else 'N/A':>6} | "
                f"min_ultra={min_u if min_u is not None else 'N/A':>6}"
            )
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def main(args=None):
    rclpy.init(args=args)
    tester = SACTester()

    try:
        tester.run_test()
    except KeyboardInterrupt:
        tester.get_logger().info("[TEST] Interrupted by user.")
    except Exception as e:
        tester.get_logger().error(f"[TEST] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.env.close()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()