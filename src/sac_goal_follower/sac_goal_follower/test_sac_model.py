#!/usr/bin/env python3
import os
import time
import json

import rclpy
from rclpy.node import Node

from stable_baselines3 import SAC

from goal_env import GoalFollowerEnv


class SACTester(Node):
    def __init__(self):
        super().__init__("sac_model_tester")

        # --- Parameters (override from CLI if you want) ---
        self.declare_parameter("model_path", "/home/mohammadabedi/Documents/Autonomous Scooter/Simulation/Gazebo/src/sac_goal_follower/sac_goal_follower/results/Static_goal_marker_random_position_2_static_obstacles_in_the_FOV_1M_steps/models/sac_goal_follower_best.zip")
        self.declare_parameter("episodes", 5)
        self.declare_parameter("deterministic", True)
        self.declare_parameter("sleep_after_reset_sec", 0.2)
        self.declare_parameter("save_rollouts", True)
        self.declare_parameter("rollout_out", "test_rollouts.json")

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.episodes = int(self.get_parameter("episodes").value)
        self.deterministic = bool(self.get_parameter("deterministic").value)
        self.sleep_after_reset_sec = float(self.get_parameter("sleep_after_reset_sec").value)
        self.save_rollouts = bool(self.get_parameter("save_rollouts").value)
        self.rollout_out = self.get_parameter("rollout_out").get_parameter_value().string_value

        # Resolve relative path based on script location (like your trainer does)
        base_dir = os.path.dirname(__file__)
        if not os.path.isabs(model_path):
            model_path = os.path.join(base_dir, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.get_logger().info(f"Loading SAC model from: {model_path}")

        # --- Create env exactly like training ---
        # These defaults match your trainer wiring. :contentReference[oaicite:3]{index=3}
        self.env = GoalFollowerEnv(
            cmd_topic="/follower_robot/cmd_vel",
            goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
            goal_odom_topic="/goal_marker/odom",
            robot_odom_topic="/follower_robot/odom",
            dt=0.1,
            lost_timeout=25.0,
            success_radius=1.5,
            time_limit=80.0,
            c_time=0.1,
            c_dist=0.3,
            c_lost=0.1,
            R_goal=50.0,
        )

        # --- Load model ---
        # IMPORTANT: do NOT pass env into load unless you want SB3 to wrap it;
        # we’ll just use model.predict(obs). This is typical for inference.
        self.model = SAC.load(model_path, device="auto")

        self.rollouts = []

    def run(self):
        self.get_logger().info(
            f"Starting test: episodes={self.episodes}, deterministic={self.deterministic}"
        )

        try:
            for ep in range(1, self.episodes + 1):
                obs, info = self.env.reset()
                time.sleep(self.sleep_after_reset_sec)

                done = False
                trunc = False
                ep_reward = 0.0
                ep_steps = 0

                ep_log = {
                    "episode": ep,
                    "reward_total": 0.0,
                    "steps": 0,
                    "terminated": False,
                    "truncated": False,
                    "final_reason": "",
                    "info_tail": {},
                }

                # Step loop
                while not (done or trunc):
                    # SB3 expects obs as np.array shaped (obs_dim,)
                    action, _state = self.model.predict(
                        obs, deterministic=self.deterministic
                    )

                    obs, reward, done, trunc, step_info = self.env.step(action)
                    ep_reward += float(reward)
                    ep_steps += 1

                    # Keep a small tail of useful info (optional)
                    ep_log["final_reason"] = step_info.get("reason", "")
                    ep_log["info_tail"] = {
                        "min_dist": step_info.get("min_dist", None),
                        "nearest_obstacle_depth": step_info.get("nearest_obstacle_depth", None),
                        "min_ultrasonic_distance": step_info.get("min_ultrasonic_distance", None),
                    }

                ep_log["reward_total"] = ep_reward
                ep_log["steps"] = ep_steps
                ep_log["terminated"] = bool(done)
                ep_log["truncated"] = bool(trunc)

                self.rollouts.append(ep_log)

                self.get_logger().info(
                    f"[TEST] Ep {ep}/{self.episodes} | "
                    f"R={ep_reward:.2f} | steps={ep_steps} | "
                    f"done={done} trunc={trunc} | reason={ep_log['final_reason']}"
                )

        finally:
            # Always stop robot
            try:
                self.env.ros.send_cmd(0.0, 0.0)
                self.env._spin(0.2)
            except Exception:
                pass

            if self.save_rollouts:
                out_path = self.rollout_out
                if not os.path.isabs(out_path):
                    out_path = os.path.join(os.path.dirname(__file__), out_path)
                try:
                    with open(out_path, "w") as f:
                        json.dump(self.rollouts, f, indent=2)
                    self.get_logger().info(f"Saved rollout summary to: {out_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to save rollout summary: {e}")

            self.env.close()
            self.get_logger().info("Test finished; env closed.")


def main(args=None):
    rclpy.init(args=args)
    node = SACTester()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
