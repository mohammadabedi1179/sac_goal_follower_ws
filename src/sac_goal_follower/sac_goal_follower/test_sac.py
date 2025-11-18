# sac_goal_follower/test_sac.py

import os
import time

import rclpy
from stable_baselines3 import SAC

from goal_env import GoalFollowerEnv


def run_eval(
    model_path: str,
    n_episodes: int = 10,
    max_steps_per_episode: int = 500,
):
    # rclpy will be shut down inside env.close()
    rclpy.init()

    # Use the same env params you used during training
    env = GoalFollowerEnv(
        cmd_topic='/follower_robot/cmd_vel',
        goal_state_topic='/follower_robot/depth_cam/goal_marker_state',
        goal_odom_topic='/goal_marker/odom',
        robot_odom_topic='/follower_robot/odom',
        dt=0.1,
        lost_timeout=5.0,
        success_radius=2.0,
        time_limit=50.0,
        c_time=0.01,
        c_dist=0.1,
        c_lost=0.1,
        R_goal=50.0,
    )

    # Load SAC model
    model = SAC.load(model_path, env=env)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ep_return = 0.0
        step = 0

        env.ros.get_logger().info(
            f"===== EVAL EPISODE {ep+1}/{n_episodes} ====="
        )

        while not (done or trunc) and step < max_steps_per_episode:
            # deterministic=True -> no exploration noise
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)

            ep_return += reward
            step += 1

        reason = info.get("reason", "")
        env.ros.get_logger().info(
            f"[EVAL] Episode {ep+1} finished: "
            f"steps={step}, return={ep_return:.2f}, reason={reason}"
        )

        # small pause between episodes (optional)
        time.sleep(1.0)

    env.close()   # this sends 0 cmd, destroys node, and calls rclpy.shutdown()

def main():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(
        base_dir,
        "models",
        "sac_goal_follower_245700_steps.zip",  # <-- your file name
    )
    run_eval(model_path, n_episodes=10)


if __name__ == "__main__":
    main()