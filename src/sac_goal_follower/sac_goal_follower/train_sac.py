# sac_goal_follower/train_sac.py

import numpy as np
import torch
import os
import time
import json   # <-- NEW

import rclpy
from rclpy.node import Node

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

from goal_env import GoalFollowerEnv


class JsonLoggerCallback(BaseCallback):
    """
    Custom callback to save training statistics to a JSON file
    every N episodes.

    It records:
      - episode number
      - total timesteps
      - episode reward and length (from Monitor's 'episode' info)
      - the latest logger values (e.g. rollout/ep_rew_mean, train/actor_loss, ...)
    """

    def __init__(self, json_path: str, save_every_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.json_path = json_path
        self.save_every_episodes = save_every_episodes
        self.episode_count = 0
        self.data = []

        # If file already exists, load it so we append instead of overwrite history
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                # If file is corrupted or not JSON, start fresh
                self.data = []

    def _on_step(self) -> bool:
        # 'infos' comes from the VecEnv; each time an episode ends,
        # Monitor puts an 'episode' dict inside info.
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                ep_info = info["episode"]

                if self.episode_count % self.save_every_episodes == 0:
                    # Collect the latest values from SB3 logger if available
                    log_dict = {}
                    if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                        # This contains entries like 'rollout/ep_len_mean', 'train/critic_loss', ...
                        log_dict = dict(self.model.logger.name_to_value)

                    # Add episode-specific info
                    log_dict.update(
                        {
                            "episode": int(self.episode_count),
                            "total_timesteps": int(self.num_timesteps),
                            "episode_reward": float(ep_info.get("r", 0.0)),
                            "episode_length": int(ep_info.get("l", 0)),
                        }
                    )

                    self.data.append(log_dict)

                    # Save the whole list as a JSON array
                    try:
                        with open(self.json_path, "w") as f:
                            json.dump(self.data, f, indent=2)
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"[JsonLoggerCallback] Error writing JSON file: {e}")

                    if self.verbose > 0:
                        print(
                            f"[JsonLoggerCallback] Saved stats at episode "
                            f"{self.episode_count} to {self.json_path}"
                        )

        return True


class SACTrainer(Node):
    def __init__(self):
        super().__init__('sac_trainer')

        self.env = GoalFollowerEnv(
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

        base_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(base_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

    def train(self):
        env = Monitor(self.env)

        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.0003,
            buffer_size=100000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            tensorboard_log=self.log_dir,
            policy_kwargs=None,
            verbose=1,
            seed=None,
            device="auto",
            _init_setup_model=True,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=100,
            save_path=self.model_dir,
            name_prefix="sac_goal_follower",
            save_replay_buffer=True,
            save_vecnormalize=False,
        )

        # NEW: JSON logger callback (every 10 episodes)
        json_log_path = os.path.join(self.log_dir, "episode_stats.json")
        json_logger_callback = JsonLoggerCallback(
            json_path=json_log_path,
            save_every_episodes=10,
            verbose=1,
        )

        self.get_logger().info(
            f"Starting training with models at {self.model_dir} and logs at {self.log_dir}"
        )

        # You can pass a list of callbacks directly
        model.learn(
            total_timesteps=500000,
            callback=[checkpoint_callback, json_logger_callback],
            log_interval=10,
        )
        model.save(os.path.join(self.model_dir, "sac_goal_follower_final"))

        self.get_logger().info("Training completed and model saved")


def main(args=None):
    rclpy.init(args=args)
    trainer = SACTrainer()
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.get_logger().info("Training interrupted by user")
    finally:
        # Close env (which destroys its ROS node + executor)
        trainer.env.close()
        # Destroy the trainer node itself
        trainer.destroy_node()
        # Now shut down the global ROS client library
        rclpy.shutdown()



if __name__ == "__main__":
    main()
