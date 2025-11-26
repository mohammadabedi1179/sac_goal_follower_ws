#!/usr/bin/env python3
import os
import json
import time

import numpy as np
import torch

import rclpy
from rclpy.node import Node

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from goal_env import GoalFollowerEnv


class EpisodeStatsCallback(BaseCallback):
    """
    Collects episode statistics and some training metrics and:

    - saves episode stats to JSON every `save_freq_episodes` episodes
    - saves the model every `model_save_every_episodes` episodes
    """

    def __init__(
        self,
        save_path: str,
        save_freq_episodes: int,
        model_save_dir: str | None = None,
        model_save_every_episodes: int = 0,
        model_name_prefix: str = "sac_goal_follower",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq_episodes = save_freq_episodes
        self.model_save_dir = model_save_dir
        self.model_save_every_episodes = model_save_every_episodes
        self.model_name_prefix = model_name_prefix

        self.episode_stats = []
        self._episode_counter = 0

        if self.model_save_dir is not None:
            os.makedirs(self.model_save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        log_vals = self.logger.name_to_value

        for done, info in zip(dones, infos):
            if done:
                self._episode_counter += 1
                ep_data = {
                    "episode": self._episode_counter,
                }

                # Monitor wrapper episode info
                if "episode" in info:
                    ep_info = info["episode"]
                    ep_data["reward"] = float(ep_info.get("r", 0.0))
                    ep_data["length"] = int(ep_info.get("l", 0))
                    ep_data["time"] = float(ep_info.get("t", 0.0))

                # Training metrics
                ep_data["actor_loss"] = float(log_vals.get("train/actor_loss", np.nan))
                ep_data["critic_loss"] = float(log_vals.get("train/critic_loss", np.nan))
                ep_data["ent_coef"] = float(log_vals.get("train/ent_coef", np.nan))
                ep_data["learning_rate"] = float(
                    log_vals.get("train/learning_rate", np.nan)
                )

                self.episode_stats.append(ep_data)

                if self.verbose > 0:
                    print(
                        f"[EpisodeStatsCallback] Ep {self._episode_counter}: "
                        f"R={ep_data['reward']:.2f}, L={ep_data['length']}, "
                        f"actor_loss={ep_data['actor_loss']:.3f}, "
                        f"critic_loss={ep_data['critic_loss']:.3f}"
                    )

                # ---- Save stats JSON every N episodes ----
                if (self._episode_counter % self.save_freq_episodes) == 0:
                    try:
                        with open(self.save_path, "w") as f:
                            json.dump(self.episode_stats, f, indent=2)
                        if self.verbose > 0:
                            print(
                                f"[EpisodeStatsCallback] Saved stats to {self.save_path}"
                            )
                    except Exception as e:
                        print(f"[EpisodeStatsCallback] Error saving stats: {e}")

                # ---- Save model every M episodes ----
                if (
                    self.model_save_dir is not None
                    and self.model_save_every_episodes > 0
                    and (self._episode_counter % self.model_save_every_episodes) == 0
                ):
                    model_file = os.path.join(
                        self.model_save_dir,
                        f"{self.model_name_prefix}_ep_{self._episode_counter}_steps_{self.num_timesteps}.zip",
                    )
                    try:
                        self.model.save(model_file)
                        if self.verbose > 0:
                            print(
                                f"[EpisodeStatsCallback] Saved model at episode "
                                f"{self._episode_counter} to {model_file}"
                            )
                    except Exception as e:
                        print(f"[EpisodeStatsCallback] Error saving model: {e}")

        return True


class SACTrainer(Node):
    def __init__(self):
        super().__init__("sac_trainer")

        # Create environment
        self.env = GoalFollowerEnv(
            cmd_topic="/follower_robot/cmd_vel",
            goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
            goal_odom_topic="/goal_marker/odom",
            robot_odom_topic="/follower_robot/odom",
            dt=0.1,
            lost_timeout=8.0,
            success_radius=1.5,
            time_limit=80.0,
            c_time=0.1,
            c_dist=0.1,
            c_lost=0.1,
            R_goal=120.0,
        )

        # Directories for models and logs
        base_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(base_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Logger for SB3 (stdout + CSV + TensorBoard)
        self.logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

    def train(self):
        env = Monitor(self.env)

        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.001,
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

        # Use our configured logger
        model.set_logger(self.logger)

        # Episode stats + model saving callback
        stats_path = os.path.join(self.log_dir, "episode_stats.json")
        stats_callback = EpisodeStatsCallback(
            save_path=stats_path,
            save_freq_episodes=10,          # <-- JSON every 10 episodes
            model_save_dir=self.model_dir,
            model_save_every_episodes=1000, # <-- model every 1000 episodes
            model_name_prefix="sac_goal_follower",
            verbose=1,
        )

        self.get_logger().info(
            f"Starting training with models at {self.model_dir} and logs at {self.log_dir}"
        )

        model.learn(
            total_timesteps=1000000,
            callback=stats_callback,
            log_interval=10,
        )

        final_path = os.path.join(self.model_dir, "sac_goal_follower_final")
        model.save(final_path)

        self.get_logger().info(
            f"Training completed and final model saved to {final_path}. "
            f"Episode stats JSON at {stats_path}"
        )

        self.env.close()


def main(args=None):
    rclpy.init(args=args)
    trainer = SACTrainer()
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.get_logger().info("Training interrupted by user")
    finally:
        trainer.env.close()
        trainer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
