#!/usr/bin/env python3
"""
LSTM-SAC Training Script

This script trains an LSTM-SAC agent on the goal following task.
The main differences from the standard SAC training:
1. Uses GoalFollowerLSTMEnv (returns dict observations)
2. Uses LSTMSACPolicy (processes obstacles with LSTM)
3. All 6 obstacles are used (vs top-3 in the original)
"""

import os
import json
import re

import numpy as np
import torch

import rclpy
from rclpy.node import Node

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from goal_env_lstm import GoalFollowerLSTMEnv
from lstm_sac_policy import create_lstm_sac_policy


_MODEL_NAME_RE = re.compile(r".*?_ep_(\d+)_steps_(\d+)\.zip$")


def _parse_ep_steps_from_model_name(name: str):
    m = _MODEL_NAME_RE.match(os.path.basename(name))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _safe_load_episode_stats(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_replay_buffer_if_possible(model: SAC, path: str, verbose: int = 1) -> None:
    try:
        model.save_replay_buffer(path)
        if verbose > 0:
            print(f"[ReplayBuffer] Saved replay buffer to {path}")
    except AttributeError:
        if verbose > 0:
            print("[ReplayBuffer] model.save_replay_buffer not available in this SB3 version; skipping.")
    except Exception as e:
        print(f"[ReplayBuffer] Error saving replay buffer: {e}")


def _lr_schedule(progress_remaining: float) -> float:
    """Linear decay from 3e-4 to 1.5e-4 over training."""
    return 3e-4 * max(0.5, progress_remaining)


class EpisodeStatsCallback(BaseCallback):
    """
    Callback for logging episode statistics and saving models periodically.
    Same as the original training script.
    """
    def __init__(
        self,
        save_path: str,
        save_freq_episodes: int,
        model_save_dir: str | None = None,
        model_save_every_episodes: int = 0,
        model_name_prefix: str = "lstm_sac_goal_follower",
        best_model_path: str | None = None,
        best_metric_key: str = "reward",
        traj_save_every_episodes: int = 25,
        verbose: int = 1,
        last_model_path: str | None = None,
        last_replay_buffer_path: str | None = None,
        existing_stats: list | None = None,
        episode_start_offset: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq_episodes = int(save_freq_episodes)
        self.model_save_dir = model_save_dir
        self.model_save_every_episodes = int(model_save_every_episodes)
        self.model_name_prefix = model_name_prefix

        self.best_model_path = best_model_path
        self.best_metric_key = best_metric_key
        self.traj_save_every_episodes = int(traj_save_every_episodes)

        self.episode_stats = existing_stats[:] if isinstance(existing_stats, list) else []
        self._episode_counter = int(episode_start_offset)

        self.best_metric_value = -np.inf
        if self.episode_stats and self.best_metric_key:
            vals = []
            for e in self.episode_stats:
                v = e.get(self.best_metric_key, None)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.append(float(v))
            if vals:
                self.best_metric_value = float(np.max(vals))

        if self.model_save_dir is not None:
            os.makedirs(self.model_save_dir, exist_ok=True)

        self._step_buffers = None
        self.last_model_path = last_model_path
        self.last_replay_buffer_path = last_replay_buffer_path

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._step_buffers = []
        for _ in range(n_envs):
            self._step_buffers.append(
                {
                    "robot_path": [],
                    "goal_path": [],
                    "gt_obstacles_path": [],
                    "seen_obstacles_local": [],
                    "seen_obstacles_world": [],
                }
            )

    @staticmethod
    def _to_jsonable_gt_obstacles(gt_dict):
        if gt_dict is None or not isinstance(gt_dict, dict):
            return None
        out = {}
        for k, v in gt_dict.items():
            if v is None:
                out[str(k)] = None
            else:
                out[str(k)] = [float(v[0]), float(v[1])]
        return out

    def _clear_env_buffer(self, env_i: int):
        if self._step_buffers is None:
            return
        self._step_buffers[env_i] = {
            "robot_path": [],
            "goal_path": [],
            "gt_obstacles_path": [],
            "seen_obstacles_local": [],
            "seen_obstacles_world": [],
        }

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        log_vals = self.logger.name_to_value
        n_envs = len(dones)

        # Accumulate per-step data
        if self._step_buffers is not None:
            for env_i in range(n_envs):
                info = infos[env_i]
                buf = self._step_buffers[env_i]

                rp = info.get("step_robot_pose", None)
                if rp is not None and len(rp) == 3:
                    buf["robot_path"].append([float(rp[0]), float(rp[1]), float(rp[2])])

                gp = info.get("step_goal_pose", None)
                if gp is not None and len(gp) == 2:
                    buf["goal_path"].append([float(gp[0]), float(gp[1])])

                gt_obs = self._to_jsonable_gt_obstacles(info.get("step_gt_obstacles_world", None))
                if gt_obs is not None:
                    buf["gt_obstacles_path"].append(gt_obs)

                seen_local = info.get("step_seen_obstacles_local", None)
                if seen_local is not None:
                    buf["seen_obstacles_local"].append(
                        [[float(a), float(b), float(c)] for (a, b, c) in seen_local]
                    )

                seen_world = info.get("step_seen_obstacles_world", None)
                if seen_world is not None:
                    buf["seen_obstacles_world"].append(
                        [[float(a), float(b), float(c)] for (a, b, c) in seen_world]
                    )

        # Episode ends
        for env_i in range(n_envs):
            if not dones[env_i]:
                continue

            info = infos[env_i]
            self._episode_counter += 1

            ep_data = {"episode": self._episode_counter}

            if "episode" in info:
                ep_info = info["episode"]
                ep_data["reward"] = float(ep_info.get("r", 0.0))
                ep_data["length"] = int(ep_info.get("l", 0))
                ep_data["time"] = float(ep_info.get("t", 0.0))

            ep_data["actor_loss"] = float(log_vals.get("train/actor_loss", np.nan))
            ep_data["critic_loss"] = float(log_vals.get("train/critic_loss", np.nan))
            ep_data["ent_coef"] = float(log_vals.get("train/ent_coef", np.nan))
            ep_data["learning_rate"] = float(log_vals.get("train/learning_rate", np.nan))

            for key in ["reason", "robot_start", "goal_start", "robot_final", "goal_final", "min_dist"]:
                if key in info:
                    ep_data[key] = info[key]

            for key in ["ttc", "v_cmd", "w_cmd", "min_obstacle_depth", "min_ultrasonic_distance"]:
                if key in info:
                    ep_data[key] = float(info[key])

            for key in ["shield_level", "shield_interventions_ep"]:
                if key in info:
                    ep_data[key] = info[key]

            # Save trajectory every N episodes
            if self.traj_save_every_episodes > 0 and (self._episode_counter % self.traj_save_every_episodes == 0):
                if self._step_buffers is not None and env_i < len(self._step_buffers):
                    buf = self._step_buffers[env_i]
                    ep_data["robot_path"] = buf["robot_path"]
                    ep_data["goal_path"] = buf["goal_path"]
                    ep_data["gt_obstacles_path"] = buf["gt_obstacles_path"]
                    ep_data["seen_obstacles_local"] = buf["seen_obstacles_local"]
                    ep_data["seen_obstacles_world"] = buf["seen_obstacles_world"]

            self.episode_stats.append(ep_data)

            # Save episode stats
            if self._episode_counter % self.save_freq_episodes == 0:
                try:
                    with open(self.save_path, "w") as f:
                        json.dump(self.episode_stats, f, indent=2)
                except Exception as e:
                    print(f"[EpisodeStatsCallback] Error saving episode stats: {e}")

            # Save periodic checkpoint
            if self.model_save_dir and self.model_save_every_episodes > 0:
                if self._episode_counter % self.model_save_every_episodes == 0:
                    model_name = (
                        f"{self.model_name_prefix}_ep_{self._episode_counter}_"
                        f"steps_{self.model.num_timesteps}.zip"
                    )
                    chk_path = os.path.join(self.model_save_dir, model_name)
                    try:
                        self.model.save(chk_path)
                        if self.verbose > 0:
                            print(f"[EpisodeStatsCallback] Checkpoint saved at episode {self._episode_counter}")
                        
                        chk_replay = chk_path.replace(".zip", "_replay.pkl")
                        _save_replay_buffer_if_possible(self.model, chk_replay, verbose=0)
                    except Exception as e:
                        print(f"[EpisodeStatsCallback] Error saving checkpoint: {e}")

            # Save last model
            if self.last_model_path is not None:
                try:
                    self.model.save(self.last_model_path)
                    if self.last_replay_buffer_path:
                        _save_replay_buffer_if_possible(self.model, self.last_replay_buffer_path, verbose=0)
                except Exception as e:
                    print(f"[EpisodeStatsCallback] Error saving last model: {e}")

            # Save best model
            if self.best_model_path and self.best_metric_key:
                metric_val = ep_data.get(self.best_metric_key, None)
                if isinstance(metric_val, (int, float)) and np.isfinite(metric_val):
                    if metric_val > self.best_metric_value:
                        self.best_metric_value = metric_val
                        try:
                            self.model.save(self.best_model_path)
                            if self.verbose > 0:
                                print(
                                    f"[EpisodeStatsCallback] New best {self.best_metric_key}={metric_val:.3f} "
                                    f"at episode {self._episode_counter}; saved best model"
                                )
                        except Exception as e:
                            print(f"[EpisodeStatsCallback] Error saving best model: {e}")

                        best_replay = self.best_model_path.replace(".zip", "_replay.pkl")
                        _save_replay_buffer_if_possible(self.model, best_replay, verbose=self.verbose)

            self._clear_env_buffer(env_i)

        return True


class LSTMSACTrainer(Node):
    """
    ROS2 Node for training LSTM-SAC on the goal following task.
    """
    def __init__(self):
        super().__init__("lstm_sac_trainer")

        self.declare_parameter("resume", False)
        self.declare_parameter("model_name", "")
        self.declare_parameter("lstm_hidden_size", 64)
        self.declare_parameter("lstm_num_layers", 1)
        self.declare_parameter("net_arch", "256,256")  # Comma-separated

        self.resume = bool(self.get_parameter("resume").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.lstm_hidden_size = int(self.get_parameter("lstm_hidden_size").value)
        self.lstm_num_layers = int(self.get_parameter("lstm_num_layers").value)
        
        # Parse net_arch
        net_arch_str = str(self.get_parameter("net_arch").value)
        self.net_arch = [int(x.strip()) for x in net_arch_str.split(",")]

        self.get_logger().info(
            f"LSTM Configuration:\n"
            f"  - LSTM hidden size: {self.lstm_hidden_size}\n"
            f"  - LSTM layers: {self.lstm_num_layers}\n"
            f"  - Network architecture: {self.net_arch}"
        )

        # Create environment
        self.env = GoalFollowerLSTMEnv(
            cmd_topic="/follower_robot/cmd_vel",
            goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
            goal_odom_topic="/goal_marker/odom",
            dt=0.1,
            success_radius=1.5,
            time_limit=80.0,
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

        base_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(base_dir, "models_lstm")
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_dir = os.path.join(base_dir, "logs_lstm")
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

    def train(self):
        env = Monitor(self.env)

        stats_path = os.path.join(self.log_dir, "episode_stats.json")
        best_model_path = os.path.join(self.model_dir, "lstm_sac_goal_follower_best.zip")
        last_model_path = os.path.join(self.model_dir, "lstm_sac_goal_follower_last.zip")
        last_replay_path = os.path.join(self.model_dir, "lstm_sac_goal_follower_last_replay.pkl")

        existing_stats = []
        episode_offset = 0

        if self.resume:
            existing_stats = _safe_load_episode_stats(stats_path)
            if existing_stats:
                try:
                    episode_offset = int(max(int(e.get("episode", 0)) for e in existing_stats))
                except Exception:
                    episode_offset = 0

            ep_from_name, steps_from_name = _parse_ep_steps_from_model_name(self.model_name) if self.model_name else (None, None)
            self.get_logger().info(
                f"[RESUME] episode_stats loaded: {len(existing_stats)} episodes, last_episode={episode_offset}\n"
                f"[RESUME] model_name parsed: ep={ep_from_name}, steps={steps_from_name}\n"
                f"[RESUME] Will continue from episode {episode_offset + 1}."
            )

        # Create custom LSTM-SAC policy
        PolicyClass = create_lstm_sac_policy(
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            net_arch=self.net_arch,
        )

        if self.resume:
            if not self.model_name:
                raise RuntimeError("resume:=true but model_name is empty. Provide model_name parameter.")

            model_path = self.model_name
            if not os.path.isabs(model_path):
                model_path = os.path.join(self.model_dir, model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Resume model not found: {model_path}")

            model = SAC.load(model_path, env=env, device="auto")
            model.set_logger(self.logger)
            reset_num_timesteps = False

            # Load replay buffer
            replay_path_candidates = [
                model_path.replace(".zip", "_replay.pkl"),
                last_replay_path,
            ]
            replay_loaded = False
            for rp in replay_path_candidates:
                if os.path.exists(rp):
                    try:
                        model.load_replay_buffer(rp)
                        replay_loaded = True
                        self.get_logger().info(
                            f"[RESUME] Loaded replay buffer from {rp} "
                            f"({model.replay_buffer.size()} transitions)"
                        )
                        break
                    except Exception as e:
                        self.get_logger().warn(f"[RESUME] Failed to load replay buffer {rp}: {e}")

            if not replay_loaded:
                self.get_logger().warn(
                    "[RESUME] No replay buffer found — starting with empty buffer."
                )

            self.get_logger().info(
                f"[RESUME] Loaded model: {model_path}\n"
                f"[RESUME] SB3 model.num_timesteps={model.num_timesteps}\n"
                f"[RESUME] reset_num_timesteps={reset_num_timesteps}\n"
                f"[RESUME] replay_buffer_loaded={replay_loaded}"
            )
        else:
            # Create new model with LSTM policy
            model = SAC(
                policy=PolicyClass,
                env=env,
                learning_rate=_lr_schedule,
                buffer_size=1_000_000,
                batch_size=256,
                tau=0.005,
                gamma=0.97,
                train_freq=1,
                gradient_steps=1,
                learning_starts=5000,
                ent_coef="auto",
                target_entropy=-1.5,
                verbose=1,
                tensorboard_log=self.log_dir,
                device="auto",
            )
            model.set_logger(self.logger)
            reset_num_timesteps = True

        stats_callback = EpisodeStatsCallback(
            save_path=stats_path,
            save_freq_episodes=1,
            model_save_dir=self.model_dir,
            model_save_every_episodes=100,
            model_name_prefix="lstm_sac_goal_follower",
            best_model_path=best_model_path,
            best_metric_key="reward",
            traj_save_every_episodes=25,
            verbose=1,
            existing_stats=existing_stats if self.resume else None,
            episode_start_offset=episode_offset if self.resume else 0,
            last_model_path=last_model_path,
            last_replay_buffer_path=last_replay_path,
        )

        # Log observation space info
        fixed_dim = env.observation_space.spaces['fixed'].shape[0]
        obs_dim = env.observation_space.spaces['obstacles'].shape
        
        self.get_logger().info(
            f"Starting LSTM-SAC training:\n"
            f"  Models: {self.model_dir}\n"
            f"  Logs: {self.log_dir}\n"
            f"  Resume: {self.resume}\n"
            f"  Model name: {self.model_name}\n"
            f"  Observation space:\n"
            f"    - Fixed features: {fixed_dim}D\n"
            f"    - Obstacles: {obs_dim}\n"
            f"  Action space: {env.action_space.shape}\n"
            f"  LSTM hidden: {self.lstm_hidden_size}\n"
            f"  LSTM layers: {self.lstm_num_layers}\n"
            f"  Network arch: {self.net_arch}"
        )

        model.learn(
            total_timesteps=1_000_000,
            callback=stats_callback,
            log_interval=10,
            reset_num_timesteps=reset_num_timesteps,
        )

        final_path = os.path.join(self.model_dir, "lstm_sac_goal_follower_final.zip")
        final_replay = os.path.join(self.model_dir, "lstm_sac_goal_follower_final_replay.pkl")
        model.save(final_path)
        _save_replay_buffer_if_possible(model, final_replay, verbose=1)

        self.get_logger().info(
            f"Training completed and final model saved to {final_path}. "
            f"Episode stats JSON at {stats_path}"
        )
        self.env.close()


def main(args=None):
    rclpy.init(args=args)
    trainer = LSTMSACTrainer()
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
