# sac_goal_follower/train_sac.py
import numpy as np, torch, os, time
import rclpy
from rclpy.node import Node
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from goal_env import GoalFollowerEnv

class SACTrainer(Node):
    def __init__(self):
        super().__init__('sac_trainer')
        self.env = GoalFollowerEnv(
            cmd_topic='/cmd_vel',
            left_rgb_topic='/left_camera/image_raw',
            right_rgb_topic='/right_camera/image_raw',
            goal_odom_topic='/goal_marker/odom',
            robot_odom_topic='/odom',
            min_blob_area=300,
            dt=0.1,
            lost_timeout=10.0,
            success_radius=0.35,
            time_limit=40.0,
            c_time=0.01,
            c_dist=0.1,
            c_lost=0.1,
            R_goal=50.0
        )
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_dir = os.path.join(os.path.dirname(__file__), 'logs')
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
            ent_coef='auto',
            target_update_interval=1,
            target_entropy='auto',
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            tensorboard_log=self.log_dir,
            policy_kwargs=None,
            verbose=1,
            seed=None,
            device='auto',
            _init_setup_model=True
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=self.model_dir,
            name_prefix='sac_goal_follower',
            save_replay_buffer=True,
            save_vecnormalize=False
        )
        self.get_logger().info(f"Starting training with {self.model_dir} and {self.log_dir}")
        model.learn(total_timesteps=100000, callback=checkpoint_callback, log_interval=10)
        model.save(os.path.join(self.model_dir, 'sac_goal_follower_final'))
        self.get_logger().info("Training completed and model saved")
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
