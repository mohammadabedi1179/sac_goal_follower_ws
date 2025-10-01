# sac_goal_follower/train_sac.py
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from sac_goal_follower.goal_env import GoalFollowerEnv
import os, site
site.addsitedir(os.path.expanduser('~/anaconda3/envs/sac_env/lib/python3.10/site-packages'))

def main():
    env = GoalFollowerEnv(
        cmd_topic="/cmd_vel",
        rgb_topic="/depth_cam/image_raw",
        depth_topic="/depth_cam/depth/image_raw",
        disp_topic="",                    # set if using stereo disparity
        goal_odom_topic="/goal_marker/odom",
        wheel_radius=0.10,
        wheel_separation=0.85,
        fov_deg=90.0,
        hsv=(35,85,80,80),
        min_blob_area=300,
        dt=0.10,
        lost_timeout=10.0,
        success_radius=0.35,
        time_limit=40.0,
        c_time=0.01, c_dist=0.5, c_lost=0.1, R_goal=50.0
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=512,
        train_freq=64,
        gradient_steps=64,
        gamma=0.99,
        tau=0.02,
        ent_coef="auto"
    )

    ckpt_dir = os.path.expanduser("~/sac_goal_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    callback = CheckpointCallback(save_freq=20_000, save_path=ckpt_dir, name_prefix="sac_goal")

    model.learn(total_timesteps=300_000, callback=callback)
    model.save(os.path.join(ckpt_dir, "sac_goal_final"))
    env.close()

if __name__ == "__main__":
    main()
