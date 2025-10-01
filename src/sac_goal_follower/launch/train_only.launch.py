# sac_goal_follower/launch/train_only.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sac_goal_follower',
            executable='train_sac',
            output='screen'
        )
    ])