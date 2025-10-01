from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('four_wheel_robot')
    goal_sdf = os.path.join(pkg_share, 'models', 'goal_marker', 'model.sdf')

    spawn_goal = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_goal',
        arguments=[
            '-entity', 'goal_marker',
            '-file', goal_sdf,
            '-x', '20.0', '-y', '25.0', '-z', '0.0'
        ],
        output='screen'
    )

    return LaunchDescription([spawn_goal])