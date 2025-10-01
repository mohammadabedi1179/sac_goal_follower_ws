# goal_marker_description/launch/spawn_goal_marker.launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import xacro

def generate_launch_description():
    pkg_path = os.path.join(get_package_share_directory('goal_marker_description'))
    xacro_file = os.path.join(pkg_path,'urdf','goal_marker.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    #xacro_path = PathJoinSubstitution([pkg, 'urdf', 'goal_marker.urdf.xacro'])

    robot_description = {'robot_description': robot_description_config.toxml()}

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description],
        output='screen',
        arguments=[]
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'goal_marker',
            '-x', '5.0', '-y', '5.0', '-z', '0.15'
        ],
        output='screen'
    )

    return LaunchDescription([rsp, spawn])