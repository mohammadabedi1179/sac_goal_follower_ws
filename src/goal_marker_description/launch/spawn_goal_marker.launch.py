import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, LogInfo
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro

def generate_launch_description():
    try:
        pkg_share = FindPackageShare('goal_marker_description').find('goal_marker_description')
        sdf_file = os.path.join(pkg_share, 'models', 'goal_marker', 'model.sdf')
        if not os.path.exists(sdf_file):
            raise FileNotFoundError(f"SDF file not found: {sdf_file}")
        print(f"Using SDF file: {sdf_file}")

        # Set environment variables for Gazebo to include the installed model path
        set_model_path = SetEnvironmentVariable(
            'GAZEBO_MODEL_PATH',
            f"{os.path.join(pkg_share, 'models')}:/usr/share/gazebo-11/models"
        )
        set_resource_path = SetEnvironmentVariable(
            'GAZEBO_RESOURCE_PATH',
            f"{os.path.join(pkg_share, 'models')}:/usr/share/gazebo-11"
        )
        set_ogre_log = SetEnvironmentVariable('OGRE_LOG_DEST', '/home/mohammadabedi/.gazebo/ogre.log')
        set_ogre_verbose = SetEnvironmentVariable('OGRE_VERBOSE', 'true')

        # Optional: Robot State Publisher for URDF-based kinematics
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': xacro.process_file(os.path.join(pkg_share, 'models', 'goal_marker', 'urdf', 'goal_marker.urdf.xacro')).toxml()}],
            output='screen'
        )

        # Spawn entity using SDF, connecting to the existing Gazebo instance
        spawn_entity = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-file', sdf_file, '-entity', 'goal_marker', '-x', '5.0', '-y', '5.0', '-z', '0.15'],
            output='screen'
        )

        return LaunchDescription([
            set_model_path,
            set_resource_path,
            set_ogre_log,
            set_ogre_verbose,
            LogInfo(msg=f"Spawning from SDF file: {sdf_file}"),
            rsp,
            spawn_entity
        ])

    except Exception as e:
        print(f"Error processing launch file: {e}")
        raise