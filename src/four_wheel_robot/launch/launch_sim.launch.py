import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # robot_state_publisher
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("four_wheel_robot"),
                         "launch", "rsp.launch.py")
        ]),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

      # Gazebo Classic with ONLY the factory system plugin + empty world
    world = os.path.join(
    get_package_share_directory("four_wheel_robot"),
    "worlds",
    "empty_with_state.world",
)

    gazebo = ExecuteProcess(
        cmd=[
            "gazebo", "--verbose",
            "-s", "libgazebo_ros_factory.so",   # keep factory
            world                               # load world that contains the state WORLD plugin
        ],
        output="screen",
        respawn=True,
    )



    # Spawn robot from robot_description
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description",
                   "-entity", "my_robot",
                   "-x", "0.0", "-y", "0.0", "-z", "0.30", "-Y", "0.7854"],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([rsp, gazebo, spawn_entity])